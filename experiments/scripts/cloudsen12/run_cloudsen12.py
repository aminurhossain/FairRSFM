#!/usr/bin/env python3
"""
Universal CloudSEN12 Benchmark & Mitigation Pipeline
===================================================
A unified entry point for Foundation Models (TerraFM, Prithvi, SatMAE, DOFA)
and Fairness Mitigation Methods (ERM, Group DRO, DBR, BOLP) on the
Stratified 14-Biome CloudSEN12 Benchmark (4-class semantic segmentation).

Usage:
  python run_cloudsen12.py --model <terrafm|prithvi|satmae|dofa> --method <erm|groupdro|dbr|bolp> --mode <train|eval>
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Environment & Proj Setup
os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

import pyproj.datadir
proj_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"]  = proj_dir

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from fairrsfm.data.cloudsen12_datamodule import (
    CloudSEN12SegmentationDataModule,
    NUM_CLASSES,
    RESOLVE_BIOME_NAMES,
)
from fairrsfm.models import (
    build_cloudsen12_model,
    get_cloudsen12_band_indices,
)
from fairrsfm.evaluation.unified_cloudsen12_evaluator import (
    evaluate_dataset_confusion_matrix,
    evaluate_cloudsen12_pipeline,
    CloudSEN12EpochValTestEvalCallback,
)

DEFAULT_CHECKPOINTS = {
    "terrafm": REPO_DIR / "checkpoints/terrafm/TerraFM-B.pth",
    "dofa":    REPO_DIR / "checkpoints/dofa/dofa_base_patch16_224.pth",
    "satmae":  REPO_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth",
    "prithvi": REPO_DIR / "checkpoints/Prithvi_EO_V2_300M_TL.pt",
}


# ── Method Lightning Modules ──────────────────────────────────────────────────

class CloudSEN12ERMModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.val_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["mask"]
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["mask"]
        groups = batch["group"]
        logits = self(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.val_step_outputs.append({
            "val_loss": loss.detach(),
            "preds": preds.detach().cpu(),
            "targets": targets.detach().cpu(),
            "groups": groups.detach().cpu(),
        })
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
        self.log("val/loss", avg_loss, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class CloudSEN12GroupDROModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        group_lr: float = 0.01,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.group_lr = group_lr
        self.max_epochs = max_epochs
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.register_buffer("q", torch.ones(14) / 14.0)
        self.val_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["mask"]
        groups = batch["group"]

        logits = self(images)
        pixel_losses = self.criterion(logits, targets)
        sample_losses = pixel_losses.mean(dim=[1, 2])

        group_losses = torch.zeros(14, device=self.device)
        group_counts = torch.zeros(14, device=self.device)
        for g in range(14):
            mask_g = (groups == g)
            if mask_g.any():
                group_losses[g] = sample_losses[mask_g].mean()
                group_counts[g] = mask_g.sum()

        active_groups = group_counts > 0
        if active_groups.any():
            self.q[active_groups] *= torch.exp(self.group_lr * group_losses[active_groups].detach())
            self.q /= self.q.sum()

        dro_loss = torch.sum(self.q * group_losses)
        self.log("train/loss", dro_loss, on_step=True, on_epoch=True, prog_bar=True)
        return dro_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["mask"]
        logits = self(images)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        self.val_step_outputs.append({"val_loss": loss.detach()})
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
        self.log("val/loss", avg_loss, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class CloudSEN12DBRModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        group_counts: Dict[int, int],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 0.5,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

        weights = np.zeros(14, dtype=np.float32)
        for g in range(14):
            count = group_counts.get(g, 1)
            weights[g] = 1.0 / (max(1, count) ** gamma)
        weights = weights / weights.mean()
        self.register_buffer("group_weights", torch.tensor(weights, dtype=torch.float32))
        self.val_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["mask"]
        groups = batch["group"]

        logits = self(images)
        pixel_losses = self.criterion(logits, targets)
        sample_losses = pixel_losses.mean(dim=[1, 2])

        sample_weights = self.group_weights[groups]
        dbr_loss = (sample_losses * sample_weights).mean()
        self.log("train/loss", dbr_loss, on_step=True, on_epoch=True, prog_bar=True)
        return dbr_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["mask"]
        logits = self(images)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        self.val_step_outputs.append({"val_loss": loss.detach()})
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
        self.log("val/loss", avg_loss, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class CloudSEN12BOLPModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha: float = 0.5,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.val_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["mask"]
        groups = batch["group"]

        logits = self(images)
        pixel_losses = self.criterion(logits, targets)
        sample_losses = pixel_losses.mean(dim=[1, 2])

        erm_loss = sample_losses.mean()
        active_group_losses = []
        for g in range(14):
            mask_g = (groups == g)
            if mask_g.any():
                active_group_losses.append(sample_losses[mask_g].mean())

        worst_loss = torch.stack(active_group_losses).max() if active_group_losses else erm_loss
        total_loss = (1.0 - self.alpha) * erm_loss + self.alpha * worst_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["mask"]
        logits = self(images)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        self.val_step_outputs.append({"val_loss": loss.detach()})
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
        self.log("val/loss", avg_loss, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Universal CloudSEN12 Benchmark Pipeline")
    p.add_argument("--model", choices=["terrafm", "prithvi", "satmae", "dofa"], default="terrafm",
                   help="Foundation model backbone")
    p.add_argument("--method", choices=["erm", "groupdro", "dbr", "bolp"], default="erm",
                   help="Fairness optimization / mitigation method")
    p.add_argument("--mode", choices=["train", "eval"], default="train",
                   help="Run mode: train from scratch or evaluate existing checkpoint")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to pretrained model checkpoint or trained evaluation checkpoint")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (defaults to Final_cloudsen12/<model>/<method>/)")
    p.add_argument("--gpu", type=str, default="0",
                   help="CUDA GPU device ID (e.g. 0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of epochs (default 50, no early stopping)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for AdamW")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay for AdamW")
    p.add_argument("--group-lr", type=float, default=0.01,
                   help="Step size for Group DRO exponentiated gradient ascent")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="DBR smoothing exponent gamma")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="BOLP convex combination weight alpha")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker count")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_dir = Path(args.output_dir) if args.output_dir else REPO_DIR / f"Final_cloudsen12/{args.model}/{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f" UNIVERSAL CLOUDSEN12 BENCHMARK")
    print(f" Model       : {args.model.upper()}")
    print(f" Method      : {args.method.upper()}")
    print(f" Mode        : {args.mode.upper()}")
    print(f" Output Dir  : {out_dir}")
    print(f" Device      : {device}")
    print("=" * 80 + "\n")

    band_indices = get_cloudsen12_band_indices(args.model)
    print(f"[{args.model.upper()}] CloudSEN12 selecting {len(band_indices)} bands: {band_indices}")

    dm = CloudSEN12SegmentationDataModule(
        band_indices=band_indices,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    dm.setup()

    backbone_ckpt = args.checkpoint or str(DEFAULT_CHECKPOINTS.get(args.model, ""))

    if args.mode == "train":
        L.seed_everything(args.seed, workers=True)

        model_core = build_cloudsen12_model(
            model_name=args.model,
            num_classes=NUM_CLASSES,
            checkpoint_path=backbone_ckpt,
            freeze_backbone=True,
        )

        if args.method == "erm":
            model = CloudSEN12ERMModule(
                model=model_core,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )
        elif args.method == "groupdro":
            model = CloudSEN12GroupDROModule(
                model=model_core,
                lr=args.lr,
                group_lr=args.group_lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )
        elif args.method == "dbr":
            train_groups = [dm.train_dataset.biome_map[idx] - 1 for idx in dm.train_dataset.indices]
            group_counts = {g: train_groups.count(g) for g in range(14)}
            model = CloudSEN12DBRModule(
                model=model_core,
                group_counts=group_counts,
                lr=args.lr,
                gamma=args.gamma,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )
        elif args.method == "bolp":
            model = CloudSEN12BOLPModule(
                model=model_core,
                lr=args.lr,
                alpha=args.alpha,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )

        ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir / "checkpoints"),
            filename="checkpoint-{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        lr_cb = LearningRateMonitor(logging_interval="epoch")
        prog_cb = TQDMProgressBar(refresh_rate=20)
        logger = TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")

        eval_cb = CloudSEN12EpochValTestEvalCallback(
            val_loader=dm.val_dataloader(),
            test_loader=dm.test_dataloader(),
            out_dir=out_dir,
            start_epoch=0,
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=args.epochs,
            callbacks=[ckpt_cb, lr_cb, prog_cb, eval_cb],
            logger=logger,
            log_every_n_steps=10,
        )

        print(f"[Training] Launching {args.epochs} epochs of {args.model.upper()} ({args.method.upper()}) on CloudSEN12...")
        trainer.fit(model, datamodule=dm)

        best_candidate = out_dir / "checkpoints/best_val_worst_miou.ckpt"
        best_ckpt = str(best_candidate) if best_candidate.exists() else (ckpt_cb.best_model_path or str(out_dir / "checkpoints/last.ckpt"))
        print(f"[Evaluation] Selected best checkpoint: {best_ckpt}")
        eval_model = model.model
    else:
        eval_ckpt = args.checkpoint or str(out_dir / "checkpoints/best_val_worst_miou.ckpt")
        print(f"[Evaluation] Loading evaluation checkpoint: {eval_ckpt}")
        model_core = build_cloudsen12_model(
            model_name=args.model,
            num_classes=NUM_CLASSES,
            checkpoint_path=backbone_ckpt,
            freeze_backbone=True,
        )
        ckpt_dict = torch.load(eval_ckpt, map_location="cpu")
        state_dict = ckpt_dict.get("state_dict", ckpt_dict)
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}
        model_core.load_state_dict(clean_state_dict if clean_state_dict else state_dict, strict=False)
        eval_model = model_core

    eval_model.to(device)
    eval_model.eval()

    print("\n>>> Computing Final Test Confusion Matrix on Selected Checkpoint...")
    test_loader = dm.test_dataloader()
    conf_mat = evaluate_dataset_confusion_matrix(eval_model, test_loader, device=device)

    print("\n>>> Generating Canonical Evaluation Artifacts & Fairness Metrics...")
    evaluate_cloudsen12_pipeline(
        conf_mat=conf_mat,
        output_dir=out_dir,
        model_name=args.model,
        method_name=args.method,
    )
    print(f"\n [DONE] {args.model.upper()} ({args.method.upper()}) CloudSEN12 Pipeline Complete!")


if __name__ == "__main__":
    main()
