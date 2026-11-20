#!/usr/bin/env python3
"""
TerraFM ViT-Base — CloudSEN12 Dense Cloud Segmentation (ERM UPerNet)
===================================================================
Standard Empirical Risk Minimization (ERM) on frozen TerraFM backbone
with a 4-channel UPerNet segmentation head across 14 RESOLVE biomes.
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

REPO_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_DIR))
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
from fairrsfm.models.terrafm_wrapper import TerraFMUPerNetSegmentationClassifier
from fairrsfm.evaluation.unified_cloudsen12_evaluator import (
    evaluate_dataset_confusion_matrix,
    evaluate_cloudsen12_pipeline,
    CloudSEN12EpochValTestEvalCallback,
)

DEFAULT_CHECKPOINT = REPO_DIR / "checkpoints/terrafm/TerraFM-B.pth"
DEFAULT_OUTPUT_DIR = REPO_DIR / "Final_cloudsen12/terrafm/erm"


class CloudSEN12ERMModule(L.LightningModule):
    def __init__(
        self,
        model_size: str = "base",
        checkpoint_path: str = str(DEFAULT_CHECKPOINT),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        eval_epoch_start: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.eval_epoch_start = eval_epoch_start

        self.model = TerraFMUPerNetSegmentationClassifier(
            num_classes=NUM_CLASSES,
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            fpn_dim=256,
        )
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
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.val_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.val_step_outputs], dim=0)
        all_groups = torch.cat([x["groups"] for x in self.val_step_outputs], dim=0)
        self.val_step_outputs.clear()

        # Compute per-biome mIoU on validation split
        group_mious = []
        for g in range(14):
            mask_g = (all_groups == g)
            if not mask_g.any():
                continue
            p_g = all_preds[mask_g].view(-1)
            t_g = all_targets[mask_g].view(-1)

            valid = (t_g >= 0) & (t_g < NUM_CLASSES)
            if not valid.any():
                continue
            p_v = p_g[valid]
            t_v = t_g[valid]

            ious = []
            for c in range(NUM_CLASSES):
                tp = ((p_v == c) & (t_v == c)).sum().item()
                fp = ((p_v == c) & (t_v != c)).sum().item()
                fn = ((p_v != c) & (t_v == c)).sum().item()
                u = tp + fp + fn
                if u > 0:
                    ious.append(tp / u)
            if ious:
                group_mious.append(np.mean(ious))

        worst_miou = min(group_mious) * 100.0 if group_mious else 0.0
        mean_miou = np.mean(group_mious) * 100.0 if group_mious else 0.0

        self.log("val/worst_miou", worst_miou, prog_bar=True)
        self.log("val/mean_miou", mean_miou, prog_bar=True)

    def configure_optimizers(self):
        # Only optimize trainable parameters (decoder head)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def main():
    parser = argparse.ArgumentParser(description="TerraFM ERM on CloudSEN12")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", "--gpu", dest="device", type=int, default=0)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting TerraFM ERM on CloudSEN12 (Seed: {args.seed}) ===")
    dm = CloudSEN12SegmentationDataModule(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    dm.setup()

    model = CloudSEN12ERMModule(
        checkpoint_path=args.checkpoint,
        lr=args.lr,
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
        devices=[args.device],
        max_epochs=args.epochs,
        callbacks=[ckpt_cb, lr_cb, prog_cb, eval_cb],
        logger=logger,
        log_every_n_steps=10,
    )

    print(f"[Training] Running {args.epochs} epochs with cosine annealing (No early stopping)...")
    trainer.fit(model, datamodule=dm)

    # Load best checkpoint for full publication evaluation
    best_candidate = out_dir / "checkpoints/best_val_worst_miou.ckpt"
    if best_candidate.exists():
        best_ckpt = str(best_candidate)
    else:
        best_ckpt = ckpt_cb.best_model_path or str(out_dir / "checkpoints/last.ckpt")
    print(f"[Evaluation] Loading best checkpoint: {best_ckpt}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = dm.test_dataloader()
    print("[Evaluation] Computing 14-Biome 4x4 Confusion Matrix on Test Split...")
    conf_mat = evaluate_dataset_confusion_matrix(model.model, test_loader, device=device)

    print(f"[Artifacts] Generating publication metrics into {out_dir}...")
    results = evaluate_cloudsen12_pipeline(
        conf_mat=conf_mat,
        output_dir=out_dir,
        model_name="terrafm",
        method_name="erm",
    )

    print("=== [DONE] TerraFM ERM Training and Evaluation Complete! ===")
    print(f"Test mIoU: {results['global_metrics']['mIoU']*100:.2f}% | Worst Biome: {results['fairness_metrics']['M_worst']*100:.2f}% | PER: {results['fairness_metrics']['PER']:.4f}")


if __name__ == "__main__":
    main()
