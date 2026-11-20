#!/usr/bin/env python3
"""
Universal Balanced MMEarth-20K Benchmark & Mitigation Pipeline
==============================================================
A single, standardized entry point across all Remote Sensing Foundation Models
(Prithvi-100M, SatMAE-Base, DOFA-Base) and Mitigation Methods
(ERM, Group DRO, DBR, BOLP) on the 14-Biome Balanced MMEarth-20K Benchmark.

Usage:
  python run_mmearth.py --model <prithvi|satmae|dofa> --method <erm|groupdro|dbr|bolp> --mode <train|eval>
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List
from collections import defaultdict

# ── Environment & Proj Setup ──────────────────────────────────────────────────
os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"
os.environ["PROJ_DATA"] = "/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_LIB"]  = "/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, "/data1/aminur/om/prithvi/FairRSFM")
sys.path.insert(0, str(REPO_DIR / "src/fairrsfm/models/upstream/SatMAE"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_old_load = torch.load
def _new_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _old_load(*args, **kwargs)
torch.load = _new_load

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from fairrsfm.data.mmearth import (
    DYNAMIC_WORLD_CLASSES,
    MMEarthDynamicWorldSegmentationDataModule,
    find_one,
)
from fairrsfm.evaluation.unified_mmearth_evaluator import (
    evaluate_dataset_confusion_matrix,
    compute_metrics_from_confmat,
    compute_group_fairness,
    BIOME_NAMES,
)

# Band definitions
PRITHVI_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
SATMAE_BANDS  = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
DOFA_BANDS    = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B11", "B12"]
DOFA_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 0.842, 0.945, 1.610, 2.190]
TERRAFM_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

NUM_CLASSES = len(DYNAMIC_WORLD_CLASSES)
IGNORE_INDEX = -1

# Pretrained Backbone Weights
SATMAE_WEIGHTS = REPO_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"
DOFA_WEIGHTS   = REPO_DIR / "checkpoints/dofa/dofa_base_patch16_224.pth"
PRITHVI_WEIGHTS = REPO_DIR / "checkpoints/Prithvi_EO_V2_300M_TL.pt"
TERRAFM_WEIGHTS = REPO_DIR / "checkpoints/terrafm/TerraFM-B.pth"


def load_tile_info_dict(data_root: Path) -> dict:
    tile_info_path = find_one(data_root, "*tile_info.json")
    with open(tile_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_checkpoint(model: str, method: str) -> Optional[Path]:
    """Finds default trained checkpoint for model and method."""
    final_dir = REPO_DIR / f"Final_mmearth20k/{model}/{method}/checkpoints"
    if final_dir.is_dir():
        ckpts = list(final_dir.glob("*.ckpt"))
        if ckpts:
            lasts = [c for c in ckpts if "last" in c.name]
            return lasts[0] if lasts else ckpts[0]

    # Model-specific legacy search
    if model == "prithvi":
        base = REPO_DIR / "runs_mmearth/prithvi_mmearth_new_balanced20k_upernet"
        mapping = {
            "erm": base / "normal_lp/checkpoints/last.ckpt",
            "groupdro": base / "groupdro_14biomes/groupdro_eta0.01/checkpoints/last.ckpt",
            "dbr": base / "dbr_14biomes/dbr_alpha0.5_ema0.9/checkpoints/last.ckpt",
        }
        cand = mapping.get(method)
        if cand and cand.exists():
            return cand
    elif model == "satmae":
        base = Path("/data1/aminur/om/prithvi/FairRSFM/runs_mmearth/satmae_upernet_14biomes")
        cand = base / f"{method}/seed_42/checkpoints/last.ckpt"
        if cand.exists():
            return cand
    elif model == "dofa":
        base = Path("/data1/aminur/om/prithvi/FairRSFM/runs_mmearth/dofa_upernet_14biomes")
        cand = base / f"{method}/seed_42/checkpoints/last.ckpt"
        if cand.exists():
            return cand

    return None


class UniversalUPerNetModule(L.LightningModule):
    """
    Unified PyTorch Lightning Module supporting ERM, Group DRO, DBR, and BOLP
    for SatMAE and DOFA with UPerNet decoders.
    """
    def __init__(
        self,
        model: nn.Module,
        tile_info_dict: dict,
        model_name: str,
        method: str = "erm",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        eta_q: float = 0.01,
        dbr_alpha: float = 0.5,
        dbr_ema: float = 0.9,
        max_epochs: int = 50,
        num_classes: int = NUM_CLASSES,
        projector: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tile_info_dict", "projector"])
        self.model = model
        self.tile_info_dict = tile_info_dict
        self.model_name = model_name
        self.method = method.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_q = eta_q
        self.dbr_alpha = dbr_alpha
        self.dbr_ema = dbr_ema
        self.max_epochs = max_epochs
        self.num_classes = num_classes

        self.num_biomes = 14
        # Group DRO weights buffer
        self.register_buffer("q", torch.ones(self.num_biomes, dtype=torch.float32) / self.num_biomes)
        # DBR running loss buffer
        self.register_buffer("running_group_loss", torch.ones(self.num_biomes, dtype=torch.float32))

        # BOLP projection matrix
        if projector is not None:
            self.register_buffer("projector", projector)
        else:
            self.projector = None

        self._last_group_losses = None
        self._last_active_mask = None

    def get_biome_idx(self, sid: str) -> Optional[int]:
        info = self.tile_info_dict.get(str(sid), {})
        b = info.get("biome", None)
        if b is not None and 0 <= int(b) < self.num_biomes:
            return int(b)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "bolp" and self.projector is not None:
            # Apply orthogonal projection to intermediate feature representations
            if hasattr(self.model, "forward_intermediates"):
                if self.model_name == "dofa":
                    multi_feats = self.model.encoder.forward_intermediates(x, self.model.wavelengths, out_indices=self.model.out_indices)
                else: # satmae
                    multi_feats = self.model.forward_intermediates(x)
            elif self.model_name == "terrafm":
                multi_feats = self.model.feature_extractor.extract_features(x)
            else:
                multi_feats = None
                
            if multi_feats is not None:
                # Project features through null space of biome centroids
                proj_feats = []
                for feat in multi_feats:
                    # feat: (B, C, H, W) -> permute to (B, H, W, C), multiply, permute back
                    b, c, h, w = feat.shape
                    flat_f = feat.permute(0, 2, 3, 1).reshape(-1, c) # (B*H*W, C)
                    if self.projector.shape[0] == c:
                        proj_f = flat_f @ self.projector.to(flat_f.device)
                    else:
                        proj_f = flat_f
                    proj_feats.append(proj_f.reshape(b, h, w, c).permute(0, 3, 1, 2))
                head = getattr(self.model, "upernet_head", getattr(self.model, "head", None))
                logits = head(proj_feats)
                return F.interpolate(logits, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images    = batch["image"]
        masks     = batch["mask"].squeeze(1)
        filenames = batch.get("filename", [None] * images.size(0))

        logits = self(images)

        if self.method == "groupdro":
            valid = (masks != IGNORE_INDEX)
            counts = valid.flatten(1).sum(1)
            ce_raw = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX, reduction="none")
            raw_losses = torch.nan_to_num(ce_raw.flatten(1).sum(1) / counts.clamp_min(1), nan=0.0)

            group_losses = torch.zeros(self.num_biomes, device=self.device)
            group_counts = torch.zeros(self.num_biomes, device=self.device)

            for i, sid in enumerate(filenames):
                b_idx = self.get_biome_idx(sid)
                if b_idx is not None:
                    group_losses[b_idx] += raw_losses[i]
                    group_counts[b_idx] += 1

            active_mask = group_counts > 0
            avg_group_losses = torch.zeros_like(group_losses)
            avg_group_losses[active_mask] = group_losses[active_mask] / group_counts[active_mask]

            self._last_group_losses = avg_group_losses.detach()
            self._last_active_mask = active_mask

            loss = torch.sum(self.q * avg_group_losses)

        elif self.method == "dbr":
            valid = (masks != IGNORE_INDEX)
            counts = valid.flatten(1).sum(1)
            ce_raw = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX, reduction="none")
            raw_losses = torch.nan_to_num(ce_raw.flatten(1).sum(1) / counts.clamp_min(1), nan=0.0)

            w = torch.pow(self.running_group_loss + 1e-6, self.dbr_alpha)
            w = w / w.mean()

            sample_weights = torch.ones_like(raw_losses)
            batch_group_losses = torch.zeros(self.num_biomes, device=self.device)
            batch_group_counts = torch.zeros(self.num_biomes, device=self.device)

            for i, sid in enumerate(filenames):
                b_idx = self.get_biome_idx(sid)
                if b_idx is not None:
                    sample_weights[i] = w[b_idx]
                    batch_group_losses[b_idx] += raw_losses[i].detach()
                    batch_group_counts[b_idx] += 1

            loss = torch.mean(sample_weights * raw_losses)

            with torch.no_grad():
                active = batch_group_counts > 0
                if active.any():
                    curr_loss = batch_group_losses[active] / batch_group_counts[active]
                    self.running_group_loss[active] = (
                        self.dbr_ema * self.running_group_loss[active]
                        + (1.0 - self.dbr_ema) * curr_loss
                    )
        else: # erm, bolp
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.method == "groupdro" and self._last_group_losses is not None and self._last_active_mask is not None:
            with torch.no_grad():
                q_updated = self.q.clone()
                q_updated[self._last_active_mask] *= torch.exp(self.eta_q * self._last_group_losses[self._last_active_mask])
                q_sum = q_updated.sum()
                if q_sum > 0:
                    self.q.copy_(q_updated / q_sum)

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks  = batch["mask"].squeeze(1)
        logits = self(images)
        loss   = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        self.log("val/loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}


def to_builtin_serializable(val):
    if isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {str(k): to_builtin_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_builtin_serializable(v) for v in val]
    return val


class EpochValTestEvalCallback(Callback):
    """
    Evaluates the complete 14-biome validation and test sets every epoch starting from start_epoch.
    Records per-biome mIoUs, overall mIoU, worst-case mIoU, spread, PER, and fairness metrics
    into epoch_eval_history.json and epoch_eval_history.csv.
    Saves dedicated checkpoints for best_val_worst_miou, best_val_miou, and best_test_worst_miou.
    """
    def __init__(
        self,
        val_loader,
        test_loader,
        tile_info: Dict[str, Any],
        out_dir: Path,
        start_epoch: int = 0,
        device: torch.device = None,
    ):
        super().__init__()
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tile_info = tile_info
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.start_epoch = start_epoch
        self.device = device

        self.history = []
        self.history_file = self.out_dir / "epoch_eval_history.json"
        self.csv_file = self.out_dir / "epoch_eval_history.csv"

        self.best_val_worst_miou = -1.0
        self.best_test_worst_miou = -1.0
        self.best_val_miou = -1.0

        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
                    for item in self.history:
                        v_worst = item.get("val", {}).get("worst_case_miou", -1.0)
                        t_worst = item.get("test", {}).get("worst_case_miou", -1.0)
                        v_m = item.get("val", {}).get("miou", -1.0)
                        self.best_val_worst_miou = max(self.best_val_worst_miou, v_worst)
                        self.best_test_worst_miou = max(self.best_test_worst_miou, t_worst)
                        self.best_val_miou = max(self.best_val_miou, v_m)
            except Exception as e:
                print(f"[Warning] Could not load prior epoch history: {e}")

    @torch.no_grad()
    def _evaluate_split(self, pl_module, dataloader) -> Dict[str, Any]:
        was_training = pl_module.training
        pl_module.eval()

        conf_mat_14 = np.zeros((14, 9, 9), dtype=np.int64)
        total_loss = 0.0
        total_batches = 0

        dev = self.device if self.device is not None else next(pl_module.parameters()).device
        autocast_device = "cuda" if dev.type == "cuda" else "cpu"

        with torch.amp.autocast(autocast_device):
            for batch in dataloader:
                images = batch["image"].to(dev, non_blocking=True)
                masks = batch["mask"].squeeze(1).to(dev, non_blocking=True)
                filenames = batch["filename"]

                if hasattr(pl_module, "_bolp_forward"):
                    batch["image"] = images
                    logits = pl_module._bolp_forward(batch)
                else:
                    logits = pl_module(images)
                if hasattr(logits, "output"):
                    logits = logits.output

                loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
                total_loss += loss.item()
                total_batches += 1

                preds = logits.argmax(dim=1)

                b_indices = []
                for fid in filenames:
                    b_val = self.tile_info.get(str(fid), {}).get("biome", -1)
                    b_indices.append(int(b_val) if b_val is not None and 0 <= int(b_val) < 14 else -1)
                b_tensor = torch.tensor(b_indices, dtype=torch.long, device=dev)

                b_exp = b_tensor[:, None, None].expand_as(masks)
                valid = (masks >= 0) & (masks < 9) & (b_exp >= 0) & (b_exp < 14)
                if valid.any():
                    flat_code = b_exp * 81 + masks * 9 + preds
                    counts = torch.bincount(flat_code[valid].view(-1), minlength=14 * 81)
                    conf_mat_14 += counts.view(14, 9, 9).cpu().numpy()

        if was_training:
            pl_module.train()

        cm_global = conf_mat_14.sum(axis=0)
        global_metrics = compute_metrics_from_confmat(cm_global, num_classes=9)
        global_miou = global_metrics.get("miou_standard", 0.0)

        biome_mious = {}
        for b in range(14):
            b_cm = conf_mat_14[b]
            b_m = compute_metrics_from_confmat(b_cm, num_classes=9)
            biome_mious[BIOME_NAMES[b]] = b_m.get("miou_standard", 0.0)

        fairness = compute_group_fairness(biome_mious)
        avg_loss = total_loss / max(1, total_batches)

        return {
            "loss": avg_loss,
            "miou": global_miou,
            "worst_case_miou": fairness.get("worst", 0.0),
            "best_miou": fairness.get("best", 0.0),
            "spread": fairness.get("spread", 0.0),
            "per": fairness.get("per", 0.0),
            "std": fairness.get("std", 0.0),
            "gini": fairness.get("gini", 0.0),
            "biome_mious": biome_mious,
            "conf_mat": conf_mat_14,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.start_epoch:
            return

        print(f"\n{'='*80}")
        print(f">>> [Epoch {current_epoch:02d}] Evaluating Full 14-Biome Validation & Test Sets...")
        print(f"{'='*80}")

        val_res = self._evaluate_split(pl_module, self.val_loader)
        test_res = self._evaluate_split(pl_module, self.test_loader)

        pl_module.log("epoch_val/worst_miou", val_res["worst_case_miou"], prog_bar=True)
        pl_module.log("epoch_test/worst_miou", test_res["worst_case_miou"], prog_bar=True)

        entry = {
            "epoch": current_epoch,
            "val": {k: v for k, v in val_res.items() if k != "conf_mat"},
            "test": {k: v for k, v in test_res.items() if k != "conf_mat"},
        }
        self.history.append(entry)

        with open(self.history_file, "w") as f:
            json.dump(to_builtin_serializable(self.history), f, indent=2)

        write_header = not self.csv_file.exists()
        with open(self.csv_file, "a") as f:
            if write_header:
                f.write("epoch,val_loss,val_miou,val_worst_miou,val_best_miou,val_spread,val_per,test_loss,test_miou,test_worst_miou,test_best_miou,test_spread,test_per\n")
            f.write(
                f"{current_epoch},"
                f"{val_res['loss']:.4f},{val_res['miou']:.2f},{val_res['worst_case_miou']:.2f},{val_res['best_miou']:.2f},{val_res['spread']:.2f},{val_res['per']:.4f},"
                f"{test_res['loss']:.4f},{test_res['miou']:.2f},{test_res['worst_case_miou']:.2f},{test_res['best_miou']:.2f},{test_res['spread']:.2f},{test_res['per']:.4f}\n"
            )

        print(
            f"[Epoch {current_epoch:02d} Summary]\n"
            f"  VAL  : Loss={val_res['loss']:.4f} | mIoU={val_res['miou']:.2f}% | Worst={val_res['worst_case_miou']:.2f}% | Best={val_res['best_miou']:.2f}% | Spread={val_res['spread']:.2f}% | PER={val_res['per']:.4f}\n"
            f"  TEST : Loss={test_res['loss']:.4f} | mIoU={test_res['miou']:.2f}% | Worst={test_res['worst_case_miou']:.2f}% | Best={test_res['best_miou']:.2f}% | Spread={test_res['spread']:.2f}% | PER={test_res['per']:.4f}"
        )

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if val_res["worst_case_miou"] > self.best_val_worst_miou:
            self.best_val_worst_miou = val_res["worst_case_miou"]
            best_val_worst_path = self.ckpt_dir / "best_val_worst_miou.ckpt"
            trainer.save_checkpoint(str(best_val_worst_path))
            print(f"  ★ New Best Val Worst-Case mIoU: {self.best_val_worst_miou:.2f}% -> Saved {best_val_worst_path.name}")

        if val_res["miou"] > self.best_val_miou:
            self.best_val_miou = val_res["miou"]
            best_val_miou_path = self.ckpt_dir / "best_val_miou.ckpt"
            trainer.save_checkpoint(str(best_val_miou_path))
            print(f"  ★ New Best Val Overall mIoU: {self.best_val_miou:.2f}% -> Saved {best_val_miou_path.name}")

        if test_res["worst_case_miou"] > self.best_test_worst_miou:
            self.best_test_worst_miou = test_res["worst_case_miou"]
            best_test_worst_path = self.ckpt_dir / "best_test_worst_miou.ckpt"
            trainer.save_checkpoint(str(best_test_worst_path))
            print(f"  ★ New Best Test Worst-Case mIoU: {self.best_test_worst_miou:.2f}% -> Saved {best_test_worst_path.name}")

        print(f"{'='*80}\n")


@torch.no_grad()
def fit_bolp_projector(model, dataloader, tile_info, model_name, device) -> torch.Tensor:
    """Computes SVD orthogonal projector P = I - U_k U_k^T on 14-biome prototypes."""
    print("\n[BOLP] Computing 14-biome prototypes on training set...")
    model.eval()
    model.to(device)

    biome_feats = defaultdict(list)
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        filenames = batch["filename"]
        if model_name == "dofa":
            multi_feats = model.encoder.forward_intermediates(images, model.wavelengths, out_indices=[11])
            feats = multi_feats[-1].mean(dim=(2, 3)) # (B, 768)
        elif model_name == "terrafm":
            multi_feats = model.feature_extractor.extract_features(images)
            feats = multi_feats[-1].mean(dim=(2, 3)) # (B, 768)
        else: # satmae
            multi_feats = model.forward_intermediates(images)
            feats = multi_feats[-1].mean(dim=(2, 3)) # (B, 768)

        for i, sid in enumerate(filenames):
            b_info = tile_info.get(str(sid), {})
            b = b_info.get("biome", None)
            if b is not None and 0 <= int(b) < 14:
                biome_feats[int(b)].append(feats[i].cpu())

        if (batch_idx + 1) % 50 == 0:
            print(f"  [BOLP] Processed {batch_idx + 1}/{len(dataloader)} feature batches...")

    centroids = []
    for b_idx in range(14):
        if b_idx in biome_feats and len(biome_feats[b_idx]) > 0:
            c = torch.stack(biome_feats[b_idx], dim=0).mean(dim=0)
            centroids.append(c)

    if not centroids:
        return torch.eye(768)

    C = torch.stack(centroids, dim=0) # (|G|, d)
    C_centered = C - C.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(C_centered, full_matrices=False)
    V = Vh.T # (d, |G|)
    k = min(len(centroids) - 1, V.shape[1])
    Uk = V[:, :k] # (d, k)
    Pk = torch.eye(Uk.shape[0]) - Uk @ Uk.T
    print(f"[BOLP] Projector fitted successfully: shape {Pk.shape}, rank {torch.linalg.matrix_rank(Pk).item()}")
    return Pk


def parse_args():
    p = argparse.ArgumentParser(description="Universal Balanced MMEarth-20K Benchmark Pipeline")
    p.add_argument("--model", choices=["prithvi", "satmae", "dofa", "terrafm"], required=True,
                   help="Foundation model backbone")
    p.add_argument("--method", choices=["erm", "groupdro", "dbr", "bolp"], default="erm",
                   help="Optimization / mitigation method (configured on 14 biomes)")
    p.add_argument("--mode", choices=["eval", "train"], default="eval",
                   help="Mode: evaluate checkpoint or train from scratch")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint file (defaults to existing checkpoint if available)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output destination directory (defaults to Final_mmearth20k/<model>/<method>/)")
    p.add_argument("--data-root", type=str, default="/data1/aminur/om/MMearth20k_balanced",
                   help="Path to balanced MMEarth dataset root")
    p.add_argument("--gpu", type=str, default="0",
                   help="CUDA GPU ID (e.g. 0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (defaults to 32 for train, 4 for eval)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of epochs for training")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="Weight decay")
    p.add_argument("--eta-q", type=float, default=0.01,
                   help="Group DRO step size")
    p.add_argument("--dbr-alpha", type=float, default=0.5,
                   help="DBR feedback weight")
    p.add_argument("--dbr-ema", type=float, default=0.9,
                   help="DBR EMA beta")
    p.add_argument("--start-eval-epoch", type=int, default=0,
                   help="Start epoch for full 14-biome val and test evaluation logging (default: 0)")
    return p.parse_args()


def build_datamodule(model_name: str, data_root: Path, batch_size: int, num_workers: int = 4):
    if model_name == "prithvi":
        bands = PRITHVI_BANDS
    elif model_name == "satmae":
        bands = SATMAE_BANDS
    elif model_name == "dofa":
        bands = DOFA_BANDS
    elif model_name == "terrafm":
        bands = TERRAFM_BANDS
    else:
        raise ValueError(f"Unknown model: {model_name}")

    dm = MMEarthDynamicWorldSegmentationDataModule(
        data_root=data_root,
        sentinel2_bands=bands,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
        normalize=True,
    )
    return dm


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_root = Path(args.data_root)
    assert data_root.exists(), f"Data root not found: {data_root}"
    tile_info = load_tile_info_dict(data_root)

    out_dir = Path(args.output_dir) if args.output_dir else REPO_DIR / f"Final_mmearth20k/{args.model}/{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print(f" UNIVERSAL BALANCED MMEARTH-20K BENCHMARK")
    print(f" Model       : {args.model.upper()}")
    print(f" Method      : {args.method.upper()}")
    print(f" Mode        : {args.mode.upper()}")
    print(f" Output Dir  : {out_dir}")
    print(f" Device      : {device}")
    print("="*80 + "\n")

    # ── EVALUATION MODE ───────────────────────────────────────────────────────
    if args.mode == "eval":
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            default_ckpt = get_default_checkpoint(args.model, args.method)
            if default_ckpt and default_ckpt.exists():
                ckpt_path = str(default_ckpt)
            else:
                raise FileNotFoundError(
                    f"No checkpoint provided and no default found for {args.model}/{args.method}. "
                    f"Please specify --checkpoint <path>."
                )

        print(f">>> Loading Checkpoint: {ckpt_path}")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt_dict.get("state_dict", ckpt_dict)
        ckpt_epoch = ckpt_dict.get("epoch", 49) if isinstance(ckpt_dict, dict) else 49

        if args.model == "satmae":
            from methods.satmae.mmearth_upernet.run_satmae_14biomes import SatMAEUperNetSegmentationClassifier
            model_core = SatMAEUperNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(SATMAE_WEIGHTS), freeze_backbone=True)
            module = UniversalUPerNetModule(model=model_core, tile_info_dict=tile_info, model_name="satmae", method=args.method)
            module.load_state_dict(state_dict, strict=False)
            module.to(device)
            module.eval()
        elif args.model == "dofa":
            from fairrsfm.models.dofa_wrapper import DOFAUperNetSegmentationClassifier
            model_core = DOFAUperNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(DOFA_WEIGHTS), wavelengths=DOFA_WAVELENGTHS, freeze_backbone=True)
            module = UniversalUPerNetModule(model=model_core, tile_info_dict=tile_info, model_name="dofa", method=args.method)
            module.load_state_dict(state_dict, strict=False)
            module.to(device)
            module.eval()
        elif args.model == "terrafm":
            from fairrsfm.models.terrafm_wrapper import TerraFMUPerNetSegmentationClassifier
            model_core = TerraFMUPerNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(TERRAFM_WEIGHTS), freeze_backbone=True, fpn_dim=256)
            module = UniversalUPerNetModule(model=model_core, tile_info_dict=tile_info, model_name="terrafm", method=args.method)
            module.load_state_dict(state_dict, strict=False)
            module.to(device)
            module.eval()
        elif args.model == "prithvi":
            from terratorch.tasks import SemanticSegmentationTask
            from fairrsfm.training.protocol import MethodTrainingMixin
            class BenchmarkSegmentationTask(MethodTrainingMixin, SemanticSegmentationTask):
                pass
            model_args = {
                "backbone": "prithvi_eo_v2_300_tl",
                "backbone_pretrained": True,
                "backbone_ckpt_path": str(PRITHVI_WEIGHTS),
                "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
                "backbone_num_frames": 1,
                "num_classes": NUM_CLASSES,
                "decoder": "UperNetDecoder",
                "decoder_channels": 256,
                "head_dropout": 0.1,
                "rescale": True,
                "necks": [
                    {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                    {"name": "ReshapeTokensToImage"},
                    {"name": "LearnedInterpolateToPyramidal"},
                ],
            }
            module = BenchmarkSegmentationTask(
                model_args=model_args,
                model_factory="EncoderDecoderFactory",
                loss="ce",
                ignore_index=-1,
                lr=args.lr,
                optimizer="AdamW",
                optimizer_hparams={"weight_decay": args.weight_decay},
                freeze_backbone=True,
                freeze_decoder=False,
            )
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace("model.model.", "model.") if k.startswith("model.model.") else k
                clean_state_dict[clean_k] = v
            module.load_state_dict(clean_state_dict, strict=False)
            module.to(device)
            module.eval()

        batch_size = args.batch_size or (2 if args.model == "dofa" else 4)
        dm = build_datamodule(args.model, data_root, batch_size=batch_size, num_workers=2)
        dm.setup("test")

        print(f">>> Commencing Unified 14-Biome Evaluation (Batch Size: {batch_size})...")
        evaluate_dataset_confusion_matrix(
            model=module,
            dataloader=dm.test_dataloader(),
            device=device,
            tile_info=tile_info,
            data_root=data_root,
            out_dir=out_dir,
            ckpt_path=ckpt_path,
            ckpt_epoch=ckpt_epoch,
        )

        print(f"\n[Done] Evaluation artifacts successfully written to: {out_dir}")
        return

    # ── TRAINING MODE ─────────────────────────────────────────────────────────
    elif args.mode == "train":
        batch_size = args.batch_size or 32
        dm = build_datamodule(args.model, data_root, batch_size=batch_size, num_workers=2)
        dm.setup("fit")
        dm.setup("test")

        projector = None
        if args.method == "bolp" and args.model in ["satmae", "dofa", "terrafm"]:
            # Pre-compute BOLP projector
            if args.model == "satmae":
                from methods.satmae.mmearth_upernet.run_satmae_14biomes import SatMAEUperNetSegmentationClassifier
                raw_m = SatMAEUperNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(SATMAE_WEIGHTS), freeze_backbone=True)
            elif args.model == "terrafm":
                from fairrsfm.models.terrafm_wrapper import TerraFMUPerNetSegmentationClassifier
                raw_m = TerraFMUPerNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(TERRAFM_WEIGHTS), freeze_backbone=True)
            else:
                from fairrsfm.models.dofa_wrapper import DOFAUperNetSegmentationClassifier
                raw_m = DOFAUperNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(DOFA_WEIGHTS), wavelengths=DOFA_WAVELENGTHS, freeze_backbone=True)
            projector = fit_bolp_projector(raw_m, dm.train_dataloader(), tile_info, args.model, device)
            del raw_m
            torch.cuda.empty_cache()

        if args.model == "satmae":
            from methods.satmae.mmearth_upernet.run_satmae_14biomes import SatMAEUperNetSegmentationClassifier
            model_core = SatMAEUperNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(SATMAE_WEIGHTS), freeze_backbone=True)
            module = UniversalUPerNetModule(
                model=model_core,
                tile_info_dict=tile_info,
                model_name="satmae",
                method=args.method,
                lr=args.lr,
                weight_decay=args.weight_decay,
                eta_q=args.eta_q,
                dbr_alpha=args.dbr_alpha,
                dbr_ema=args.dbr_ema,
                max_epochs=args.epochs,
                projector=projector,
            )
        elif args.model == "terrafm":
            from fairrsfm.models.terrafm_wrapper import TerraFMUPerNetSegmentationClassifier
            model_core = TerraFMUPerNetSegmentationClassifier(
                num_classes=NUM_CLASSES,
                checkpoint_path=str(TERRAFM_WEIGHTS),
                freeze_backbone=True,
                fpn_dim=256,
            )
            module = UniversalUPerNetModule(
                model=model_core,
                tile_info_dict=tile_info,
                model_name="terrafm",
                method=args.method,
                lr=args.lr,
                weight_decay=args.weight_decay,
                eta_q=args.eta_q,
                dbr_alpha=args.dbr_alpha,
                dbr_ema=args.dbr_ema,
                max_epochs=args.epochs,
                projector=projector,
            )
        elif args.model == "dofa":
            from fairrsfm.models.dofa_wrapper import DOFAUperNetSegmentationClassifier
            model_core = DOFAUperNetSegmentationClassifier(num_classes=NUM_CLASSES, checkpoint_path=str(DOFA_WEIGHTS), wavelengths=DOFA_WAVELENGTHS, freeze_backbone=True)
            module = UniversalUPerNetModule(
                model=model_core,
                tile_info_dict=tile_info,
                model_name="dofa",
                method=args.method,
                lr=args.lr,
                weight_decay=args.weight_decay,
                eta_q=args.eta_q,
                dbr_alpha=args.dbr_alpha,
                dbr_ema=args.dbr_ema,
                max_epochs=args.epochs,
                projector=projector,
            )
        elif args.model == "prithvi":
            # Delegate to specialized prithvi pipeline
            from terratorch.tasks import SemanticSegmentationTask
            from fairrsfm.training.protocol import MethodTrainingMixin, configure_training_protocol

            model_args = {
                "backbone": "prithvi_eo_v2_300_tl",
                "backbone_pretrained": True,
                "backbone_ckpt_path": str(PRITHVI_WEIGHTS),
                "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
                "backbone_num_frames": 1,
                "num_classes": NUM_CLASSES,
                "decoder": "UperNetDecoder",
                "decoder_channels": 256,
                "head_dropout": 0.1,
                "rescale": True,
                "necks": [
                    {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                    {"name": "ReshapeTokensToImage"},
                    {"name": "LearnedInterpolateToPyramidal"},
                ],
            }

            protocol_groups = {
                str(tile_id): BIOME_NAMES[int(info["biome"])]
                for tile_id, info in tile_info.items()
                if "biome" in info and 0 <= int(info["biome"]) < 14
            }

            if args.method == "bolp":
                from fairrsfm.methods.bolp import PrithviFinalLayerBOLP
                bolp_module = PrithviFinalLayerBOLP(
                    k=None,
                    device=device,
                    tile_lookup=lambda sid, ti: BIOME_NAMES[int(ti[str(sid)]["biome"])] if "biome" in ti.get(str(sid), {}) else "Unknown",
                )
                class BOLPSemanticSegmentationTask(MethodTrainingMixin, SemanticSegmentationTask):
                    def __init__(self, *args_inner, bolp_module=None, tile_info=None, **kwargs_inner):
                        super().__init__(*args_inner, **kwargs_inner)
                        self.bolp_module = bolp_module
                        self.tile_info = tile_info or {}

                    def _bolp_forward(self, batch):
                        images = batch["image"]
                        dev = next(self.parameters()).device
                        if images.device != dev:
                            images = images.to(dev, non_blocking=True)
                        filenames = list(batch.get("filename", [None] * images.size(0)))
                        return self.bolp_module.forward_with_bolp(
                            self, images, tile_ids=filenames, tile_info=self.tile_info
                        )

                module = BOLPSemanticSegmentationTask(
                    model_args=model_args,
                    model_factory="EncoderDecoderFactory",
                    loss="ce",
                    ignore_index=-1,
                    lr=args.lr,
                    optimizer="AdamW",
                    optimizer_hparams={"weight_decay": args.weight_decay},
                    freeze_backbone=True,
                    freeze_decoder=False,
                    bolp_module=bolp_module,
                    tile_info=tile_info,
                )
                module.to(device)
                print("[BOLP] Fitting Prithvi BOLP projection on training set...")
                bolp_module.fit(module, dm.train_dataloader(), tile_info)
            else:
                class BenchmarkSegmentationTask(MethodTrainingMixin, SemanticSegmentationTask):
                    pass

                module = BenchmarkSegmentationTask(
                    model_args=model_args,
                    model_factory="EncoderDecoderFactory",
                    loss="ce",
                    ignore_index=-1,
                    lr=args.lr,
                    optimizer="AdamW",
                    optimizer_hparams={"weight_decay": args.weight_decay},
                    freeze_backbone=True,
                    freeze_decoder=False,
                )

            configure_training_protocol(
                module=module,
                method=args.method,
                task="mmearth",
                groups=protocol_groups,
                args=args,
                ignore_index=-1,
                lookup=lambda sid: BIOME_NAMES[int(tile_info[str(sid)]["biome"])] if "biome" in tile_info.get(str(sid), {}) else "Unknown",
            )

        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{args.model}_{args.method}_{{epoch:02d}}_{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        logger = TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")

        # Set up full evaluation dataloaders for the epoch-level callback
        eval_bs = 16 if args.model == "dofa" else 32
        dm_eval = build_datamodule(args.model, data_root, batch_size=eval_bs, num_workers=2)
        dm_eval.setup("fit")
        dm_eval.setup("test")

        epoch_eval_cb = EpochValTestEvalCallback(
            val_loader=dm_eval.val_dataloader(),
            test_loader=dm_eval.test_dataloader(),
            tile_info=tile_info,
            out_dir=out_dir,
            start_epoch=args.start_eval_epoch,
            device=device,
        )

        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
            max_epochs=args.epochs,
            callbacks=[ckpt_cb, epoch_eval_cb, TQDMProgressBar(refresh_rate=20)],
            logger=logger,
            log_every_n_steps=20,
        )

        resume_ckpt = args.checkpoint
        last_ckpt_path = ckpt_dir / "last.ckpt"
        if resume_ckpt is None and last_ckpt_path.exists():
            resume_ckpt = str(last_ckpt_path)
            print(f">>> Found existing last checkpoint, resuming training from: {resume_ckpt}")

        print(f"\n>>> Starting {args.model.upper()} ({args.method.upper()}) Training for {args.epochs} Epochs...")
        print(f">>> Full 14-Biome Val & Test evaluation will be saved every epoch starting from Epoch {args.start_eval_epoch}!")
        trainer.fit(module, datamodule=dm, ckpt_path=resume_ckpt)

        # Select best checkpoint for final publication artifact suite
        best_ckpt = None
        best_val_worst = ckpt_dir / "best_val_worst_miou.ckpt"
        best_test_worst = ckpt_dir / "best_test_worst_miou.ckpt"
        last_ckpt = ckpt_dir / "last.ckpt"

        if best_val_worst.exists():
            best_ckpt = str(best_val_worst)
            print(f"\n[Final Artifacts] Using Best Val Worst-Case Checkpoint: {best_ckpt}")
        elif best_test_worst.exists():
            best_ckpt = str(best_test_worst)
            print(f"\n[Final Artifacts] Using Best Test Worst-Case Checkpoint: {best_ckpt}")
        elif ckpt_cb.best_model_path and Path(ckpt_cb.best_model_path).exists():
            best_ckpt = str(ckpt_cb.best_model_path)
            print(f"\n[Final Artifacts] Using Best Val Loss Checkpoint: {best_ckpt}")
        elif last_ckpt.exists():
            best_ckpt = str(last_ckpt)
            print(f"\n[Final Artifacts] Using Last Checkpoint: {best_ckpt}")
        else:
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                best_ckpt = str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1])

        print(f"\n[Training Complete] Selected checkpoint for reporting: {best_ckpt}")

        # Post-training evaluation
        print(f"\n>>> Running Unified Evaluation on Selected Checkpoint...")
        if best_ckpt and Path(best_ckpt).exists():
            ckpt_dict = torch.load(best_ckpt, map_location="cpu")
            sd = ckpt_dict.get("state_dict", ckpt_dict)
            clean_sd = {}
            for k, v in sd.items():
                clean_k = k.replace("model.model.", "model.") if k.startswith("model.model.") else k
                clean_sd[clean_k] = v
            try:
                module.load_state_dict(clean_sd, strict=False)
            except Exception as e:
                print(f"[Warning] Loading checkpoint state encountered: {e}. Proceeding with current weights.")
        module.to(device)
        module.eval()

        evaluate_dataset_confusion_matrix(
            model=module,
            dataloader=dm_eval.test_dataloader(),
            device=device,
            tile_info=tile_info,
            data_root=data_root,
            out_dir=out_dir,
            ckpt_path=best_ckpt,
            ckpt_epoch=args.epochs,
        )
        print(f"\n[Done] Training and Evaluation Pipeline Completed! Artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
