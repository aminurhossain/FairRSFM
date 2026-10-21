#!/usr/bin/env python3
"""
DOFA ViT-Base 16 — MMEarth-20k Dynamic World Segmentation (Official UPerNet Decoder - DBR)
========================================================================================
Key Features:
  • Task: Semantic Segmentation (MMEarth Dynamic World, 9 classes, CrossEntropyLoss)
  • Architecture: DOFA ViT-Base 16 + Multi-Level UPerNet Decoder (layers [2, 5, 8, 11] + PPM + FPN)
  • Method: Distribution-Balanced Reweighting (DBR) with Dynamic Group Loss Balancing
  • Wavelengths (µm): [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 0.842, 0.945, 1.610, 2.190]
  • Bands: Sentinel-2 12 Bands (B1, B2, B3, B4, B5, B6, B7, B8A, B8, B9, B11, B12)
  • Optimizer: AdamW (LR=1e-3, Weight Decay=0.01) + CosineAnnealingLR
  • Grouping: Macro Biomes (5 macro groups from 14 RESOLVE biomes)
  • Subgroup Unique-Class Evaluation: mIoU, Macro F1, Pixel Accuracy, and ECE per biome
  • Metrics Suite: Global & Per-biome mIoU, Worst/Best mIoU, PER, Gini, NFR, EqOdds, DPM, Per-Class IoU Table.
  • Slim Checkpoints: Strips frozen encoder weights (~1.5 MB per .ckpt file).
  • Consolidated JSONs: Saves val_results.json and test_results.json at every epoch.
"""

import sys
import os
import json
import logging
import warnings
import argparse
from pathlib import Path
from collections import defaultdict

# Project root and module resolution
ROOT_DIR = Path(__file__).resolve()
while ROOT_DIR.parent != ROOT_DIR:
    if (ROOT_DIR / "pyproject.toml").is_file() or (ROOT_DIR / "src/fairrsfm").is_dir():
        break
    ROOT_DIR = ROOT_DIR.parent

for p in [ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "src/fairrsfm", ROOT_DIR / "src/fairrsfm/models/upstream/SatMAE"]:
    if str(p) not in sys.path and p.is_dir():
        sys.path.insert(0, str(p))

MMEARTH_DIR = ROOT_DIR / "segmentation" / "mmearth_20k"
if str(MMEARTH_DIR) not in sys.path:
    sys.path.insert(0, str(MMEARTH_DIR))

import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from models.dofa_wrapper import DOFAUperNetSegmentationClassifier
from mmearth_dynamic_world_datamodule import (
    DYNAMIC_WORLD_CLASSES,
    MMEarthDynamicWorldSegmentationDataModule,
    find_one
)

from sklearn.metrics import f1_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEE LOGGER & CLEAN PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────

class TeeLogger:
    def __init__(self, log_file_path: Path):
        self.terminal = sys.stdout
        self.log_file_path = log_file_path
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if "\r" in message or "%|" in message or "it/s]" in message:
            return
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


class CleanProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 20):
        super().__init__(refresh_rate=refresh_rate)

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        for k in list(items.keys()):
            if k.startswith("val/"):
                items.pop(k, None)
        return items


# ─────────────────────────────────────────────────────────────────────────────
# 2. PATHS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = ROOT_DIR
DATA_ROOT = MMEARTH_DIR / "hardcoded"
WEIGHTS_PT = BASE_DIR / "checkpoints/dofa/dofa_base_patch16_224.pth"

NUM_CLASSES  = len(DYNAMIC_WORLD_CLASSES)  # 9
IGNORE_INDEX = -1
CLASS_NAMES  = DYNAMIC_WORLD_CLASSES

DOFA_S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B11", "B12"]
DOFA_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 0.842, 0.945, 1.610, 2.190]

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 0.01
SEED         = 42
NUM_WORKERS  = 8
IMAGE_SIZE   = 224
FREEZE_BACKBONE = True

EARLY_STOP_PATIENCE  = 1000
EARLY_STOP_MIN_DELTA = 1e-4

# 14 RESOLVE biomes -> 5 Macro groups
BIOME_MACRO_GROUPS = {
    1:  "high biomass",
    3:  "high biomass",
    5:  "high biomass",
    2:  "High amplitude phenological",
    6:  "High amplitude phenological",
    4:  "High amplitude phenological",
    7:  "transitional herbaceous & scrub",
    8:  "transitional herbaceous & scrub",
    12: "transitional herbaceous & scrub",
    10: "cryospheric & short-cycle",
    11: "cryospheric & short-cycle",
    13: "xeric & mineralogical",
    9:  "hydrologically modulated",
    14: "hydrologically modulated",
}


def load_tile_info_dict(data_root: Path) -> dict:
    tile_info_path = find_one(data_root, "*tile_info.json")
    with open(tile_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_macro_biome(tile_info_dict: dict, tile_id: str) -> str:
    if tile_id not in tile_info_dict:
        return "Unknown"
    info = tile_info_dict[tile_id]
    b_idx = info.get("biome", None)
    if b_idx is None:
        return "Unknown"
    key = int(b_idx) + 1
    return BIOME_MACRO_GROUPS.get(key, "Unknown")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXACT METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, num_classes: int, ignore_index: int = -1) -> float:
    mask    = y_true_flat != ignore_index
    y_true  = y_true_flat[mask]
    y_pred  = y_pred_flat[mask]
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    if not present:
        return float("nan")
    ious = []
    for c in present:
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_per_class_iou(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, num_classes: int, ignore_index: int = -1) -> np.ndarray:
    mask    = y_true_flat != ignore_index
    y_true  = y_true_flat[mask]
    y_pred  = y_pred_flat[mask]
    ious = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        if (y_true == c).sum() == 0:
            continue
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else float("nan")
    return ious


def compute_macro_f1_seg(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, num_classes: int, ignore_index: int = -1) -> float:
    mask   = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    if not present:
        return float("nan")
    f1s = []
    for c in present:
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in f1s if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_ece_seg(y_true_flat: np.ndarray, probs_max: np.ndarray, preds_flat: np.ndarray, ignore_index: int = -1, n_bins: int = 10) -> float:
    mask       = y_true_flat != ignore_index
    y_true     = y_true_flat[mask]
    probs_max  = probs_max[mask]
    preds_flat = preds_flat[mask]
    if len(y_true) == 0:
        return float("nan")
    correct = (preds_flat == y_true).astype(float)
    ece, n  = 0.0, len(y_true)
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (probs_max >= lo) & (probs_max < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - probs_max[m].mean())
    return float(ece)


def compute_gini(values: list) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    arr = np.sort(np.array(vals))
    n = len(arr)
    sum_val = np.sum(arr)
    if sum_val == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * arr) / (n * sum_val))


def compute_eqodds_seg(biome_data: dict, num_classes: int, ignore_index: int = -1) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.concatenate(biome_data[b]["true"])
        yp = np.concatenate(biome_data[b]["pred"])
        mask = yt != ignore_index
        yt_v = yt[mask]
        yp_v = yp[mask]
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            yt_c = (yt_v == c).astype(int)
            yp_c = (yp_v == c).astype(int)
            tp = ((yp_c == 1) & (yt_c == 1)).sum()
            fn = ((yp_c == 0) & (yt_c == 1)).sum()
            fp = ((yp_c == 1) & (yt_c == 0)).sum()
            tn = ((yp_c == 0) & (yt_c == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            if not (np.isnan(tpr) or np.isnan(fpr)):
                j_b[c] = tpr - fpr
        J[b] = j_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i+1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes) if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
    return worst


def compute_demographic_parity_seg(biome_data: dict, num_classes: int, ignore_index: int = -1) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.concatenate(biome_data[b]["pred"])
        yt = np.concatenate(biome_data[b]["true"])
        mask = yt != ignore_index
        yp_v = yp[mask]
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = (yp_v == c).mean() if len(yp_v) > 0 else 0.0
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i+1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIGHTNING MODULE (DOFA UPERNET DBR)
# ─────────────────────────────────────────────────────────────────────────────

class DOFAUperNetMMEarthDBR(L.LightningModule):
    def __init__(self, model: DOFAUperNetSegmentationClassifier, tile_info_dict: dict, lr: float,
                 alpha: float = 0.5, ema_decay: float = 0.9, weight_decay: float = WEIGHT_DECAY,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tile_info_dict"])
        self.model          = model
        self.tile_info_dict = tile_info_dict
        self.num_classes    = num_classes
        self.alpha          = alpha
        self.ema_decay      = ema_decay

        known_biomes = sorted(set(BIOME_MACRO_GROUPS.values()))
        self.biome_to_idx = {b: i for i, b in enumerate(known_biomes)}
        self.num_biomes = len(known_biomes)

        self.register_buffer("running_group_loss", torch.ones(self.num_biomes, dtype=torch.float32))

        if FREEZE_BACKBONE:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()
            frozen    = sum(p.numel() for p in self.model.encoder.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n[Model] Frozen    : {frozen/1e6:.1f}M  (encoder)")
            print(f"[Model] Trainable : {trainable/1e6:.1f}M  (UPerNet head)")

    def on_save_checkpoint(self, checkpoint):
        if FREEZE_BACKBONE:
            state_dict = checkpoint.get("state_dict", {})
            keys_to_remove = [k for k in list(state_dict.keys()) if k.startswith("model.encoder.")]
            for k in keys_to_remove:
                del state_dict[k]

    def on_load_checkpoint(self, checkpoint):
        if FREEZE_BACKBONE:
            state_dict = checkpoint.get("state_dict", {})
            for k, v in self.model.encoder.state_dict().items():
                state_dict[f"model.encoder.{k}"] = v

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images    = batch["image"]
        masks     = batch["mask"].squeeze(1)
        filenames = batch.get("filename", [None] * images.size(0))

        if FREEZE_BACKBONE:
            self.model.encoder.eval()

        logits = self(images)
        raw_losses = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX, reduction="none").mean(dim=(1, 2))

        # Compute dynamic group weights
        w = torch.pow(self.running_group_loss + 1e-6, self.alpha)
        w = w / w.mean()

        sample_weights = torch.ones_like(raw_losses)
        batch_group_losses = torch.zeros(self.num_biomes, device=self.device)
        batch_group_counts = torch.zeros(self.num_biomes, device=self.device)

        for i, sid in enumerate(filenames):
            biome = get_macro_biome(self.tile_info_dict, str(sid))
            b_idx = self.biome_to_idx.get(biome, None)
            if b_idx is not None:
                sample_weights[i] = w[b_idx]
                batch_group_losses[b_idx] += raw_losses[i].detach()
                batch_group_counts[b_idx] += 1

        loss = torch.mean(sample_weights * raw_losses)

        # Update EMA running group losses
        with torch.no_grad():
            active_mask = batch_group_counts > 0
            if active_mask.any():
                curr_group_loss = batch_group_losses[active_mask] / batch_group_counts[active_mask]
                self.running_group_loss[active_mask] = (
                    self.ema_decay * self.running_group_loss[active_mask]
                    + (1.0 - self.ema_decay) * curr_group_loss
                )

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images    = batch["image"]
        masks     = batch["mask"].squeeze(1)
        filenames = batch.get("filename", [None] * images.size(0))
        logits = self(images)
        loss   = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        self.log("val/loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=images.size(0))
        return {"logits": logits, "y": masks, "sample_ids": filenames}

    def test_step(self, batch, batch_idx):
        images    = batch["image"]
        masks     = batch["mask"].squeeze(1)
        filenames = batch.get("filename", [None] * images.size(0))
        logits = self(images)
        loss   = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        self.log("test/loss", loss, add_dataloader_idx=False, batch_size=images.size(0))
        return {"logits": logits, "y": masks, "sample_ids": filenames}

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/miou"}}


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOGGERS & METRIC CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class GlobalSegmentationLogger(Callback):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._g_labels = []
        pl_module._g_preds  = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        preds  = logits.argmax(dim=1).cpu().numpy().flatten()
        labels = y.cpu().numpy().flatten()
        pl_module._g_labels.append(labels)
        pl_module._g_preds.append(preds)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._g_labels:
            return
        y_true = np.concatenate(pl_module._g_labels)
        y_pred = np.concatenate(pl_module._g_preds)

        global_miou = compute_miou(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        global_f1   = compute_macro_f1_seg(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        mask = y_true != IGNORE_INDEX
        global_acc  = float((y_pred[mask] == y_true[mask]).mean()) if mask.sum() > 0 else float("nan")

        pl_module.log("val/miou", global_miou, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/macro_f1", global_f1, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/pixel_acc", global_acc, prog_bar=False, add_dataloader_idx=False)
        print(f"\n  [GlobalMetrics] epoch {trainer.current_epoch} → val/miou = {global_miou:.4f} | val/macro_f1 = {global_f1:.4f} | val/pixel_acc = {global_acc*100:.2f}%")
        pl_module._g_labels = None
        pl_module._g_preds  = None


class BiomeSegmentationLogger(Callback):
    def __init__(self, tile_info_dict: dict, num_classes: int, class_names: list, output_dir: Path, test_loader=None):
        super().__init__()
        self.tile_info_dict = tile_info_dict
        self.num_classes    = num_classes
        self.class_names    = class_names
        self.output_dir     = Path(output_dir)
        self.test_loader    = test_loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
        val_json = self.output_dir / "val_results.json"
        if val_json.exists():
            try:
                with open(val_json, "r") as fp_j:
                    self.val_history = json.load(fp_j)
            except Exception:
                self.val_history = []
        else:
            self.val_history = []

        test_json = self.output_dir / "test_results.json"
        if test_json.exists():
            try:
                with open(test_json, "r") as fp_j:
                    self.test_history = json.load(fp_j)
            except Exception:
                self.test_history = []
        else:
            self.test_history = []

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label, pl_module=None):
        all_yt = np.concatenate([np.concatenate(d["true"]) for d in biome_data.values()])
        all_yp = np.concatenate([np.concatenate(d["pred"]) for d in biome_data.values()])
        all_pr = np.concatenate([np.concatenate(d["probs_max"]) for d in biome_data.values()])

        overall_miou = compute_miou(all_yt, all_yp, self.num_classes, ignore_index=IGNORE_INDEX)
        overall_f1   = compute_macro_f1_seg(all_yt, all_yp, self.num_classes, ignore_index=IGNORE_INDEX)
        mask = all_yt != IGNORE_INDEX
        overall_acc  = float((all_yp[mask] == all_yt[mask]).mean()) if mask.sum() > 0 else float("nan")
        overall_ece  = compute_ece_seg(all_yt, all_pr, all_yp, ignore_index=IGNORE_INDEX)

        known_biome_data = {b: d for b, d in biome_data.items() if b != "Unknown"}

        print(f"\n{'═'*82}")
        print(f"  Segmentation Group Metrics [{label}] — epoch {epoch}")
        print(f"  Overall Pixel Acc: {overall_acc*100:.2f}% | Overall mIoU: {overall_miou*100:.2f}% | Overall Macro F1: {overall_f1*100:.2f}% | ECE: {overall_ece:.4f}")
        print(f"{'═'*82}")
        print(f"\n  ── Per-group Segmentation Metrics ─────────────────────────────────────────")
        print(f"  {'Biome':<44}  {'mIoU':>8}  {'Macro F1':>8}  {'Pixel Acc':>9}  {'ECE':>7}")
        print(f"  {'─'*44}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*7}")

        group_mious  = {}
        group_f1s    = {}
        group_accs   = {}
        group_eces   = {}
        group_counts = {}

        biomes_sorted = sorted(known_biome_data.keys())
        n_biome = len(biomes_sorted)
        iou_matrix = np.full((self.num_classes, n_biome), float("nan"))

        for j, group in enumerate(biomes_sorted):
            d      = known_biome_data[group]
            n      = len(d["sample_ids"])
            y_true = np.concatenate(d["true"])
            y_pred = np.concatenate(d["pred"])
            p_max  = np.concatenate(d["probs_max"])

            miou = compute_miou(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)
            f1   = compute_macro_f1_seg(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)
            m = y_true != IGNORE_INDEX
            acc  = float((y_pred[m] == y_true[m]).mean()) if m.sum() > 0 else float("nan")
            ece  = compute_ece_seg(y_true, p_max, y_pred, ignore_index=IGNORE_INDEX)

            group_mious[group]  = miou
            group_f1s[group]    = f1
            group_accs[group]   = acc
            group_eces[group]   = ece
            group_counts[group] = n

            iou_matrix[:, j] = compute_per_class_iou(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)

            print(f"  {group:<44}  {miou:>8.4f}  {f1:>8.4f}  {acc:>9.4f}  {ece:>7.4f}")

        class_disparity = {}
        per_class_iou_table = {}
        for c in range(self.num_classes):
            cname = self.class_names[c] if c < len(self.class_names) else f"cls_{c}"
            valid = iou_matrix[c, ~np.isnan(iou_matrix[c])]
            disp = float(valid.max() - valid.min()) if len(valid) >= 2 else float("nan")
            class_disparity[cname] = disp
            per_class_iou_table[cname] = {biomes_sorted[j]: float(iou_matrix[c, j]) for j in range(n_biome) if not np.isnan(iou_matrix[c, j])}

        valid_disp = [v for v in class_disparity.values() if not np.isnan(v)]
        mean_class_disp = float(np.mean(valid_disp)) if valid_disp else float("nan")
        max_class_disp  = float(np.max(valid_disp))  if valid_disp else float("nan")

        valid_mious = [v for v in group_mious.values() if v is not None and not np.isnan(v)]
        valid_f1s   = [v for v in group_f1s.values() if v is not None and not np.isnan(v)]
        valid_accs  = [v for v in group_accs.values() if v is not None and not np.isnan(v)]
        valid_eces  = [v for v in group_eces.values() if v is not None and not np.isnan(v)]

        if len(valid_mious) >= 2:
            worst_miou = float(np.min(valid_mious))
            best_miou  = float(np.max(valid_mious))
            mean_miou  = float(np.mean(valid_mious))
            std_miou   = float(np.std(valid_mious))
            per_ratio  = float(worst_miou / best_miou) if best_miou > 0 else float("nan")
            gini_coef  = compute_gini(valid_mious)
            nfr        = float((best_miou - worst_miou) / mean_miou) if mean_miou > 0 else float("nan")
        else:
            worst_miou = best_miou = mean_miou = std_miou = per_ratio = gini_coef = nfr = float("nan")

        worst_f1  = float(np.min(valid_f1s))  if len(valid_f1s) >= 2 else float("nan")
        worst_acc = float(np.min(valid_accs)) if len(valid_accs) >= 2 else float("nan")
        ece_disparity = float(np.max(valid_eces) - np.min(valid_eces)) if len(valid_eces) >= 2 else float("nan")

        eqodds_disp = compute_eqodds_seg(known_biome_data, self.num_classes, ignore_index=IGNORE_INDEX)
        dpm_disp    = compute_demographic_parity_seg(known_biome_data, self.num_classes, ignore_index=IGNORE_INDEX)

        print(f"\n  ── Fairness & Disparity Summary ───────────────────────────────────────────")
        print(f"    mIoU (Macro-avg over groups)     : {mean_miou:.4f}" if not np.isnan(mean_miou) else "    mIoU     : N/A")
        print(f"    Worst-Case Group mIoU            : {worst_miou:.4f}" if not np.isnan(worst_miou) else "    Worst mIoU : N/A")
        print(f"    Best-Case Group mIoU             : {best_miou:.4f}" if not np.isnan(best_miou) else "    Best mIoU  : N/A")
        print(f"    Performance Equity Ratio (PER)   : {per_ratio:.4f}" if not np.isnan(per_ratio) else "    PER        : N/A")
        print(f"    Group mIoU Std Dev               : {std_miou:.4f}" if not np.isnan(std_miou) else "    Std Dev    : N/A")
        print(f"    Gini Coefficient                 : {gini_coef:.4f}" if not np.isnan(gini_coef) else "    Gini       : N/A")
        print(f"    ECE Disparity (Max - Min)        : {ece_disparity:.4f}" if not np.isnan(ece_disparity) else "    ECE Disp   : N/A")
        print(f"    Equalized Odds Disparity         : {eqodds_disp:.4f}" if not np.isnan(eqodds_disp) else "    EqOdds     : N/A")
        print(f"    Demographic Parity (DPM)         : {dpm_disp:.4f}" if not np.isnan(dpm_disp) else "    DPM        : N/A")
        print(f"{'═'*82}\n")

        if pl_module is not None and label == "val":
            if not np.isnan(worst_miou):
                pl_module.log("val/worst_case_miou", worst_miou, prog_bar=False, add_dataloader_idx=False)
            if not np.isnan(mean_miou):
                pl_module.log("val/group_mean_miou", mean_miou, prog_bar=False, add_dataloader_idx=False)

        if tb:
            p = tb_prefix
            tb.add_scalar(f"{p}/overall_mIoU", overall_miou, epoch)
            tb.add_scalar(f"{p}/overall_macro_f1", overall_f1, epoch)
            tb.add_scalar(f"{p}/overall_pixel_acc", overall_acc, epoch)
            tb.add_scalar(f"{p}/overall_ECE", overall_ece, epoch)
            if not np.isnan(worst_miou):
                tb.add_scalar(f"{p}/worst_case_mIoU", worst_miou, epoch)
                tb.add_scalar(f"{p}/best_case_mIoU", best_miou, epoch)
                tb.add_scalar(f"{p}/group_mIoU_std", std_miou, epoch)
                tb.add_scalar(f"{p}/per_ratio", per_ratio, epoch)
                tb.add_scalar(f"{p}/gini_coefficient", gini_coef, epoch)
            if not np.isnan(ece_disparity):
                tb.add_scalar(f"{p}/ece_disparity", ece_disparity, epoch)
            if not np.isnan(eqodds_disp):
                tb.add_scalar(f"{p}/equalized_odds_disparity", eqodds_disp, epoch)
            if not np.isnan(dpm_disp):
                tb.add_scalar(f"{p}/demographic_parity_difference", dpm_disp, epoch)

        record = {
            "epoch": epoch,
            "overall_mIoU": overall_miou,
            "overall_macro_f1": overall_f1,
            "overall_pixel_acc": overall_acc,
            "overall_ECE": overall_ece,
            "worst_case_mIoU": worst_miou if not np.isnan(worst_miou) else None,
            "best_case_mIoU": best_miou if not np.isnan(best_miou) else None,
            "group_mean_mIoU": mean_miou if not np.isnan(mean_miou) else None,
            "group_std_mIoU": std_miou if not np.isnan(std_miou) else None,
            "per_ratio": per_ratio if not np.isnan(per_ratio) else None,
            "normalized_fairness_range": nfr if not np.isnan(nfr) else None,
            "gini_coefficient": gini_coef if not np.isnan(gini_coef) else None,
            "worst_case_macro_f1": worst_f1 if not np.isnan(worst_f1) else None,
            "worst_case_pixel_acc": worst_acc if not np.isnan(worst_acc) else None,
            "ece_disparity": ece_disparity if not np.isnan(ece_disparity) else None,
            "equalized_odds_disparity": eqodds_disp if not np.isnan(eqodds_disp) else None,
            "demographic_parity_difference": dpm_disp if not np.isnan(dpm_disp) else None,
            "mean_class_disparity": mean_class_disp if not np.isnan(mean_class_disp) else None,
            "max_class_disparity": max_class_disp if not np.isnan(max_class_disp) else None,
            "per_group_metrics": {
                g: {
                    "mIoU": group_mious[g],
                    "macro_f1": group_f1s[g],
                    "pixel_acc": group_accs[g],
                    "ece": group_eces[g],
                    "sample_count": group_counts[g],
                }
                for g in group_mious
            },
            "class_disparities": class_disparity,
            "per_class_iou_table": per_class_iou_table
        }

        if label == "val":
            self.val_history = [r for r in self.val_history if r.get("epoch") != epoch]
            self.val_history.append(record)
            with open(self.output_dir / "val_results.json", "w") as f:
                json.dump(self.val_history, f, indent=2)
        elif label == "test":
            self.test_history = [r for r in self.test_history if r.get("epoch") != epoch]
            self.test_history.append(record)
            with open(self.output_dir / "test_results.json", "w") as f:
                json.dump(self.test_history, f, indent=2)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_data = defaultdict(lambda: {"true": [], "pred": [], "probs_max": [], "sample_ids": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        sids   = outputs["sample_ids"]

        probs = F.softmax(logits.float(), dim=1)
        p_max, preds = probs.max(dim=1)

        preds_np  = preds.cpu().numpy()
        labels_np = y.cpu().numpy()
        p_max_np  = p_max.cpu().numpy()

        for sid, label_arr, pred_arr, p_max_arr in zip(sids, labels_np, preds_np, p_max_np):
            biome = get_macro_biome(self.tile_info_dict, str(sid))
            self._val_data[biome]["true"].append(label_arr.flatten())
            self._val_data[biome]["pred"].append(pred_arr.flatten())
            self._val_data[biome]["probs_max"].append(p_max_arr.flatten())
            self._val_data[biome]["sample_ids"].append(sid)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._val_data:
            return
        epoch = trainer.current_epoch
        tb = pl_module.logger.experiment if pl_module.logger else None

        self._compute_and_log(self._val_data, epoch, tb, "val", "val", pl_module=pl_module)

        if self.test_loader is not None:
            self._run_test_evaluation(trainer, pl_module, epoch, tb)

        self._val_data = None

    def _run_test_evaluation(self, trainer, pl_module, epoch, tb):
        pl_module.eval()
        test_data = defaultdict(lambda: {"true": [], "pred": [], "probs_max": [], "sample_ids": []})
        device = pl_module.device

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(device)
                masks  = batch["mask"].squeeze(1)
                sids   = batch.get("filename", [None] * images.size(0))

                logits = pl_module(images)
                probs = F.softmax(logits.float(), dim=1)
                p_max, preds = probs.max(dim=1)

                preds_np  = preds.cpu().numpy()
                labels_np = masks.numpy()
                p_max_np  = p_max.cpu().numpy()

                for sid, label_arr, pred_arr, p_max_arr in zip(sids, labels_np, preds_np, p_max_np):
                    biome = get_macro_biome(self.tile_info_dict, str(sid))
                    test_data[biome]["true"].append(label_arr.flatten())
                    test_data[biome]["pred"].append(pred_arr.flatten())
                    test_data[biome]["probs_max"].append(p_max_arr.flatten())
                    test_data[biome]["sample_ids"].append(sid)

        self._compute_and_log(test_data, epoch, tb, "test", "test", pl_module=None)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DOFA MMEarth-20k UPerNet DBR Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-3)")
    parser.add_argument("--alpha", type=float, default=0.5, help="DBR power exponent alpha (default: 0.5)")
    parser.add_argument("--ema-decay", type=float, default=0.9, help="DBR EMA smoothing factor (default: 0.9)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    mode_str = "DOFA_mmearth_upernet_dbr"
    OUT_DIR  = BASE_DIR / f"DOFA/DOFA_runs/{mode_str}/seed_{seed}"
    CKPT_DIR = OUT_DIR  / "checkpoints"

    for p in [DATA_ROOT, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset      : MMEarth-20k Dynamic World ({NUM_CLASSES} classes)")
    print(f"[Config] Architecture : DOFA ViT-Base 16 (Official UPerNet Decoder)")
    print(f"[Config] Mode         : Distribution-Balanced Reweighting (DBR)")
    print(f"[Config] LR           : {args.lr}")
    print(f"[Config] Alpha        : {args.alpha}")
    print(f"[Config] EMA Decay    : {args.ema_decay}")
    print(f"[Config] Seed         : {seed}")
    print(f"[Config] Grouping     : 5 Macro Biome Groups (tile_info.json)")
    print(f"[Logging] Console log : {OUT_DIR / 'train.log'}")
    print(f"[Logging] JSON dir    : {OUT_DIR}")

    tile_info_dict = load_tile_info_dict(DATA_ROOT)

    datamodule = MMEarthDynamicWorldSegmentationDataModule(
        data_root              = DATA_ROOT,
        sentinel2_bands        = DOFA_S2_BANDS,
        batch_size             = BATCH_SIZE,
        num_workers            = NUM_WORKERS,
        image_size             = IMAGE_SIZE,
        normalize              = True,
        fallback_val_fraction  = 0.1,
        fallback_test_fraction = 0.1,
        seed                   = seed,
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()
    test_loader  = datamodule.test_dataloader()

    print(f"[Data] Train: {len(datamodule.train_dataset):,} | Val: {len(datamodule.val_dataset):,} | Test: {len(datamodule.test_dataset):,}")

    base_model = DOFAUperNetSegmentationClassifier(
        num_classes     = NUM_CLASSES,
        model_type      = "dofa_base_patch16_224",
        checkpoint_path = str(WEIGHTS_PT),
        wavelengths     = DOFA_WAVELENGTHS,
        freeze_backbone = FREEZE_BACKBONE,
        fpn_dim         = 512,
        out_indices     = [2, 5, 8, 11]
    )

    dbr_module = DOFAUperNetMMEarthDBR(
        model          = base_model,
        tile_info_dict = tile_info_dict,
        lr             = args.lr,
        alpha          = args.alpha,
        ema_decay      = args.ema_decay,
        weight_decay   = WEIGHT_DECAY,
        num_classes    = NUM_CLASSES,
    )

    CKPT_MACRO_DIR = CKPT_DIR / "best_miou"
    CKPT_WORST_DIR = CKPT_DIR / "best_worst_miou"
    CKPT_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_WORST_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_macro_cb = ModelCheckpoint(
        monitor                 = "val/miou",
        mode                    = "max",
        dirpath                 = CKPT_MACRO_DIR,
        filename                = f"{mode_str}-miou-epoch={{epoch:02d}}-val_miou={{val/miou:.4f}}",
        auto_insert_metric_name = False,
        save_top_k              = 3,
        save_last               = True,
        save_weights_only       = False,
        verbose                 = False,
    )
    ckpt_worst_cb = ModelCheckpoint(
        monitor                 = "val/worst_case_miou",
        mode                    = "max",
        dirpath                 = CKPT_WORST_DIR,
        filename                = f"{mode_str}-worst-epoch={{epoch:02d}}-worst_miou={{val/worst_case_miou:.4f}}",
        auto_insert_metric_name = False,
        save_top_k              = 3,
        save_last               = False,
        save_weights_only       = False,
        verbose                 = False,
    )

    early_stop_cb = EarlyStopping(
        monitor   = "val/miou",
        mode      = "max",
        patience  = EARLY_STOP_PATIENCE,
        min_delta = EARLY_STOP_MIN_DELTA,
        verbose   = False,
    )
    global_seg_cb   = GlobalSegmentationLogger(num_classes=NUM_CLASSES)
    biome_cb        = BiomeSegmentationLogger(
        tile_info_dict = tile_info_dict,
        num_classes    = NUM_CLASSES,
        class_names    = CLASS_NAMES,
        output_dir     = OUT_DIR,
        test_loader    = None,
    )
    progress_bar_cb = CleanProgressBar()

    tb_logger = TensorBoardLogger(save_dir=BASE_DIR / "DOFA/DOFA_runs", name=mode_str, version=0)

    trainer = L.Trainer(
        accelerator             = "gpu",
        devices                 = 1,
        strategy                = "auto",
        precision               = "bf16-mixed",
        max_epochs              = args.epochs,
        check_val_every_n_epoch = 1,
        log_every_n_steps       = 10,
        logger                  = tb_logger,
        callbacks               = [global_seg_cb, ckpt_macro_cb, ckpt_worst_cb, early_stop_cb, biome_cb, progress_bar_cb],
        enable_checkpointing    = True,
        enable_progress_bar     = True,
        deterministic           = "warn",
        benchmark               = False,
        num_sanity_val_steps    = 2,
    )

    resume_path = args.resume_from
    if resume_path is None:
        last_ckpt = CKPT_MACRO_DIR / "last.ckpt"
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            print(f"[AutoResume] Found existing checkpoint, resuming from: {resume_path}")

    trainer.fit(dbr_module, train_loader, val_loader, ckpt_path=resume_path)

    print(f"\n[Done] Best mIoU Checkpoint       : {ckpt_macro_cb.best_model_path}")
    print(f"[Done] Best mIoU Score            : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f"[Done] Best Worst mIoU Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f"[Done] Best Worst mIoU Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
