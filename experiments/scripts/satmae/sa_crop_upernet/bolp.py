#!/usr/bin/env python3
"""
SatMAE ViT-Base — SA-Crop Semantic Segmentation with Official UPerNet Decoder (ERM)
===================================================================================
Key Features:
  • Task: Semantic Segmentation (m-SA-crop-type, 10 classes, CrossEntropyLoss)
  • Architecture: SatMAE ViT-Base 16 + Multi-Level UPerNet Decoder (layers [2, 5, 8, 11] + PPM + FPN)
  • Method: Biome Orthogonal Linear Projection (BOLP) with Frozen Backbone
  • Biome Grouping: Evaluated across macro biomes from sa_crop_biome_mapping.json.
  • Subgroup Unique-Class Evaluation: mIoU, Macro F1, Pixel Accuracy, and ECE per biome.
  • Metrics Suite: Global & Per-biome mIoU, Worst/Best mIoU, PER, Gini, NFR, EqOdds, DPM, Per-Class IoU Table.
  • JSON Outputs: Saves val_results.json and test_results.json at every epoch.
  • Slim Checkpoints: Strips frozen encoder weights (~1.5 MB per .ckpt file).
"""

from fairrsfm.training.protocol import (MethodTrainingMixin, known_training_dataset, lookup_biome, training_loader, configure_training_protocol, dataset_sample_ids, balanced_group_loader, forward_with_sample_ids)
import sys
import os
import json
import logging
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

import numpy as np
import albumentations
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import geobench
import geobench.task
from fairrsfm.models import SatMAEUperNetSegmentationClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEE LOGGER & CLEAN PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────

class TeeLogger:
    def __init__(self, log_file_path: Path):
        self.terminal = sys.stdout
        self.log_file_path = log_file_path
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")

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

DATA_DIR = Path("/data1/aminur/dataset/geobench")
if not DATA_DIR.exists():
    DATA_DIR = ROOT_DIR / "data/geobench"

BIOME_JSON = ROOT_DIR / "configs/data_mappings/sa_crop_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = Path("/data1/aminur/om/prithvi/FairRSFM/configs/data_mappings/sa_crop_mapping.json")

WEIGHTS_PT = ROOT_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"
if not WEIGHTS_PT.exists():
    WEIGHTS_PT = Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/satmae/pretrain-vit-base-e199.pth")

BENCHMARK_NAME = "segmentation_v1.0"
BENCHMARK_DIR  = DATA_DIR / "segmentation_v1.0"
TASK_KEYWORD   = "crop"
NUM_CLASSES    = 10
IGNORE_INDEX   = -1
IMG_SIZE       = 224

CLASS_NAMES = [f"Crop_{i}" for i in range(NUM_CLASSES)]

SATMAE_SA_CROP_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

SA_CROP_10BAND_MEAN = [16.5267, 26.6364, 36.6966, 46.3887, 58.2815, 63.5758, 68.1836, 69.1426, 83.6268, 65.7677]
SA_CROP_10BAND_STD  = [ 9.3295, 12.6745, 19.4219, 19.4874, 19.9592, 21.5381, 23.0508, 22.3297, 28.1442, 27.2346]

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 0.01
SEED         = 42
NUM_WORKERS  = 8
FREEZE_BACKBONE = True

EARLY_STOP_PATIENCE  = 1000
EARLY_STOP_MIN_DELTA = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 3. MACRO GROUPING BIOME MAP
# ─────────────────────────────────────────────────────────────────────────────

def load_biome_map(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"[BiomeMap] WARNING: {json_path} not found.")
        return {}
    with open(json_path) as f:
        records = json.load(f)

    macro_groups = {
        12: "Mediterranean Forests, Woodlands & Scrub",
        13: "Deserts & Xeric Shrublands",
    }

    biome_map = {}
    for r in records:
        code = r.get("biome_code", None)
        if code is not None:
            try:
                code_int = int(code)
                group_name = macro_groups.get(code_int, r.get("biome_name", "Unknown"))
            except ValueError:
                group_name = r.get("biome_name", "Unknown")
        else:
            group_name = "Unknown"
        biome_map[str(r["geobench_id"])] = group_name

    unique_groups = sorted(set(biome_map.values()) - {"Unknown"})
    print(f"[BiomeMap] Loaded {len(biome_map):,} entries | {len(unique_groups)} biomes: {unique_groups}")
    return biome_map


def print_split_biome_distribution(split_name: str, dataset, biome_map: dict):
    counts = defaultdict(int)
    for sid in dataset_sample_ids(dataset):
        counts[lookup_biome(biome_map, sid)] += 1
    print(f"  [{split_name.upper()} Split] Total: {sum(counts.values()):,}")
    for b in sorted(counts.keys()):
        print(f"    • {b:<48} : {counts[b]:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET & NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class SatMAENormalize:
    def __init__(self, mean: list, std: list):
        mean_arr = np.array(mean, dtype=np.float32)[:, None, None]
        std_arr  = np.array(std,  dtype=np.float32)[:, None, None]
        self.min_val = torch.from_numpy(mean_arr - 2.0 * std_arr)
        self.max_val = torch.from_numpy(mean_arr + 2.0 * std_arr)
        self.range   = self.max_val - self.min_val

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = (tensor - self.min_val) / self.range
        return torch.clamp(norm, 0.0, 1.0)


class GeoBenchSegmentationDataset(Dataset):
    def __init__(self, task, split: str, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = SATMAE_SA_CROP_BAND_INDICES
        self.augment      = augment
        self.normalizer   = SatMAENormalize(SA_CROP_10BAND_MEAN, SA_CROP_10BAND_STD)

        self._spatial_aug = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Resize(IMG_SIZE, IMG_SIZE),
        ])
        self._spatial_val = albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        array, _ = sample.pack_to_3d(
            band_names=self.band_names,
            resample=False,
            fill_value=None,
            resample_order=3,
        )
        image = array.astype(np.float32)[:, :, self.band_indices]
        mask  = sample.label.data.astype(np.int64)

        tfm = self._spatial_aug if self.augment else self._spatial_val
        out = tfm(image=image, mask=mask)
        image = out["image"]
        mask  = out["mask"]

        x = torch.from_numpy(image).permute(2, 0, 1).float()
        x = self.normalizer(x)
        y = torch.from_numpy(mask).long()
        sid = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXACT METRIC FUNCTIONS
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
        b = (probs_max > lo) & (probs_max <= hi)
        if b.sum() > 0:
            ece += np.abs(correct[b].mean() - probs_max[b].mean()) * (b.sum() / n)
    return float(ece)


def compute_gini(values: list) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    array = np.sort(np.array(vals, dtype=np.float64))
    index = np.arange(1, len(array) + 1)
    n = len(array)
    sum_val = np.sum(array)
    if sum_val == 0:
        return 0.0
    return float(np.sum((2 * index - n - 1) * array) / (n * sum_val))


def compute_eqodds_seg(biome_data: dict, num_classes: int, ignore_index: int = -1) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown"])
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
# 6. LIGHTNING MODULE (SATMAE UPERNET ERM)
# ─────────────────────────────────────────────────────────────────────────────

class SatMAEUperNetSACropModule(MethodTrainingMixin, L.LightningModule):
    def __init__(self, model: SatMAEUperNetSegmentationClassifier, lr: float, weight_decay: float = WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model   = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        if FREEZE_BACKBONE:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
            self.model.backbone.eval()
            frozen    = sum(p.numel() for p in self.model.backbone.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n[Model] Frozen    : {frozen/1e6:.1f}M  (SatMAE ViT encoder)")
            print(f"[Model] Trainable : {trainable/1e6:.1f}M  (UPerNet decoder + head)")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/miou"}}


# ─────────────────────────────────────────────────────────────────────────────
# 7. LOGGERS & METRIC CALLBACKS
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
        pl_module._g_labels = None
        pl_module._g_preds  = None


class BiomeSegmentationLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader=None):
        super().__init__()
        self.biome_map    = biome_map
        self.num_classes  = num_classes
        self.class_names  = class_names
        self.output_dir   = output_dir
        self.test_loader  = test_loader
        self.val_history  = []
        self.test_history = []
        self._val_data    = None

    def _compute_and_log(self, data_dict: dict, epoch: int, tb, prefix: str, label: str, pl_module=None):
        known_biome_data = {b: data_dict[b] for b in data_dict if b != "Unknown" and len(data_dict[b]["true"]) > 0}
        all_biomes = sorted(data_dict.keys())
        all_true = np.concatenate([np.concatenate(data_dict[b]["true"]) for b in all_biomes if len(data_dict[b]["true"]) > 0])
        all_pred = np.concatenate([np.concatenate(data_dict[b]["pred"]) for b in all_biomes if len(data_dict[b]["pred"]) > 0])
        all_pmax = np.concatenate([np.concatenate(data_dict[b]["probs_max"]) for b in all_biomes if len(data_dict[b]["probs_max"]) > 0])

        overall_miou = compute_miou(all_true, all_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        overall_f1   = compute_macro_f1_seg(all_true, all_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        mask = all_true != IGNORE_INDEX
        overall_acc  = float((all_pred[mask] == all_true[mask]).mean()) if mask.sum() > 0 else float("nan")
        overall_ece  = compute_ece_seg(all_true, all_pmax, all_pred, ignore_index=IGNORE_INDEX)

        if pl_module is not None and label == "val":
            pl_module.log("val/benchmark_miou", overall_miou, prog_bar=True, add_dataloader_idx=False)

        print(f"\n{'═'*82}")
        print(f"  [{label.upper()} EVALUATION — Epoch {epoch:02d}] Overall mIoU: {overall_miou:.4f} | Macro F1: {overall_f1:.4f} | Acc: {overall_acc:.4f} | ECE: {overall_ece:.4f}")
        print(f"{'─'*82}")

        group_mious, group_f1s, group_accs, group_eces, group_counts = {}, {}, {}, {}, {}
        for b in sorted(known_biome_data.keys()):
            yt = np.concatenate(known_biome_data[b]["true"])
            yp = np.concatenate(known_biome_data[b]["pred"])
            ym = np.concatenate(known_biome_data[b]["probs_max"])
            group_counts[b] = len(known_biome_data[b]["sample_ids"])
            group_mious[b]  = compute_miou(yt, yp, self.num_classes, ignore_index=IGNORE_INDEX)
            group_f1s[b]    = compute_macro_f1_seg(yt, yp, self.num_classes, ignore_index=IGNORE_INDEX)
            valid_mask      = yt != IGNORE_INDEX
            group_accs[b]   = float((yp[valid_mask] == yt[valid_mask]).mean()) if valid_mask.sum() > 0 else float("nan")
            group_eces[b]   = compute_ece_seg(yt, ym, yp, ignore_index=IGNORE_INDEX)

            print(f"    • {b:<45} : mIoU={group_mious[b]:.4f} | F1={group_f1s[b]:.4f} | Acc={group_accs[b]:.4f} | ECE={group_eces[b]:.4f} ({group_counts[b]} tiles)")

        valid_mious = [v for v in group_mious.values() if v is not None and not np.isnan(v)]
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

        eqodds_disp = compute_eqodds_seg(known_biome_data, self.num_classes, ignore_index=IGNORE_INDEX)
        dpm_disp    = compute_demographic_parity_seg(known_biome_data, self.num_classes, ignore_index=IGNORE_INDEX)

        print(f"\n  ── Fairness & Disparity Summary ───────────────────────────────────────────")
        print(f"    mIoU (Macro-avg over groups)     : {mean_miou:.4f}" if not np.isnan(mean_miou) else "    mIoU     : N/A")
        print(f"    Worst-Case Group mIoU            : {worst_miou:.4f}" if not np.isnan(worst_miou) else "    Worst mIoU : N/A")
        print(f"    Best-Case Group mIoU             : {best_miou:.4f}" if not np.isnan(best_miou) else "    Best mIoU  : N/A")
        print(f"    Performance Equity Ratio (PER)   : {per_ratio:.4f}" if not np.isnan(per_ratio) else "    PER        : N/A")
        print(f"    Group mIoU Std Dev               : {std_miou:.4f}" if not np.isnan(std_miou) else "    Std Dev    : N/A")
        print(f"    Gini Coefficient                 : {gini_coef:.4f}" if not np.isnan(gini_coef) else "    Gini       : N/A")
        print(f"    Equalized Odds Disparity         : {eqodds_disp:.4f}" if not np.isnan(eqodds_disp) else "    EqOdds     : N/A")
        print(f"    Demographic Parity (DPM)         : {dpm_disp:.4f}" if not np.isnan(dpm_disp) else "    DPM        : N/A")
        print(f"{'═'*82}\n")

        if pl_module is not None and label == "val":
            if not np.isnan(worst_miou):
                pl_module.log("val/worst_case_miou", worst_miou, prog_bar=False, add_dataloader_idx=False)
            if not np.isnan(mean_miou):
                pl_module.log("val/group_mean_miou", mean_miou, prog_bar=False, add_dataloader_idx=False)

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
            "equalized_odds_disparity": eqodds_disp if not np.isnan(eqodds_disp) else None,
            "demographic_parity_difference": dpm_disp if not np.isnan(dpm_disp) else None,
            "per_group_metrics": {
                g: {
                    "mIoU": group_mious[g],
                    "macro_f1": group_f1s[g],
                    "pixel_acc": group_accs[g],
                    "ece": group_eces[g],
                    "sample_count": group_counts[g],
                }
                for g in group_mious
            }
        }

        if label == "val":
            self.val_history.append(record)
            with open(self.output_dir / "val_results.json", "w") as f:
                json.dump(self.val_history, f, indent=2)
        elif label == "test":
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
            biome = self.biome_map.get(str(sid), "Unknown")
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
                x, y, sids = batch
                x = x.to(device)
                logits = pl_module(x)
                probs = F.softmax(logits.float(), dim=1)
                p_max, preds = probs.max(dim=1)

                preds_np  = preds.cpu().numpy()
                labels_np = y.numpy()
                p_max_np  = p_max.cpu().numpy()

                for sid, label_arr, pred_arr, p_max_arr in zip(sids, labels_np, preds_np, p_max_np):
                    biome = self.biome_map.get(str(sid), "Unknown")
                    test_data[biome]["true"].append(label_arr.flatten())
                    test_data[biome]["pred"].append(pred_arr.flatten())
                    test_data[biome]["probs_max"].append(p_max_arr.flatten())
                    test_data[biome]["sample_ids"].append(sid)

        self._compute_and_log(test_data, epoch, tb, "test", "test", pl_module=None)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SatMAE SA-Crop UPerNet ERM Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    mode_str = "SatMAE_sa_crop_upernet_bolp"
    OUT_DIR  = ROOT_DIR / f"runs/satmae/{mode_str}/seed_{seed}"
    CKPT_DIR = OUT_DIR  / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset      : m-SA-crop-type (10 classes)")
    print(f"[Config] Architecture : SatMAE ViT-Base 16 (Official UPerNet Decoder)")
    print(f"[Config] Mode         : Biome Orthogonal Linear Projection (BOLP)")
    print(f"[Config] LR           : {args.lr}")
    print(f"[Config] Seed         : {seed}")
    print(f"[Config] Grouping     : Biome Grouping (sa_crop_mapping.json)")
    print(f"[Logging] Console log : {OUT_DIR / 'train.log'}")
    print(f"[Logging] JSON dir    : {OUT_DIR}")

    biome_map = load_biome_map(BIOME_JSON)

    geobench.GEO_BENCH_DIR         = str(DATA_DIR)
    geobench.task.GEO_BENCH_DIR    = DATA_DIR
    os.environ["GEO_BENCH_DIR"]    = str(DATA_DIR)

    tasks = list(geobench.task_iterator(
        benchmark_name=BENCHMARK_NAME,
        benchmark_dir=str(BENCHMARK_DIR)
    ))

    task = None
    for t in tasks:
        if TASK_KEYWORD in t.dataset_name.lower():
            task = t
            break
    assert task is not None, f"Task '{TASK_KEYWORD}' not found in {[t.dataset_name for t in tasks]}"
    print(f"[Data] Loaded Task: {task.dataset_name}")

    train_ds = GeoBenchSegmentationDataset(task, split="train", augment=True)
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchSegmentationDataset(task, split="valid", augment=False)
    test_ds  = GeoBenchSegmentationDataset(task, split="test",  augment=False)

    print("\n[Dataset Partition Summary]")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds,   biome_map)
    print_split_biome_distribution("test",  test_ds,  biome_map)

    train_loader = training_loader(train_ds, method='bolp', groups=protocol_groups, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = SatMAEUperNetSegmentationClassifier(
        num_classes     = NUM_CLASSES,
        model_type      = "vit_base_patch16",
        checkpoint_path = str(WEIGHTS_PT),
        freeze_backbone = FREEZE_BACKBONE,
        fpn_dim         = 512,
        out_indices     = [2, 5, 8, 11],
        img_size        = IMG_SIZE,
        patch_size      = 16,
    )

    erm_module = SatMAEUperNetSACropModule(
        model        = base_model,
        lr           = args.lr,
        weight_decay = WEIGHT_DECAY,
    )
    configure_training_protocol(erm_module, 'bolp', 'sa_crop', protocol_groups, args, ignore_index=IGNORE_INDEX, lookup=lambda sid: lookup_biome(biome_map, sid))
    args.method_protocol = erm_module.method_protocol.config

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
        monitor   = "val/benchmark_miou",
        mode      = "max",
        patience  = EARLY_STOP_PATIENCE,
        min_delta = EARLY_STOP_MIN_DELTA,
        verbose   = False,
    )
    global_seg_cb   = GlobalSegmentationLogger(num_classes=NUM_CLASSES)
    biome_cb        = BiomeSegmentationLogger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = test_loader,
    )
    progress_bar_cb = CleanProgressBar()

    tb_logger = TensorBoardLogger(save_dir=OUT_DIR / "tb_logs", name=mode_str, version=0)

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

    trainer.fit(erm_module, train_loader, val_loader, ckpt_path=resume_path)

    print(f"\n[Done] Best mIoU Checkpoint       : {ckpt_macro_cb.best_model_path}")
    print(f"[Done] Best mIoU Score            : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f"[Done] Best Worst mIoU Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f"[Done] Best Worst mIoU Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
