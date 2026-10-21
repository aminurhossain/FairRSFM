#!/usr/bin/env python3
"""
DOFA ViT-Base 16 — BigEarthNet Multi-Label Classification (Group DRO Linear Probe)
==================================================================================
Key Features:
  • Task: Multi-label Classification (m-bigearthnet, 43 classes, BCEWithLogitsLoss)
  • Architecture: DOFA ViT-Base 16 (224x224) with official 12-Band DOFA Spectral Conditioning
  • Method: Group Distributionally Robust Optimization (Group DRO, Sagawa et al. 2020)
  • Optimization: Online exponentiated gradient ascent on group weights (q_g <- q_g * exp(eta_q * loss_g))
  • Probe Head: BatchNorm1d(768, affine=False) + Linear(768, 43)
  • Wavelengths (µm): [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
  • Normalization: Exact GeoBench Task Statistics (12 Sentinel-2 Bands)
  • Augmentation: RandomResizedCrop(224, scale=(0.8, 1.0)) + RandomHorizontalFlip(0.5)
  • Optimizer: LARS (Layer-wise Adaptive Rate Scaling, LR=5.0, Base LR=10.0 for batch size 128)
  • Grouping: Macro Grouping (bigearthnet_master_mapping.json)
  • Subgroup Unique-Class Macro F1: Exact evaluation matching eval_all_models.py (labels=present)
  • Optimal Threshold Search: Per-class validation threshold search (0.05 to 0.95)
  • Metrics Suite: Macro/Micro F1 (0.5 & opt), mAP, Accuracy, ECE, PER, Gini, NFR, EqOdds, DPM.
  • Slim Checkpoints: Strips frozen encoder weights (~120 KB per .ckpt file).
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

import numpy as np
import torchvision.transforms.v2 as transforms
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import geobench
import geobench.task
from models.dofa_wrapper import DOFAClassifier
from models.lars import LARS

from sklearn.metrics import average_precision_score, f1_score, accuracy_score


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

BASE_DIR = ROOT_DIR
DATA_DIR = BASE_DIR / "Geobench"
if not DATA_DIR.exists() and os.environ.get("GEO_BENCH_DIR"):
    DATA_DIR = Path(os.environ["GEO_BENCH_DIR"])
if not DATA_DIR.exists():
    DATA_DIR = Path("/data1/aminur/dataset/geobench")

BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/bigearthnet_master_mapping.json"
WEIGHTS_PT = BASE_DIR / "checkpoints/dofa/dofa_base_patch16_224.pth"

BENCHMARK_NAME = "classification_v1.0"
BENCHMARK_DIR  = DATA_DIR / "classification_v1.0"
TASK_KEYWORD   = "bigearthnet"
NUM_CLASSES    = 43
MULTILABEL     = True

CLASS_NAMES = [
    "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops", "Pastures",
    "Complex cultivation patterns", "Land principally occupied by agriculture", "Agro-forestry areas",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grasslands", "Moors and heathland",
    "Sclerophyllous vegetation", "Transitional woodland-shrub", "Beaches, dunes, sands", "Inland marshes",
    "Peat bogs", "Salt marshes", "Salines", "Intertidal flats", "Water courses", "Water bodies",
    "Coastal lagoons", "Estuaries", "Sea and ocean", "Glaciers and perpetual snow", "Bare rocks",
    "Sparsely vegetated areas", "Burnt areas", "Terrestrial wetlands", "Inland wetlands NEC", "Marine waters",
    "Wetlands NEC", "Mineral extraction sites", "Dump sites", "Construction sites", "Green urban areas",
    "Sport and leisure facilities", "Road and rail", "Port areas", "Airports", "Other land",
]

# ── Official DOFA 12-Band Spectral Configuration for BigEarthNet ──
DOFA_BAND_NAMES = [
    "01 - Coastal aerosol", "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "09 - Water vapour",
    "11 - SWIR", "12 - SWIR"
]
DOFA_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]

# Exact GeoBench BigEarthNet Task Normalization Statistics
BIGEARTHNET_DOFA_MEAN = [386.6516, 488.9927, 714.6071, 738.2603, 1114.4219, 1910.1897, 2191.4824, 2334.2920, 2392.9136, 2367.2874, 1902.6918, 1261.0698]
BIGEARTHNET_DOFA_STD  = [467.3123, 510.7866, 551.7964, 691.7815,  700.4516,  976.7469, 1134.8865, 1238.0713, 1215.9766, 1153.8583, 1117.0094,  894.7390]

BATCH_SIZE   = 128
EPOCHS       = 50
BLR          = 10.0
LR           = BLR * BATCH_SIZE / 256.0  # 5.0 for batch size 128
WEIGHT_DECAY = 0.0
SEED         = 42
NUM_WORKERS  = 8
FREEZE_BACKBONE = True

EARLY_STOP_PATIENCE = 1000
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
        1: "high biomass", 3: "high biomass", 5: "high biomass",
        2: "High amplitude phenological", 6: "High amplitude phenological", 4: "High amplitude phenological",
        7: "transitional herbaceous & scrub", 8: "transitional herbaceous & scrub", 12: "transitional herbaceous & scrub",
        10: "cryospheric & short-cycle", 11: "cryospheric & short-cycle",
        13: "xeric & mineralogical",
        9: "hydrologically modulated", 14: "hydrologically modulated"
    }

    biome_map = {}
    for r in records:
        code = r.get("biome_code", None)
        if code is not None:
            try:
                code_int = int(code)
                group_name = macro_groups.get(code_int, "Unknown")
            except ValueError:
                group_name = "Unknown"
        else:
            group_name = "Unknown"
        biome_map[str(r["geobench_id"])] = group_name

    unique_groups = sorted(set(biome_map.values()) - {"Unknown"})
    print(f"[BiomeMap] Loaded {len(biome_map):,} entries | {len(unique_groups)} macro biome groups: {unique_groups}")
    return biome_map


def print_split_biome_distribution(split_name: str, dataset, biome_map: dict):
    counts = defaultdict(int)
    raw_ds = getattr(dataset, "dataset", dataset)
    
    if hasattr(raw_ds, "active_partition") and hasattr(raw_ds, "split") and raw_ds.split in raw_ds.active_partition.partition_dict:
        sample_names = raw_ds.active_partition.partition_dict[raw_ds.split]
    elif hasattr(raw_ds, "active_partition") and split_name in raw_ds.active_partition.partition_dict:
        sample_names = raw_ds.active_partition.partition_dict[split_name]
    else:
        sample_names = [raw_ds[i].sample_name for i in range(len(raw_ds))]
        
    for sid in sample_names:
        b = biome_map.get(str(sid), "Unknown")
        counts[b] += 1
        
    print(f"  [{split_name.upper()} Split] Total samples: {len(sample_names):,}")
    for b in sorted(counts.keys()):
        print(f"    • {b:<44} : {counts[b]:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEOBENCH DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class GeoBenchMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, augment: bool = False, norm_stats=None):
        super().__init__()
        self.dataset    = task.get_dataset(split=split, band_names=DOFA_BAND_NAMES)
        self.band_names = DOFA_BAND_NAMES
        self.augment    = augment

        mean, std = norm_stats if norm_stats is not None else (BIGEARTHNET_DOFA_MEAN, BIGEARTHNET_DOFA_STD)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(std,  dtype=torch.float32)[:, None, None]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224), antialias=True),
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
        x = torch.from_numpy(array.astype("float32")).permute(2, 0, 1)
        x = self.transform(x)
        x = (x - self.mean) / self.std

        y   = torch.tensor(sample.label, dtype=torch.float32)
        sid = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXACT FAIRNESS & METRIC COMPUTATION (MATCHING eval_all_models.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_subgroup_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    present = [c for c in range(num_classes) if y_true[:, c].sum() > 0]
    if not present:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true[:, present].astype(float), y_pred[:, present].astype(float), average="macro", zero_division=0))


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    f1s = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        yt_c = y_true[:, c]
        yp_c = y_pred[:, c]
        if yt_c.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s[c] = float(f1_score(yt_c, yp_c, zero_division=0))
    return f1s


def find_optimal_thresholds(all_labels: np.ndarray, all_probs: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        if all_labels[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            preds_c = (all_probs[:, c] >= t).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1 = f1_score(all_labels[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = float(best_t)
    return thresholds


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


def compute_ece_metric(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_max = y_true.max(axis=1)
    p_max = probs.max(axis=1)
    pred_max = preds.max(axis=1)
    correct = (pred_max == y_max).astype(float)
    ece_val, n = 0.0, len(y_true)
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (p_max >= lo) & (p_max < hi)
        if mask.sum() == 0:
            continue
        ece_val += (mask.sum() / n) * abs(correct[mask].mean() - p_max[mask].mean())
    return float(ece_val)


def compute_equalized_odds(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0]
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.array(biome_data[b]["true"])
        yp = np.array(biome_data[b]["pred_opt"])
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            yt_c = yt[:, c]
            yp_c = yp[:, c]
            tp = int(((yp_c == 1) & (yt_c == 1)).sum())
            fn = int(((yp_c == 0) & (yt_c == 1)).sum())
            fp = int(((yp_c == 1) & (yt_c == 0)).sum())
            tn = int(((yp_c == 0) & (yt_c == 0)).sum())
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


def compute_demographic_parity(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0]
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.array(biome_data[b]["pred_opt"])
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = yp[:, c].mean()
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i+1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 6. LIGHTNING MODULE (GROUP DRO MULTI-LABEL)
# ─────────────────────────────────────────────────────────────────────────────

class MultiLabelGroupDROLinearProbe(L.LightningModule):
    def __init__(self, model: DOFAClassifier, biome_map: dict, lr: float,
                 eta_q: float = 0.01, weight_decay: float = WEIGHT_DECAY, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "biome_map"])
        self.model       = model
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.eta_q       = eta_q

        known_biomes = sorted(set(biome_map.values()) - {"Unknown"})
        self.biome_to_idx = {b: i for i, b in enumerate(known_biomes)}
        self.num_biomes = len(known_biomes)

        self.register_buffer("q", torch.ones(self.num_biomes, dtype=torch.float32) / self.num_biomes)
        self._last_group_losses = None

        if FREEZE_BACKBONE:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()
            frozen    = sum(p.numel() for p in self.model.encoder.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n[Model] Frozen    : {frozen/1e6:.1f}M  (encoder)")
            print(f"[Model] Trainable : {trainable/1e6:.1f}M  (head)")

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
        return self.model(x, DOFA_WAVELENGTHS)

    def training_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        if FREEZE_BACKBONE:
            self.model.encoder.eval()
        logits = self(x)
        # Per-sample multi-label BCE loss
        raw_losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(dim=-1)

        group_losses = torch.zeros(self.num_biomes, device=self.device)
        group_counts = torch.zeros(self.num_biomes, device=self.device)

        for i, sid in enumerate(sample_ids):
            biome = self.biome_map.get(str(sid), "Unknown")
            b_idx = self.biome_to_idx.get(biome, None)
            if b_idx is not None:
                group_losses[b_idx] += raw_losses[i]
                group_counts[b_idx] += 1

        active_mask = group_counts > 0
        avg_group_losses = torch.zeros_like(group_losses)
        avg_group_losses[active_mask] = group_losses[active_mask] / group_counts[active_mask]

        self._last_group_losses = avg_group_losses.detach()
        self._last_active_mask = active_mask

        q_detached = self.q.clone().detach()
        loss = torch.sum(q_detached * avg_group_losses)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._last_group_losses is not None and self._last_active_mask is not None:
            with torch.no_grad():
                q_updated = self.q.clone()
                q_updated[self._last_active_mask] *= torch.exp(self.eta_q * self._last_group_losses[self._last_active_mask])
                q_sum = q_updated.sum()
                if q_sum > 0:
                    self.q.copy_(q_updated / q_sum)

    def validation_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss   = F.binary_cross_entropy_with_logits(logits, y)
        self.log("val/loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def test_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss   = F.binary_cross_entropy_with_logits(logits, y)
        self.log("test/loss", loss, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def configure_optimizers(self):
        trainable = [p for p in self.model.head.parameters() if p.requires_grad]
        optimizer = LARS(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}}


# ─────────────────────────────────────────────────────────────────────────────
# 7. LOGGERS & METRIC CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMacroF1Logger(Callback):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._gf1_labels = []
        pl_module._gf1_probs  = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        probs  = torch.sigmoid(logits.float()).cpu().numpy()
        labels = y.cpu().numpy()
        pl_module._gf1_labels.append(labels)
        pl_module._gf1_probs.append(probs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._gf1_labels:
            return
        y_true  = np.concatenate(pl_module._gf1_labels, axis=0)
        probs   = np.concatenate(pl_module._gf1_probs, axis=0)
        preds05 = (probs >= 0.5).astype(int)

        global_f1_05 = float(f1_score(y_true, preds05, average="macro", zero_division=0))
        global_micro_f1 = float(f1_score(y_true, preds05, average="micro", zero_division=0))
        global_acc = float(((preds05 == y_true).all(axis=1)).mean())
        
        pl_module.log("val/macro_f1", global_f1_05, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/micro_f1", global_micro_f1, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/micro_acc", global_acc, prog_bar=False, add_dataloader_idx=False)
        print(f"\n  [GlobalMacroF1] epoch {trainer.current_epoch} → val/macro_f1 = {global_f1_05:.4f} | val/micro_f1 = {global_micro_f1:.4f} | val/acc = {global_acc:.4f}")
        pl_module._gf1_labels = None
        pl_module._gf1_probs  = None


class BiomeMultiLabelLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader=None):
        super().__init__()
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.class_names = class_names
        self.output_dir  = Path(output_dir)
        self.test_loader = test_loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_history  = []
        self.test_history = []
        self.opt_thresholds = np.full(num_classes, 0.5)

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label, pl_module=None):
        all_yt = np.concatenate([d["true"] for d in biome_data.values()], axis=0)
        all_pr = np.concatenate([d["probs"] for d in biome_data.values()], axis=0)

        if label == "val":
            self.opt_thresholds = find_optimal_thresholds(all_yt, all_pr, self.num_classes)

        all_yp_opt = (all_pr >= self.opt_thresholds[None, :]).astype(int)
        all_yp_05  = (all_pr >= 0.5).astype(int)

        overall_f1_opt = float(f1_score(all_yt, all_yp_opt, average="macro", zero_division=0))
        overall_f1_05  = float(f1_score(all_yt, all_yp_05,  average="macro", zero_division=0))
        overall_micro_f1 = float(f1_score(all_yt, all_yp_opt, average="micro", zero_division=0))
        overall_acc = float(((all_yp_opt == all_yt).all(axis=1)).mean())
        overall_map = float(average_precision_score(all_yt, all_pr, average="macro"))

        known_biome_data = {b: d for b, d in biome_data.items() if b != "Unknown"}

        print(f"\n{'═'*82}")
        print(f"  Multi-Label Group Metrics [{label}] — epoch {epoch} (Total Samples: {len(all_yt):,})")
        print(f"  Overall Exact Acc: {overall_acc*100:.2f}% | Macro F1 (Opt): {overall_f1_opt*100:.2f}% | Macro F1 (0.5): {overall_f1_05*100:.2f}% | mAP: {overall_map*100:.2f}%")
        print(f"{'═'*82}")
        print(f"\n  ── Per-group Multi-Label Metrics ──────────────────────────────────────────")
        print(f"  {'Macro Group':<40}  {'F1 (Opt)':>8}  {'F1 (0.5)':>8}  {'mAP':>7}  {'Exact Acc':>9}  {'ECE':>7}")
        print(f"  {'─'*40}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*7}")

        group_f1_opts = {}
        group_f1_05s  = {}
        group_maps    = {}
        group_accs    = {}
        group_eces    = {}
        group_counts  = {}

        biomes_sorted = sorted(known_biome_data.keys())
        n_biome = len(biomes_sorted)
        f1_matrix = np.full((self.num_classes, n_biome), float("nan"))

        for j, group in enumerate(biomes_sorted):
            d      = known_biome_data[group]
            n      = len(d["true"])
            y_true = np.array(d["true"])
            probs  = np.array(d["probs"])
            yp_opt = (probs >= self.opt_thresholds[None, :]).astype(int)
            yp_05  = (probs >= 0.5).astype(int)

            d["pred_opt"] = yp_opt.tolist()
            d["pred_05"]  = yp_05.tolist()

            # Unique classes present in this biome
            f1_opt = compute_subgroup_macro_f1(y_true, yp_opt, self.num_classes)
            f1_05  = compute_subgroup_macro_f1(y_true, yp_05,  self.num_classes)
            
            present_c = [c for c in range(self.num_classes) if y_true[:, c].sum() > 0]
            map_score = float(average_precision_score(y_true[:, present_c], probs[:, present_c], average="macro")) if present_c else float("nan")
            
            acc = float(((yp_opt == y_true).all(axis=1)).mean())
            ece = compute_ece_metric(y_true, probs, yp_opt)

            group_f1_opts[group] = f1_opt
            group_f1_05s[group]  = f1_05
            group_maps[group]    = map_score
            group_accs[group]    = acc
            group_eces[group]    = ece
            group_counts[group]  = n

            # Per-class F1 for this biome
            f1_matrix[:, j] = compute_per_class_f1(y_true, yp_opt, self.num_classes)

            print(f"  {group:<40}  {f1_opt:>8.4f}  {f1_05:>8.4f}  {map_score:>7.4f}  {acc:>9.4f}  {ece:>7.4f}")

        # Per-class disparity
        class_disparity = {}
        per_class_f1_table = {}
        for c in range(self.num_classes):
            cname = self.class_names[c] if c < len(self.class_names) else f"cls_{c}"
            valid = f1_matrix[c, ~np.isnan(f1_matrix[c])]
            disp = float(valid.max() - valid.min()) if len(valid) >= 2 else float("nan")
            class_disparity[cname] = disp
            per_class_f1_table[cname] = {biomes_sorted[j]: float(f1_matrix[c, j]) for j in range(n_biome) if not np.isnan(f1_matrix[c, j])}

        valid_disp = [v for v in class_disparity.values() if not np.isnan(v)]
        mean_class_disp = float(np.mean(valid_disp)) if valid_disp else float("nan")
        max_class_disp  = float(np.max(valid_disp))  if valid_disp else float("nan")

        valid_f1s  = [v for v in group_f1_opts.values() if v is not None and not np.isnan(v)]
        valid_maps = [v for v in group_maps.values() if v is not None and not np.isnan(v)]
        valid_accs = [v for v in group_accs.values() if v is not None and not np.isnan(v)]
        valid_eces = [v for v in group_eces.values() if v is not None and not np.isnan(v)]

        if len(valid_f1s) >= 2:
            worst_f1  = float(np.min(valid_f1s))
            best_f1   = float(np.max(valid_f1s))
            mean_f1   = float(np.mean(valid_f1s))
            std_f1    = float(np.std(valid_f1s))
            per_ratio = float(worst_f1 / best_f1) if best_f1 > 0 else float("nan")
            gini_coef = compute_gini(valid_f1s)
            nfr       = float((best_f1 - worst_f1) / mean_f1) if mean_f1 > 0 else float("nan")
        else:
            worst_f1 = best_f1 = mean_f1 = std_f1 = per_ratio = gini_coef = nfr = float("nan")

        worst_map = float(np.min(valid_maps)) if len(valid_maps) >= 2 else float("nan")
        worst_acc = float(np.min(valid_accs)) if len(valid_accs) >= 2 else float("nan")
        ece_disparity = float(np.max(valid_eces) - np.min(valid_eces)) if len(valid_eces) >= 2 else float("nan")

        eqodds_disp = compute_equalized_odds(known_biome_data, self.num_classes)
        dpm_disp    = compute_demographic_parity(known_biome_data, self.num_classes)

        print(f"\n  ── Fairness & Disparity Summary ───────────────────────────────────────────")
        print(f"    Macro F1 (Macro-avg over groups) : {mean_f1:.4f}" if not np.isnan(mean_f1) else "    Macro F1 : N/A")
        print(f"    Worst-Case Group Macro F1        : {worst_f1:.4f}" if not np.isnan(worst_f1) else "    Worst F1 : N/A")
        print(f"    Best-Case Group Macro F1         : {best_f1:.4f}" if not np.isnan(best_f1) else "    Best F1  : N/A")
        print(f"    Performance Equity Ratio (PER)   : {per_ratio:.4f}" if not np.isnan(per_ratio) else "    PER      : N/A")
        print(f"    Group Macro F1 Std Dev           : {std_f1:.4f}" if not np.isnan(std_f1) else "    Std Dev  : N/A")
        print(f"    Gini Coefficient                 : {gini_coef:.4f}" if not np.isnan(gini_coef) else "    Gini     : N/A")
        print(f"    ECE Disparity (Max - Min)        : {ece_disparity:.4f}" if not np.isnan(ece_disparity) else "    ECE Disp : N/A")
        print(f"    Equalized Odds Disparity         : {eqodds_disp:.4f}" if not np.isnan(eqodds_disp) else "    EqOdds   : N/A")
        print(f"    Demographic Parity (DPM)         : {dpm_disp:.4f}" if not np.isnan(dpm_disp) else "    DPM      : N/A")
        print(f"{'═'*82}\n")

        if pl_module is not None and label == "val":
            if not np.isnan(worst_f1):
                pl_module.log("val/worst_case_macro_f1", worst_f1, prog_bar=False, add_dataloader_idx=False)
            if not np.isnan(mean_f1):
                pl_module.log("val/group_mean_macro_f1", mean_f1, prog_bar=False, add_dataloader_idx=False)

        if tb:
            p = tb_prefix
            tb.add_scalar(f"{p}/overall_macro_f1_opt", overall_f1_opt, epoch)
            tb.add_scalar(f"{p}/overall_macro_f1_05", overall_f1_05, epoch)
            tb.add_scalar(f"{p}/overall_micro_f1", overall_micro_f1, epoch)
            tb.add_scalar(f"{p}/overall_mAP", overall_map, epoch)
            tb.add_scalar(f"{p}/overall_accuracy", overall_acc, epoch)
            if not np.isnan(worst_f1):
                tb.add_scalar(f"{p}/worst_case_macro_f1", worst_f1, epoch)
                tb.add_scalar(f"{p}/best_case_macro_f1", best_f1, epoch)
                tb.add_scalar(f"{p}/group_macro_f1_std", std_f1, epoch)
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
            "overall_macro_f1_opt": overall_f1_opt,
            "overall_macro_f1_05": overall_f1_05,
            "overall_micro_f1": overall_micro_f1,
            "overall_mAP": overall_map,
            "overall_accuracy": overall_acc,
            "optimal_thresholds": self.opt_thresholds.tolist(),
            "worst_case_macro_f1": worst_f1 if not np.isnan(worst_f1) else None,
            "best_case_macro_f1": best_f1 if not np.isnan(best_f1) else None,
            "group_mean_macro_f1": mean_f1 if not np.isnan(mean_f1) else None,
            "group_std_macro_f1": std_f1 if not np.isnan(std_f1) else None,
            "per_ratio": per_ratio if not np.isnan(per_ratio) else None,
            "normalized_fairness_range": nfr if not np.isnan(nfr) else None,
            "gini_coefficient": gini_coef if not np.isnan(gini_coef) else None,
            "worst_case_mAP": worst_map if not np.isnan(worst_map) else None,
            "worst_case_accuracy": worst_acc if not np.isnan(worst_acc) else None,
            "ece_disparity": ece_disparity if not np.isnan(ece_disparity) else None,
            "equalized_odds_disparity": eqodds_disp if not np.isnan(eqodds_disp) else None,
            "demographic_parity_difference": dpm_disp if not np.isnan(dpm_disp) else None,
            "mean_class_disparity": mean_class_disp if not np.isnan(mean_class_disp) else None,
            "max_class_disparity": max_class_disp if not np.isnan(max_class_disp) else None,
            "per_group_metrics": {
                g: {
                    "macro_f1_opt": group_f1_opts[g],
                    "macro_f1_05": group_f1_05s[g],
                    "mAP": group_maps[g],
                    "accuracy": group_accs[g],
                    "ece": group_eces[g],
                    "sample_count": group_counts[g],
                }
                for g in group_f1_opts
            },
            "class_disparities": class_disparity,
            "per_class_f1_table": per_class_f1_table
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
        self._val_data = defaultdict(lambda: {"true": [], "probs": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        sids   = outputs["sample_ids"]
        probs  = torch.sigmoid(logits.float()).cpu().numpy()
        labels = y.cpu().numpy()

        for sid, label_val, prob_val in zip(sids, labels, probs):
            biome = self.biome_map.get(str(sid), "Unknown")
            self._val_data[biome]["true"].append(label_val.tolist())
            self._val_data[biome]["probs"].append(prob_val.tolist())

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
        test_data = defaultdict(lambda: {"true": [], "probs": []})
        device = pl_module.device

        with torch.no_grad():
            for batch in self.test_loader:
                x, y, sids = batch
                x = x.to(device)
                logits = pl_module(x)
                probs  = torch.sigmoid(logits.float()).cpu().numpy()
                labels = y.numpy()

                for sid, label_val, prob_val in zip(sids, labels, probs):
                    biome = self.biome_map.get(str(sid), "Unknown")
                    test_data[biome]["true"].append(label_val.tolist())
                    test_data[biome]["probs"].append(prob_val.tolist())

        self._compute_and_log(test_data, epoch, tb, "test", "test", pl_module=None)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DOFA BigEarthNet Multi-Label Classification Group DRO Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--blr", type=float, default=BLR, help="Base learning rate (default: 10.0)")
    parser.add_argument("--eta-q", type=float, default=0.01, help="Group DRO step size for group weights q (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    lr = args.blr * BATCH_SIZE / 256.0

    mode_str = "DOFA_bigearthnet_lp_groupdro"
    OUT_DIR  = BASE_DIR / f"DOFA/DOFA_runs/{mode_str}/seed_{seed}"
    CKPT_DIR = OUT_DIR  / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset      : m-bigearthnet (43 multi-label classes)")
    print(f"[Config] Architecture : DOFA ViT-Base 16 (224x224)")
    print(f"[Config] Bands        : 12-Band Sentinel-2 Subset ({DOFA_BAND_NAMES})")
    print(f"[Config] Wavelengths  : {DOFA_WAVELENGTHS}")
    print(f"[Config] Mode         : Multi-Label Group DRO (Distributionally Robust Optimization)")
    print(f"[Config] Base LR / LR : blr={args.blr} / lr={lr}")
    print(f"[Config] Eta Q        : eta_q={args.eta_q}")
    print(f"[Config] Seed         : {seed}")
    print(f"[Config] Grouping     : Macro Grouping (bigearthnet_master_mapping.json)")
    print(f"[Logging] Console log : {OUT_DIR / 'train.log'}")
    print(f"[Logging] JSON dir    : {OUT_DIR}")

    biome_map = load_biome_map(BIOME_JSON)

    geobench.GEO_BENCH_DIR       = str(DATA_DIR)
    geobench.task.GEO_BENCH_DIR  = DATA_DIR
    os.environ["GEO_BENCH_DIR"]  = str(DATA_DIR)

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

    norm_stats = (BIGEARTHNET_DOFA_MEAN, BIGEARTHNET_DOFA_STD)
    print(f"[Data] Exact Norm Mean: {[round(float(x), 2) for x in BIGEARTHNET_DOFA_MEAN]}")
    print(f"[Data] Exact Norm Std : {[round(float(x), 2) for x in BIGEARTHNET_DOFA_STD]}")

    train_ds = GeoBenchMultiLabelDataset(task, "train", augment=True, norm_stats=norm_stats)
    val_ds   = GeoBenchMultiLabelDataset(task, "valid", augment=False, norm_stats=norm_stats)
    test_ds  = GeoBenchMultiLabelDataset(task, "test",  augment=False, norm_stats=norm_stats)

    print("\n══════════════════════════════════════════════════════════════════════")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds,   biome_map)
    print_split_biome_distribution("test",  test_ds,  biome_map)
    print("══════════════════════════════════════════════════════════════════════\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = DOFAClassifier(
        num_classes     = NUM_CLASSES,
        model_type      = "dofa_base_patch16_224",
        checkpoint_path = str(WEIGHTS_PT),
        wavelengths     = DOFA_WAVELENGTHS,
        freeze_backbone = FREEZE_BACKBONE,
    )

    lp_model = MultiLabelGroupDROLinearProbe(
        model        = base_model,
        biome_map    = biome_map,
        lr           = lr,
        eta_q        = args.eta_q,
        weight_decay = WEIGHT_DECAY,
        num_classes  = NUM_CLASSES,
    )

    CKPT_MACRO_DIR = CKPT_DIR / "best_macro_f1"
    CKPT_WORST_DIR = CKPT_DIR / "best_worst_f1"
    CKPT_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_WORST_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_macro_cb = ModelCheckpoint(
        monitor                 = "val/macro_f1",
        mode                    = "max",
        dirpath                 = CKPT_MACRO_DIR,
        filename                = f"{mode_str}-macro-epoch={{epoch:02d}}-val_f1={{val/macro_f1:.4f}}",
        auto_insert_metric_name = False,
        save_top_k              = 3,
        save_last               = True,
        save_weights_only       = False,
        verbose                 = False,
    )
    ckpt_worst_cb = ModelCheckpoint(
        monitor                 = "val/worst_case_macro_f1",
        mode                    = "max",
        dirpath                 = CKPT_WORST_DIR,
        filename                = f"{mode_str}-worst-epoch={{epoch:02d}}-worst_f1={{val/worst_case_macro_f1:.4f}}",
        auto_insert_metric_name = False,
        save_top_k              = 3,
        save_last               = False,
        save_weights_only       = False,
        verbose                 = False,
    )

    early_stop_cb = EarlyStopping(
        monitor   = "val/macro_f1",
        mode      = "max",
        patience  = EARLY_STOP_PATIENCE,
        min_delta = EARLY_STOP_MIN_DELTA,
        verbose   = False,
    )
    global_f1_cb  = GlobalMacroF1Logger(num_classes=NUM_CLASSES)
    biome_cb      = BiomeMultiLabelLogger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = None,
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
        callbacks               = [global_f1_cb, ckpt_macro_cb, ckpt_worst_cb, early_stop_cb, biome_cb, progress_bar_cb],
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

    trainer.fit(lp_model, train_loader, val_loader, ckpt_path=resume_path)

    print(f"\n[Done] Best Macro F1 Checkpoint : {ckpt_macro_cb.best_model_path}")
    print(f"[Done] Best Macro F1 Score      : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f"[Done] Best Worst F1 Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f"[Done] Best Worst F1 Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
