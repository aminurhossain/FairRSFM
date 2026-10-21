#!/usr/bin/env python3
"""
SatMAE ViT (Base / Large) — EuroSAT Classification (ERM Linear Probe)
=====================================================================
Key Features:
  • Task: Single-label Multi-Class Classification (m-eurosat, 10 classes)
  • Architecture: SatMAE Group-Channels Vision Transformer (Frozen Backbone + Linear Probe Head)
  • Spectral Groups: 10 Sentinel-2 bands grouped by resolution & spectral profile:
      - Group 0 (10m):  B02, B03, B04, B08  -> indices (0, 1, 2, 6)
      - Group 1 (20m):  B05, B06, B07, B08A -> indices (3, 4, 5, 7)
      - Group 2 (SWIR): B11, B12            -> indices (8, 9)
  • Normalization: Official SatMAE Sentinel-2 min-max reflectance normalization ([0.0, 1.0])
  • Probe Head: BatchNorm1d(embed_dim, affine=False) + Linear(embed_dim, num_classes)
  • Fairness Suite: Worst/Best Macro F1, PER, Gini, ECE Disparity, Group Std, Equalized Odds, Demographic Parity
  • Macro Grouping: Maps 14 biome codes to 6 macro biome groups via master_mapping.json
  • Slim Checkpoints: Strips frozen encoder weights (~150 KB per .ckpt file)
  • Live Test Logging: Evaluates test split at every validation epoch
  • Consolidated JSONs: Saves val_results.json and test_results.json under SATMAE/runs/
"""

import sys
import os
import json
import logging
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
from functools import partial

CURRENT_DIR = Path(__file__).resolve().parent
# Project root and module resolution
ROOT_DIR = Path(__file__).resolve()
while ROOT_DIR.parent != ROOT_DIR:
    if (ROOT_DIR / "pyproject.toml").is_file() or (ROOT_DIR / "src/fairrsfm").is_dir():
        break
    ROOT_DIR = ROOT_DIR.parent

for p in [ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "src/fairrsfm", ROOT_DIR / "src/fairrsfm/models/upstream/SatMAE"]:
    if str(p) not in sys.path and p.is_dir():
        sys.path.insert(0, str(p))

# SatMAE upstream module setup
SATMAE_DIR = ROOT_DIR / "models/upstream/SatMAE"
if not SATMAE_DIR.exists():
    SATMAE_DIR = ROOT_DIR / "SatMAE-main"
if str(SATMAE_DIR) not in sys.path:
    sys.path.insert(0, str(SATMAE_DIR))

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

try:
    import models_vit_group_channels
except ImportError as e:
    raise ImportError(f"Could not import SatMAE group channels architecture from {SATMAE_DIR}: {e}")

from sklearn.metrics import average_precision_score, f1_score, accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEE LOGGER & PROGRESS BAR
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

BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/data_mappings/master_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/master_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/eurosat_biome_mapping.json"

DEFAULT_CKPT_BASE  = BASE_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"
if not DEFAULT_CKPT_BASE.exists():
    DEFAULT_CKPT_BASE = Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/satmae/pretrain-vit-base-e199.pth")
DEFAULT_CKPT_LARGE = BASE_DIR / "checkpoints/satmae/pretrain-vit-large-e199.pth"

BENCHMARK_NAME = "classification_v1.0"
BENCHMARK_DIR  = DATA_DIR / "classification_v1.0"
TASK_KEYWORD   = "eurosat"
NUM_CLASSES    = 10
CLASS_NAMES    = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
CLASS_WEIGHTS  = None

# ── EuroSAT 10 Bands matching SatMAE (dropping B01, B09, B10) ──
# EuroSAT band indices: [B02=1, B03=2, B04=3, B05=4, B06=5, B07=6, B08=7, B08A=8, B11=11, B12=12]
SATMAE_EUROSAT_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

# Official SatMAE Sentinel-2 dataset statistics for the 10 retained bands
# Order: B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12
SATMAE_10BAND_MEAN = [1184.3825, 1120.7712, 1136.2603, 1263.7395, 1645.4032, 1846.8704, 1762.5953, 1972.6242, 1732.1636, 1247.9187]
SATMAE_10BAND_STD  = [650.2843,  712.1251,  965.2312,  948.9820, 1108.0665, 1258.3639, 1233.1492, 1364.3869, 1310.3700, 1087.6021]

# Hyperparameters
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
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
# 4. DATASET & OFFICIAL SATMAE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class SatMAENormalize:
    """
    Official Sentinel-2 Normalization for SatMAE (matching SatMAE-main SentinelNormalize).
    Maps reflectance values within [mean - 2*std, mean + 2*std] to range [0.0, 1.0].
    """
    def __init__(self, mean: list, std: list):
        mean_arr = np.array(mean, dtype=np.float32)[:, None, None]
        std_arr  = np.array(std,  dtype=np.float32)[:, None, None]
        self.min_val = torch.from_numpy(mean_arr - 2.0 * std_arr)
        self.max_val = torch.from_numpy(mean_arr + 2.0 * std_arr)
        self.range   = self.max_val - self.min_val

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (C, H, W)
        norm = (tensor - self.min_val) / self.range
        return torch.clamp(norm, 0.0, 1.0)


class GeoBenchSatMAEDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, img_size: int = 96, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = SATMAE_EUROSAT_BAND_INDICES
        self.img_size     = img_size
        self.augment      = augment
        self.normalizer   = SatMAENormalize(SATMAE_10BAND_MEAN, SATMAE_10BAND_STD)

        if augment:
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size), antialias=True),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x_data = sample.pack_to_3d(band_names=self.band_names)[0]
        x_data = x_data[:, :, self.band_indices]  # (H, W, 10)
        
        x = torch.from_numpy(x_data.astype(np.float32)).permute(2, 0, 1)  # (10, H, W)
        x = self.spatial_transform(x)
        x = self.normalizer(x)

        y   = torch.tensor(int(sample.label), dtype=torch.long)
        sid = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. SATMAE BACKBONE & LINEAR PROBE ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class SatMAELinearProbe(nn.Module):
    """
    SatMAE Backbone with frozen representations and linear probe classification head.
    Probe head uses BatchNorm1d(embed_dim, affine=False) + Linear(embed_dim, num_classes).
    """
    def __init__(
        self,
        model_type: str = "vit_base_patch16",
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 10,
        num_classes: int = 10,
        checkpoint_path: str = None,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_type = model_type
        self.freeze_backbone = freeze_backbone

        # SatMAE channel groups: (10m bands), (20m red-edge/NIR), (20m SWIR)
        channel_groups = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))

        if hasattr(models_vit_group_channels, model_type):
            fn = getattr(models_vit_group_channels, model_type)
            self.backbone = fn(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                channel_groups=channel_groups,
                global_pool=True
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}' in models_vit_group_channels")

        self.embed_dim = self.backbone.embed_dim

        # Load weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)
        elif checkpoint_path:
            print(f"[SatMAE] Warning: Checkpoint '{checkpoint_path}' not found. Using random initialization.")

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Probe head: Standard MAE linear probing head with non-affine BatchNorm
        self.probe_head = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim, affine=False),
            nn.Linear(self.embed_dim, num_classes)
        )

    def load_pretrained_weights(self, ckpt_path: str):
        print(f"[SatMAE] Loading pretrained weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        model_dict = self.backbone.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"[SatMAE] Skipping shape mismatch for key: {k} (ckpt {v.shape} vs model {model_dict[k].shape})")

        missing, unexpected = self.backbone.load_state_dict(filtered_dict, strict=False)
        print(f"[SatMAE] Loaded {len(filtered_dict)}/{len(model_dict)} layers successfully.")
        if missing:
            print(f"[SatMAE] Missing keys count: {len(missing)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)

        logits = self.probe_head(feats)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRIC COMPUTATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_subgroup_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    present = list(np.unique(y_true))
    if not present:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    f1s = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        yt_c = (y_true == c).astype(int)
        yp_c = (y_pred == c).astype(int)
        if yt_c.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s[c] = float(f1_score(yt_c, yp_c, zero_division=0))
    return f1s


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
    confidences = probs.max(axis=-1)
    accuracies  = (preds == y_true)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
    return float(ece)


def compute_equalized_odds(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0]
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.array(biome_data[b]["true"])
        yp = np.array(biome_data[b]["pred"])
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            yt_c = (yt == c).astype(int)
            yp_c = (yp == c).astype(int)
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
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
    return worst


def compute_demographic_parity(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["pred"]) > 0]
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.array(biome_data[b]["pred"])
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = (yp == c).mean()
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


def compute_map_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_true_oh = np.eye(num_classes)[y_true]
    try:
        aps = []
        for c in range(num_classes):
            if y_true_oh[:, c].sum() > 0:
                ap = average_precision_score(y_true_oh[:, c], probs[:, c])
                aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class SatMAELightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float, num_classes: int):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

        if FREEZE_BACKBONE:
            frozen    = sum(p.numel() for p in self.model.backbone.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n[Model Architecture] Frozen Backbone : {frozen/1e6:.2f}M params")
            print(f"[Model Architecture] Trainable Probe  : {trainable/1e3:.2f}K params")

    def on_save_checkpoint(self, checkpoint):
        if FREEZE_BACKBONE:
            state_dict = checkpoint.get("state_dict", {})
            keys_to_remove = [k for k in list(state_dict.keys()) if k.startswith("model.backbone.")]
            for k in keys_to_remove:
                del state_dict[k]

    def on_load_checkpoint(self, checkpoint):
        if FREEZE_BACKBONE:
            state_dict = checkpoint.get("state_dict", {})
            for k, v in self.model.backbone.state_dict().items():
                state_dict[f"model.backbone.{k}"] = v

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def test_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test/loss", loss, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}}


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOGGERS & EVALUATION CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMacroF1Logger(Callback):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "_val_y_true") or pl_module._val_y_true is None:
            pl_module._val_y_true = []
            pl_module._val_y_pred = []
            pl_module._val_y_prob = []

        logits = outputs["logits"].detach().float()
        labels = outputs["y"].detach().cpu().numpy()
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        preds  = np.argmax(probs, axis=-1)

        pl_module._val_y_true.extend(labels)
        pl_module._val_y_pred.extend(preds)
        pl_module._val_y_prob.extend(probs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(pl_module, "_val_y_true") or pl_module._val_y_true is None:
            return
        y_true = np.array(pl_module._val_y_true)
        y_pred = np.array(pl_module._val_y_pred)
        probs  = np.array(pl_module._val_y_prob)

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        acc      = float(accuracy_score(y_true, y_pred))
        map_val  = compute_map_score(y_true, probs, self.num_classes)

        pl_module.log("val/macro_f1", macro_f1, prog_bar=True, logger=True)
        pl_module.log("val/micro_f1", micro_f1, prog_bar=False, logger=True)
        pl_module.log("val/accuracy", acc, prog_bar=False, logger=True)
        pl_module.log("val/map", map_val, prog_bar=False, logger=True)

        pl_module._val_y_true = None
        pl_module._val_y_pred = None
        pl_module._val_y_prob = None


class BiomeFairnessLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader: DataLoader = None):
        super().__init__()
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.class_names = class_names
        self.output_dir  = output_dir
        self.test_loader = test_loader

        self.val_history  = []
        self.test_history = []

        # Load existing history if resuming / appending
        val_file = self.output_dir / "val_results.json"
        if val_file.exists():
            try:
                with open(val_file, "r") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        self.val_history = content
                    elif isinstance(content, dict):
                        self.val_history = [content]
            except Exception:
                pass

        test_file = self.output_dir / "test_results.json"
        if test_file.exists():
            try:
                with open(test_file, "r") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        self.test_history = content
                    elif isinstance(content, dict):
                        self.test_history = [content]
            except Exception:
                pass

    def _init_biome_data(self):
        return defaultdict(lambda: {"true": [], "pred": [], "probs": [], "sample_ids": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "_biome_data_val") or pl_module._biome_data_val is None:
            pl_module._biome_data_val = self._init_biome_data()

        logits     = outputs["logits"].detach().float()
        labels     = outputs["y"].detach().cpu().numpy()
        sample_ids = outputs["sample_ids"]
        probs      = F.softmax(logits, dim=-1).cpu().numpy()
        preds      = np.argmax(probs, axis=-1)

        for i, sid in enumerate(sample_ids):
            group = self.biome_map.get(str(sid), "Unknown")
            if group == "Unknown":
                continue
            pl_module._biome_data_val[group]["true"].append(int(labels[i]))
            pl_module._biome_data_val[group]["pred"].append(int(preds[i]))
            pl_module._biome_data_val[group]["probs"].append(probs[i])
            pl_module._biome_data_val[group]["sample_ids"].append(str(sid))

    def _compute_and_log(self, biome_data, epoch, tb, section_title, prefix, pl_module=None):
        biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
        if not biomes:
            return {}

        all_true, all_pred, all_probs = [], [], []
        for b in biomes:
            all_true.extend(biome_data[b]["true"])
            all_pred.extend(biome_data[b]["pred"])
            all_probs.extend(biome_data[b]["probs"])
        all_true  = np.array(all_true)
        all_pred  = np.array(all_pred)
        all_probs = np.array(all_probs)

        global_macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
        global_micro_f1 = float(f1_score(all_true, all_pred, average="micro", zero_division=0))
        global_acc      = float(accuracy_score(all_true, all_pred))
        global_map      = compute_map_score(all_true, all_probs, self.num_classes)

        biome_macro_f1, biome_micro_f1, biome_accs, biome_maps, biome_eces, biome_counts = {}, {}, {}, {}, {}, {}
        per_group_metrics = {}
        per_class_f1_by_biome = {}

        for b in biomes:
            yt = np.array(biome_data[b]["true"])
            yp = np.array(biome_data[b]["pred"])
            pr = np.array(biome_data[b]["probs"])
            
            b_macro = compute_subgroup_macro_f1(yt, yp)
            b_micro = float(f1_score(yt, yp, average="micro", zero_division=0))
            b_acc   = float(accuracy_score(yt, yp))
            b_map   = compute_map_score(yt, pr, self.num_classes)
            b_ece   = compute_ece_metric(yt, pr, yp)
            b_count = len(yt)

            biome_macro_f1[b] = b_macro
            biome_micro_f1[b] = b_micro
            biome_accs[b]     = b_acc
            biome_maps[b]     = b_map
            biome_eces[b]     = b_ece
            biome_counts[b]   = b_count

            per_group_metrics[b] = {
                "macro_f1": b_macro,
                "micro_f1": b_micro,
                "accuracy": b_acc,
                "mAP": b_map,
                "ece": b_ece,
                "sample_count": b_count,
            }

            per_class_f1_by_biome[b] = compute_per_class_f1(yt, yp, self.num_classes)

        # Per-class F1 table & class disparities across biomes
        per_class_f1_table = {}
        class_disparities = {}
        for c in range(self.num_classes):
            c_name = self.class_names[c] if c < len(self.class_names) else f"Class_{c}"
            c_scores = {}
            c_vals = []
            for b in biomes:
                val = per_class_f1_by_biome[b][c]
                if not np.isnan(val):
                    c_scores[b] = float(val)
                    c_vals.append(float(val))
            per_class_f1_table[c_name] = c_scores
            class_disparities[c_name] = float(max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0

        class_disp_vals = list(class_disparities.values())
        mean_class_disp = float(np.mean(class_disp_vals)) if class_disp_vals else 0.0
        max_class_disp  = float(np.max(class_disp_vals)) if class_disp_vals else 0.0

        f1_vals   = [v for v in biome_macro_f1.values() if not np.isnan(v)]
        worst_f1  = min(f1_vals) if f1_vals else float("nan")
        best_f1   = max(f1_vals) if f1_vals else float("nan")
        mean_f1   = float(np.mean(f1_vals)) if f1_vals else float("nan")
        f1_std    = float(np.std(f1_vals)) if f1_vals else 0.0
        per       = (worst_f1 / best_f1) if (best_f1 > 0 and not np.isnan(worst_f1)) else 0.0
        nfr       = (best_f1 - worst_f1) / global_macro_f1 if global_macro_f1 > 0 else float("nan")
        gini      = compute_gini(f1_vals)

        ece_vals  = [v for v in biome_eces.values() if not np.isnan(v)]
        ece_disp  = (max(ece_vals) - min(ece_vals)) if ece_vals else float("nan")
        eq_odds   = compute_equalized_odds(biome_data, self.num_classes)
        dp_disp   = compute_demographic_parity(biome_data, self.num_classes)

        if prefix == "val" and pl_module is not None:
            pl_module.log("val/worst_case_macro_f1", worst_f1, prog_bar=True, logger=True)
            pl_module.log("val/performance_equity_ratio", per, prog_bar=False, logger=True)

        res_dict = {
            "epoch": epoch,
            "overall_macro_f1": global_macro_f1,
            "overall_micro_f1": global_micro_f1,
            "overall_accuracy": global_acc,
            "overall_mAP": global_map,
            "global_macro_f1": global_macro_f1,
            "global_micro_f1": global_micro_f1,
            "global_accuracy": global_acc,
            "global_map": global_map,
            "worst_case_macro_f1": worst_f1,
            "best_case_macro_f1": best_f1,
            "group_mean_macro_f1": mean_f1,
            "group_macro_f1_std": f1_std,
            "performance_equity_ratio": per,
            "normalized_fairness_range": nfr,
            "gini_coefficient": gini,
            "ece_disparity": ece_disp,
            "equalized_odds_disparity": eq_odds,
            "demographic_parity_disparity": dp_disp,
            "mean_class_disparity": mean_class_disp,
            "max_class_disparity": max_class_disp,
            "per_group_metrics": per_group_metrics,
            "per_biome_macro_f1": biome_macro_f1,
            "per_biome_micro_f1": biome_micro_f1,
            "per_biome_accuracy": biome_accs,
            "per_biome_map": biome_maps,
            "per_biome_ece": biome_eces,
            "per_biome_counts": biome_counts,
            "class_disparities": class_disparities,
            "per_class_f1_table": per_class_f1_table,
        }

        # Save cumulative epoch history
        if prefix == "val":
            self.val_history = [h for h in self.val_history if h.get("epoch") != epoch]
            self.val_history.append(res_dict)
            self.val_history.sort(key=lambda x: x.get("epoch", 0))
            with open(self.output_dir / "val_results.json", "w") as f:
                json.dump(self.val_history, f, indent=2)
        elif prefix == "test":
            self.test_history = [h for h in self.test_history if h.get("epoch") != epoch]
            self.test_history.append(res_dict)
            self.test_history.sort(key=lambda x: x.get("epoch", 0))
            with open(self.output_dir / "test_results.json", "w") as f:
                json.dump(self.test_history, f, indent=2)

        return res_dict

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(pl_module, "_biome_data_val") or pl_module._biome_data_val is None:
            return
        epoch = trainer.current_epoch
        tb    = trainer.logger.experiment if trainer.logger else None

        val_res = self._compute_and_log(pl_module._biome_data_val, epoch, tb, "group_metrics_val", "val", pl_module=pl_module)
        pl_module._biome_data_val = None

        # Live Test Split Evaluation
        if getattr(self, "test_loader", None) is not None:
            self._eval_test_split(trainer, pl_module, epoch, tb)

    def _eval_test_split(self, trainer, pl_module, epoch, tb):
        pl_module.eval()
        test_biome_data = self._init_biome_data()
        device = pl_module.device

        with torch.no_grad():
            for batch in self.test_loader:
                x, y, sample_ids = batch
                x = x.to(device)
                logits = pl_module(x)
                labels = y.numpy()
                probs  = F.softmax(logits.float(), dim=-1).cpu().numpy()
                preds  = np.argmax(probs, axis=-1)

                for i, sid in enumerate(sample_ids):
                    group = self.biome_map.get(str(sid), "Unknown")
                    if group == "Unknown":
                        continue
                    test_biome_data[group]["true"].append(int(labels[i]))
                    test_biome_data[group]["pred"].append(int(preds[i]))
                    test_biome_data[group]["probs"].append(probs[i])
                    test_biome_data[group]["sample_ids"].append(str(sid))

        test_res = self._compute_and_log(test_biome_data, epoch, tb, "group_metrics_test", "test", pl_module=None)
        
        print(f"\n── [Epoch {epoch:02d} Results] ──────────────────────────────────────────")
        print(f"   Val  Macro F1 : {trainer.callback_metrics.get('val/macro_f1', 0.0):.4f} | Worst: {trainer.callback_metrics.get('val/worst_case_macro_f1', 0.0):.4f} | PER: {trainer.callback_metrics.get('val/performance_equity_ratio', 0.0):.4f}")
        if test_res:
            print(f"   Test Macro F1 : {test_res['global_macro_f1']:.4f} | Worst: {test_res['worst_case_macro_f1']:.4f} | PER: {test_res['performance_equity_ratio']:.4f} | Acc: {test_res['global_accuracy']:.4f}")
        print("────────────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="SatMAE EuroSAT ERM Linear Probe Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay (default: 1e-4)")
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"], help="SatMAE model size (default: base)")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Custom path to pretrained SatMAE weights (.pth)")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    # Determine Checkpoint & Model Type
    if args.model_size == "large":
        model_type = "vit_large_patch16"
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else DEFAULT_CKPT_LARGE
    else:
        model_type = "vit_base_patch16"
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else DEFAULT_CKPT_BASE

    mode_str = f"satmae_eurosat_lp_erm_{args.model_size}"
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    else:
        OUT_DIR = CURRENT_DIR / f"runs/eurosat_lp_erm/seed_{seed}"

    CKPT_DIR = OUT_DIR / "checkpoints"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print("══════════════════════════════════════════════════════════════════════")
    print(f" SatMAE EuroSAT ERM Linear Probe (Seed {seed})")
    print(f" Model Type  : {model_type} (patch_size=8, img_size=96, 10 Sentinel bands)")
    print(f" Checkpoint  : {ckpt_path}")
    print(f" Output Dir  : {OUT_DIR}")
    print("══════════════════════════════════════════════════════════════════════\n")

    biome_map = load_biome_map(BIOME_JSON)

    # Load GeoBench EuroSAT Task
    geobench.GEO_BENCH_DIR      = str(DATA_DIR)
    geobench.task.GEO_BENCH_DIR = DATA_DIR

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

    train_ds = GeoBenchSatMAEDataset(task, "train", img_size=96, augment=True)
    val_ds   = GeoBenchSatMAEDataset(task, "valid", img_size=96, augment=False)
    test_ds  = GeoBenchSatMAEDataset(task, "test",  img_size=96, augment=False)

    print("\nSplit biome distributions:")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds,   biome_map)
    print_split_biome_distribution("test",  test_ds,  biome_map)
    print()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = SatMAELinearProbe(
        model_type=model_type,
        img_size=96,
        patch_size=8,
        in_chans=10,
        num_classes=NUM_CLASSES,
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
        freeze_backbone=FREEZE_BACKBONE
    )

    pl_model = SatMAELightningModule(
        model=base_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_classes=NUM_CLASSES
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
    global_f1_cb = GlobalMacroF1Logger(num_classes=NUM_CLASSES)
    biome_cb     = BiomeFairnessLogger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = None,
    )
    progress_bar_cb = CleanProgressBar(refresh_rate=20)
    tb_logger = TensorBoardLogger(save_dir=CURRENT_DIR / "runs", name="tb_logs", version=f"seed_{seed}")

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

    trainer.fit(pl_model, train_loader, val_loader)

    print(f"\n[Training Complete]")
    print(f" Best Macro F1 Checkpoint : {ckpt_macro_cb.best_model_path}")
    print(f" Best Macro F1 Score      : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f" Best Worst F1 Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f" Best Worst F1 Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
