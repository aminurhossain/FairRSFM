#!/usr/bin/env python3
"""
SatMAE ViT (Base / Large) — SA-Crop Semantic Segmentation (DBR Linear Probe)
============================================================================
Key Features:
  • Task: Semantic Segmentation (m-sa-crop, 10 classes, ignore_index=-1)
  • Architecture: SatMAE Group-Channels Vision Transformer (Frozen Backbone + Conv2d Probe Head)
  • Method: Distributionally Balanced Risk (DBR)
      - Tracks per-biome pixel error using exponential moving average (EMA, decay=0.9).
      - Reweights sample cross-entropy losses by (running_group_error + 1e-4)^alpha (alpha=0.5).
  • Spectral Groups: 10 Sentinel-2 bands grouped by resolution & spectral profile:
      - Group 0 (10m):  B02, B03, B04, B08  -> indices (0, 1, 2, 6)
      - Group 1 (20m):  B05, B06, B07, B08A -> indices (3, 4, 5, 7)
      - Group 2 (SWIR): B11, B12            -> indices (8, 9)
  • Normalization: Official SatMAE Sentinel-2 min-max reflectance normalization ([0.0, 1.0])
  • Loss Function: CrossEntropyLoss(ignore_index=-1)
  • Fairness Suite: Global mIoU & Macro F1, Subgroup mIoUs, PER, Gini, NFR, ECE Disparity,
                   Equalized Odds, Demographic Parity, per-class IoU table
  • Slim Checkpoints: Strips frozen encoder weights (~150 KB per .ckpt file)
  • Live Test Logging: Evaluates test split at every validation epoch
  • Consolidated JSONs: Saves cumulative epoch history to val_results.json and test_results.json under SATMAE/runs/
"""

import sys
import os
import json
import logging
import warnings
import argparse
from pathlib import Path
from collections import defaultdict

# Project root setup
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

from sklearn.metrics import f1_score


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
BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/data_mappings/sa_crop_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/sa_crop_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/sa_crop_mapping.json"

DEFAULT_CKPT_BASE  = BASE_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"
if not DEFAULT_CKPT_BASE.exists():
    DEFAULT_CKPT_BASE = BASE_DIR / "models/satmae/pretrain-vit-base-e199.pth"
if not DEFAULT_CKPT_BASE.exists():
    DEFAULT_CKPT_BASE = Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/satmae/pretrain-vit-base-e199.pth")

DEFAULT_CKPT_LARGE = BASE_DIR / "checkpoints/satmae/pretrain-vit-large-e199.pth"
if not DEFAULT_CKPT_LARGE.exists():
    DEFAULT_CKPT_LARGE = DEFAULT_CKPT_BASE

BENCHMARK_NAME = "segmentation_v1.0"
BENCHMARK_DIR  = DATA_DIR / "segmentation_v1.0"
TASK_KEYWORD   = "crop"
NUM_CLASSES    = 10
IGNORE_INDEX   = -1

CLASS_NAMES = [f"Crop_{i}" for i in range(NUM_CLASSES)]

SATMAE_SA_CROP_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

SA_CROP_10BAND_MEAN = [16.5267, 26.6364, 36.6966, 46.3887, 58.2815, 63.5758, 68.1836, 69.1426, 83.6268, 65.7677]
SA_CROP_10BAND_STD  = [ 9.3295, 12.6745, 19.4219, 19.4874, 19.9592, 21.5381, 23.0508, 22.3297, 28.1442, 27.2346]

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
NUM_WORKERS  = 8
FREEZE_BACKBONE = True

EARLY_STOP_PATIENCE  = 1000
EARLY_STOP_MIN_DELTA = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 3. BIOME MAP
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


class GeoBenchSACropDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, img_size: int = 96, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = SATMAE_SA_CROP_BAND_INDICES
        self.img_size     = img_size
        self.augment      = augment
        self.normalizer   = SatMAENormalize(SA_CROP_10BAND_MEAN, SA_CROP_10BAND_STD)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        array, _ = sample.pack_to_3d(band_names=self.band_names, resample=False, fill_value=None, resample_order=3)
        image = array.astype(np.float32)[:, :, self.band_indices]
        mask  = sample.label.data.astype(np.int64)

        x = torch.from_numpy(image).permute(2, 0, 1).float()
        y = torch.from_numpy(mask).long().unsqueeze(0)

        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0).float(), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).long().squeeze(0)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])

        x = self.normalizer(x)
        sid = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. SATMAE SEGMENTATION LINEAR PROBE ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class SatMAESegmentationLinearProbe(nn.Module):
    def __init__(
        self,
        model_type: str = "vit_large_patch16",
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
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.G = 3

        channel_groups = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))

        if hasattr(models_vit_group_channels, model_type):
            fn = getattr(models_vit_group_channels, model_type)
            self.backbone = fn(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                channel_groups=channel_groups,
                global_pool=False
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}' in models_vit_group_channels")

        self.embed_dim = self.backbone.embed_dim

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.probe_head = nn.Sequential(
            nn.BatchNorm2d(self.G * self.embed_dim, affine=False),
            nn.Conv2d(self.G * self.embed_dim, num_classes, kernel_size=1)
        )

    def load_pretrained_weights(self, ckpt_path: str):
        print(f"[SatMAE] Loading pretrained weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        model_dict = self.backbone.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v

        missing, unexpected = self.backbone.load_state_dict(filtered_dict, strict=False)
        print(f"[SatMAE] Loaded {len(filtered_dict)}/{len(model_dict)} layers successfully.")

    def forward_features_spatial(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        bb = self.backbone
        x_c_embed = []
        for i, group in enumerate(bb.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(bb.patch_embed[i](x_c))

        x_tokens = torch.stack(x_c_embed, dim=1)
        _, G_bb, L, D_bb = x_tokens.shape

        channel_embed = bb.channel_embed.unsqueeze(2)
        pos_embed = bb.pos_embed[:, 1:, :].unsqueeze(1)

        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)

        x_tokens = x_tokens + pos_channel
        x_tokens = x_tokens.view(b, -1, D_bb)

        cls_pos_channel = torch.cat((bb.pos_embed[:, :1, :], bb.channel_cls_embed), dim=-1)
        cls_tokens = cls_pos_channel + bb.cls_token.expand(b, -1, -1)
        x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
        x_tokens = bb.pos_drop(x_tokens)

        for blk in bb.blocks:
            x_tokens = blk(x_tokens)

        patches = x_tokens[:, 1:, :]
        patches = patches.view(b, self.G, L, self.embed_dim)
        patches = patches.permute(0, 2, 1, 3).reshape(b, L, self.G * self.embed_dim)
        spatial_features = patches.permute(0, 2, 1).view(b, self.G * self.embed_dim, self.grid_size, self.grid_size)
        return spatial_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.forward_features_spatial(x)
        else:
            feats = self.forward_features_spatial(x)

        logits = self.probe_head(feats)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXACT METRIC FUNCTIONS
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
# 7. LIGHTNING MODULE (DBR SEGMENTATION)
# ─────────────────────────────────────────────────────────────────────────────

class SatMAEDBRSegmentationLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, biome_map: dict, lr: float,
                 alpha: float = 0.5, ema_decay: float = 0.9,
                 weight_decay: float = WEIGHT_DECAY, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "biome_map"])
        self.model       = model
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.alpha       = alpha
        self.ema_decay   = ema_decay

        known_biomes = sorted(set(biome_map.values()) - {"Unknown"})
        self.biome_to_idx = {b: i for i, b in enumerate(known_biomes)}
        self.num_biomes = len(known_biomes)

        self.register_buffer("running_group_errors", torch.ones(self.num_biomes, dtype=torch.float32))

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
        x, y, sample_ids = batch
        logits = self(x)
        raw_losses = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX, reduction="none").mean(dim=(1, 2))

        batch_weights = torch.ones_like(raw_losses)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            for i, sid in enumerate(sample_ids):
                biome = self.biome_map.get(str(sid), "Unknown")
                b_idx = self.biome_to_idx.get(biome, None)
                if b_idx is not None:
                    mask = y[i] != IGNORE_INDEX
                    acc = (preds[i][mask] == y[i][mask]).float().mean().item() if mask.sum() > 0 else 0.5
                    err = 1.0 - acc
                    curr_err = self.running_group_errors[b_idx].item()
                    new_err = self.ema_decay * curr_err + (1.0 - self.ema_decay) * err
                    self.running_group_errors[b_idx] = new_err
                    batch_weights[i] = (new_err + 1e-4) ** self.alpha

        weights_sum = batch_weights.sum()
        if weights_sum > 0:
            batch_weights = batch_weights * (len(batch_weights) / weights_sum)

        loss = (raw_losses * batch_weights).mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
        self.log("val/loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def test_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
        self.log("test/loss", loss, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/miou"}}


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOGGERS & EVALUATION CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class GlobalSegmentationLogger(Callback):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "_val_y_true") or pl_module._val_y_true is None:
            pl_module._val_y_true = []
            pl_module._val_y_pred = []

        logits = outputs["logits"].detach().float()
        labels = outputs["y"].detach().cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        pl_module._val_y_true.append(labels.reshape(-1))
        pl_module._val_y_pred.append(preds.reshape(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(pl_module, "_val_y_true") or pl_module._val_y_true is None:
            return
        y_true = np.concatenate(pl_module._val_y_true, axis=0)
        y_pred = np.concatenate(pl_module._val_y_pred, axis=0)

        miou = compute_miou(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        f1   = compute_macro_f1_seg(y_true, y_pred, self.num_classes, ignore_index=IGNORE_INDEX)

        pl_module.log("val/miou", miou, prog_bar=True, logger=True)
        pl_module.log("val/macro_f1", f1, prog_bar=False, logger=True)

        pl_module._val_y_true = None
        pl_module._val_y_pred = None


class BiomeSegmentationLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader: DataLoader = None):
        super().__init__()
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.class_names = class_names
        self.output_dir  = output_dir
        self.test_loader = test_loader

        self.val_history  = []
        self.test_history = []

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
        return defaultdict(lambda: {"true": [], "pred": [], "probs_max": [], "sample_ids": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "_biome_data_val") or pl_module._biome_data_val is None:
            pl_module._biome_data_val = self._init_biome_data()

        logits     = outputs["logits"].detach().float()
        labels     = outputs["y"].detach().cpu().numpy()
        sample_ids = outputs["sample_ids"]

        probs     = torch.softmax(logits, dim=1)
        probs_max = probs.max(dim=1).values.cpu().numpy()
        preds     = logits.argmax(dim=1).cpu().numpy()

        for i, sid in enumerate(sample_ids):
            group = self.biome_map.get(str(sid), "Unknown")
            if group == "Unknown":
                continue
            pl_module._biome_data_val[group]["true"].append(labels[i].reshape(-1))
            pl_module._biome_data_val[group]["pred"].append(preds[i].reshape(-1))
            pl_module._biome_data_val[group]["probs_max"].append(probs_max[i].reshape(-1))
            pl_module._biome_data_val[group]["sample_ids"].append(str(sid))

    def _compute_and_log(self, biome_data, epoch, tb, section_title, prefix, pl_module=None):
        biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
        if not biomes:
            return {}

        all_true = np.concatenate([np.concatenate(biome_data[b]["true"]) for b in biomes], axis=0)
        all_pred = np.concatenate([np.concatenate(biome_data[b]["pred"]) for b in biomes], axis=0)
        all_pmax = np.concatenate([np.concatenate(biome_data[b]["probs_max"]) for b in biomes], axis=0)

        global_miou = compute_miou(all_true, all_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        global_f1   = compute_macro_f1_seg(all_true, all_pred, self.num_classes, ignore_index=IGNORE_INDEX)
        global_ece  = compute_ece_seg(all_true, all_pmax, all_pred, ignore_index=IGNORE_INDEX)

        biome_mious, biome_f1s, biome_eces, biome_counts = {}, {}, {}, {}
        per_group_metrics = {}
        per_class_iou_by_biome = {}

        for b in biomes:
            yt = np.concatenate(biome_data[b]["true"])
            yp = np.concatenate(biome_data[b]["pred"])
            pm = np.concatenate(biome_data[b]["probs_max"])

            b_miou  = compute_miou(yt, yp, self.num_classes, ignore_index=IGNORE_INDEX)
            b_f1    = compute_macro_f1_seg(yt, yp, self.num_classes, ignore_index=IGNORE_INDEX)
            b_ece   = compute_ece_seg(yt, pm, yp, ignore_index=IGNORE_INDEX)
            b_count = len(biome_data[b]["sample_ids"])

            biome_mious[b]  = b_miou
            biome_f1s[b]    = b_f1
            biome_eces[b]   = b_ece
            biome_counts[b] = b_count

            per_group_metrics[b] = {
                "miou": b_miou,
                "macro_f1": b_f1,
                "ece": b_ece,
                "sample_count": b_count,
            }

            per_class_iou_by_biome[b] = compute_per_class_iou(yt, yp, self.num_classes, ignore_index=IGNORE_INDEX)

        per_class_iou_table = {}
        class_disparities = {}
        for c in range(self.num_classes):
            c_name = self.class_names[c] if c < len(self.class_names) else f"Class_{c}"
            c_scores = {}
            c_vals = []
            for b in biomes:
                val = per_class_iou_by_biome[b][c]
                if not np.isnan(val):
                    c_scores[b] = float(val)
                    c_vals.append(float(val))
            per_class_iou_table[c_name] = c_scores
            class_disparities[c_name] = float(max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0

        class_disp_vals = list(class_disparities.values())
        mean_class_disp = float(np.mean(class_disp_vals)) if class_disp_vals else 0.0
        max_class_disp  = float(np.max(class_disp_vals)) if class_disp_vals else 0.0

        miou_vals = [v for v in biome_mious.values() if not np.isnan(v)]
        worst_miou = min(miou_vals) if miou_vals else float("nan")
        best_miou  = max(miou_vals) if miou_vals else float("nan")
        mean_miou  = float(np.mean(miou_vals)) if miou_vals else float("nan")
        miou_std   = float(np.std(miou_vals)) if miou_vals else 0.0
        per        = (worst_miou / best_miou) if (best_miou > 0 and not np.isnan(worst_miou)) else 0.0
        nfr        = (best_miou - worst_miou) / global_miou if global_miou > 0 else float("nan")
        gini       = compute_gini(miou_vals)

        ece_vals   = [v for v in biome_eces.values() if not np.isnan(v)]
        ece_disp   = (max(ece_vals) - min(ece_vals)) if ece_vals else float("nan")
        eq_odds    = compute_eqodds_seg(biome_data, self.num_classes, ignore_index=IGNORE_INDEX)
        dp_disp    = compute_demographic_parity_seg(biome_data, self.num_classes, ignore_index=IGNORE_INDEX)

        if prefix == "val" and pl_module is not None:
            pl_module.log("val/worst_case_miou", worst_miou, prog_bar=True, logger=True)
            pl_module.log("val/performance_equity_ratio", per, prog_bar=False, logger=True)

        res_dict = {
            "epoch": epoch,
            "overall_miou": global_miou,
            "overall_macro_f1": global_f1,
            "overall_ece": global_ece,
            "global_miou": global_miou,
            "global_macro_f1": global_f1,
            "worst_case_miou": worst_miou,
            "best_case_miou": best_miou,
            "group_mean_miou": mean_miou,
            "group_miou_std": miou_std,
            "performance_equity_ratio": per,
            "normalized_fairness_range": nfr,
            "gini_coefficient": gini,
            "ece_disparity": ece_disp,
            "equalized_odds_disparity": eq_odds,
            "demographic_parity_disparity": dp_disp,
            "mean_class_disparity": mean_class_disp,
            "max_class_disparity": max_class_disp,
            "per_group_metrics": per_group_metrics,
            "per_biome_miou": biome_mious,
            "per_biome_macro_f1": biome_f1s,
            "per_biome_ece": biome_eces,
            "per_biome_counts": biome_counts,
            "class_disparities": class_disparities,
            "per_class_iou_table": per_class_iou_table,
        }

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
                probs  = torch.softmax(logits, dim=1)
                probs_max = probs.max(dim=1).values.cpu().numpy()
                preds     = logits.argmax(dim=1).cpu().numpy()

                for i, sid in enumerate(sample_ids):
                    group = self.biome_map.get(str(sid), "Unknown")
                    if group == "Unknown":
                        continue
                    test_biome_data[group]["true"].append(labels[i].reshape(-1))
                    test_biome_data[group]["pred"].append(preds[i].reshape(-1))
                    test_biome_data[group]["probs_max"].append(probs_max[i].reshape(-1))
                    test_biome_data[group]["sample_ids"].append(str(sid))

        test_res = self._compute_and_log(test_biome_data, epoch, tb, "group_metrics_test", "test", pl_module=None)
        
        print(f"\n── [Epoch {epoch:02d} Results] ──────────────────────────────────────────")
        print(f"   Val  mIoU: {trainer.callback_metrics.get('val/miou', 0.0):.4f} | Worst: {trainer.callback_metrics.get('val/worst_case_miou', 0.0):.4f} | PER: {trainer.callback_metrics.get('val/performance_equity_ratio', 0.0):.4f}")
        if test_res:
            print(f"   Test mIoU: {test_res['global_miou']:.4f} | Worst: {test_res['worst_case_miou']:.4f} | PER: {test_res['performance_equity_ratio']:.4f} | Macro F1: {test_res['global_macro_f1']:.4f}")
        print("────────────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="SatMAE SA-Crop DBR Linear Probe Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay (default: 1e-4)")
    parser.add_argument("--alpha", type=float, default=0.5, help="DBR power exponent alpha (default: 0.5)")
    parser.add_argument("--ema-decay", type=float, default=0.9, help="DBR EMA decay (default: 0.9)")
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"], help="SatMAE model size (default: large)")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Custom path to pretrained SatMAE weights (.pth)")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    if args.model_size == "large":
        model_type = "vit_large_patch16"
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else DEFAULT_CKPT_LARGE
    else:
        model_type = "vit_base_patch16"
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else DEFAULT_CKPT_BASE

    mode_str = f"satmae_sa_crop_lp_dbr_{args.model_size}"
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    else:
        OUT_DIR = CURRENT_DIR / f"runs/sa_crop_lp_dbr/seed_{seed}"

    CKPT_DIR = OUT_DIR / "checkpoints"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print("══════════════════════════════════════════════════════════════════════")
    print(f" SatMAE SA-Crop DBR Linear Probe (Seed {seed})")
    print(f" Model Type  : {model_type} (patch_size=8, img_size=96, 10 Sentinel bands)")
    print(f" Checkpoint  : {ckpt_path}")
    print(f" Output Dir  : {OUT_DIR}")
    print(f" DBR Params  : alpha={args.alpha}, ema_decay={args.ema_decay}")
    print("══════════════════════════════════════════════════════════════════════\n")

    biome_map = load_biome_map(BIOME_JSON)

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

    train_ds = GeoBenchSACropDataset(task, "train", img_size=96, augment=True)
    val_ds   = GeoBenchSACropDataset(task, "valid", img_size=96, augment=False)
    test_ds  = GeoBenchSACropDataset(task, "test",  img_size=96, augment=False)

    print("\nSplit biome distributions:")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds,   biome_map)
    print_split_biome_distribution("test",  test_ds,  biome_map)
    print()

    base_model = SatMAESegmentationLinearProbe(
        model_type=model_type,
        img_size=96,
        patch_size=8,
        in_chans=10,
        num_classes=NUM_CLASSES,
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
        freeze_backbone=FREEZE_BACKBONE
    )

    pl_model = SatMAEDBRSegmentationLightningModule(
        model=base_model,
        biome_map=biome_map,
        lr=args.lr,
        alpha=args.alpha,
        ema_decay=args.ema_decay,
        weight_decay=args.weight_decay,
        num_classes=NUM_CLASSES
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    CKPT_MACRO_DIR = CKPT_DIR / "best_macro_iou"
    CKPT_WORST_DIR = CKPT_DIR / "best_worst_iou"
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
    global_logger_cb = GlobalSegmentationLogger(num_classes=NUM_CLASSES)
    biome_cb         = BiomeSegmentationLogger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = None,
    )
    progress_bar_cb = CleanProgressBar(refresh_rate=20)
    tb_logger = TensorBoardLogger(save_dir=CURRENT_DIR / "runs", name="tb_logs", version=f"sa_crop_dbr_seed_{seed}")

    trainer = L.Trainer(
        accelerator             = "gpu",
        devices                 = 1,
        strategy                = "auto",
        precision               = "bf16-mixed",
        max_epochs              = args.epochs,
        check_val_every_n_epoch = 1,
        log_every_n_steps       = 10,
        logger                  = tb_logger,
        callbacks               = [global_logger_cb, ckpt_macro_cb, ckpt_worst_cb, early_stop_cb, biome_cb, progress_bar_cb],
        enable_checkpointing    = True,
        enable_progress_bar     = True,
        deterministic           = "warn",
        benchmark               = False,
        num_sanity_val_steps    = 2,
    )

    trainer.fit(pl_model, train_loader, val_loader)

    print(f"\n[Training Complete]")
    print(f" Best mIoU Checkpoint     : {ckpt_macro_cb.best_model_path}")
    print(f" Best mIoU Score          : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f" Best Worst Checkpoint    : {ckpt_worst_cb.best_model_path}")
        print(f" Best Worst Score         : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
