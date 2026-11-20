#!/usr/bin/env python3
"""
SatMAE ViT (Base / Large) — BigEarthNet Multi-Label Classification (ERM Linear Probe)
====================================================================================
Key Features:
  • Task: Multi-label Classification (m-bigearthnet, 43 classes)
  • Architecture: SatMAE Group-Channels Vision Transformer (Frozen Backbone + Linear Probe Head)
  • Method: Empirical Risk Minimization (ERM) with BCEWithLogitsLoss
  • Spectral Groups: 10 Sentinel-2 bands grouped by resolution & spectral profile:
      - Group 0 (10m):  B02, B03, B04, B08  -> indices (0, 1, 2, 6)
      - Group 1 (20m):  B05, B06, B07, B08A -> indices (3, 4, 5, 7)
      - Group 2 (SWIR): B11, B12            -> indices (8, 9)
  • Normalization: Official SatMAE Sentinel-2 min-max reflectance normalization ([0.0, 1.0])
  • Probe Head: BatchNorm1d(embed_dim, affine=False) + Linear(embed_dim, 43)
  • Fairness Suite: Overall mAP & Macro F1 (optimal & 0.5 thresholding), Worst/Best Subgroup F1/mAP,
                   PER, Gini, NFR, ECE Disparity, Equalized Odds, Demographic Parity, per-class F1 table
  • Macro Grouping: Maps biome codes to macro biome groups via bigearthnet_biome_mapping.json
  • Slim Checkpoints: Strips frozen encoder weights (~150 KB per .ckpt file)
  • Live Test Logging: Evaluates test split at every validation epoch
  • Consolidated JSONs: Saves cumulative epoch history to val_results.json and test_results.json under SATMAE/runs/
"""

from fairrsfm.training.protocol import (MethodTrainingMixin, known_training_dataset, lookup_biome, training_loader, configure_training_protocol, dataset_sample_ids, balanced_group_loader, forward_with_sample_ids)
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

from sklearn.metrics import average_precision_score, f1_score
from fairrsfm.evaluation.metrics import compute_ece_multilabel


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
from fairrsfm.training.reporting import report_epoch, evaluating, macro_f1_present, write_json, finalize_selected, record_config
from fairrsfm.evaluation.serialization import json_safe
DATA_DIR = BASE_DIR / "Geobench"
if not DATA_DIR.exists() and os.environ.get("GEO_BENCH_DIR"):
    DATA_DIR = Path(os.environ["GEO_BENCH_DIR"])
if not DATA_DIR.exists():
    DATA_DIR = Path("/data1/aminur/dataset/geobench")
BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/bigearthnet_biome_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/bigearthnet_biome_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/bigearthnet_biome_mapping.json"

DEFAULT_CKPT_BASE  = BASE_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"
if not DEFAULT_CKPT_BASE.exists():
    DEFAULT_CKPT_BASE = BASE_DIR / "models/satmae/pretrain-vit-base-e199.pth"
if not DEFAULT_CKPT_BASE.exists():
    DEFAULT_CKPT_BASE = Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/satmae/pretrain-vit-base-e199.pth")

DEFAULT_CKPT_LARGE = BASE_DIR / "checkpoints/satmae/pretrain-vit-large-e199.pth"
if not DEFAULT_CKPT_LARGE.exists():
    DEFAULT_CKPT_LARGE = DEFAULT_CKPT_BASE

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

# 10 Sentinel-2 bands used in SatMAE (dropping B01 at idx 0 and B09 at idx 9 from GeoBench 12 bands)
SATMAE_BIGEARTHNET_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

# BigEarthNet 10-Band Statistics
BIGEARTHNET_10BAND_MEAN = [488.9927, 714.6071, 738.2603, 1114.4219, 1910.1897, 2191.4824, 2334.2920, 2392.9136, 1902.6918, 1261.0698]
BIGEARTHNET_10BAND_STD  = [510.7866, 551.7964, 691.7815,  700.4516,  976.7469, 1134.8865, 1238.0713, 1215.9766, 1117.0094,  894.7390]

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
    for sid in dataset_sample_ids(dataset):
        counts[lookup_biome(biome_map, sid)] += 1
    print(f"[{split_name}] {dict(counts)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET & OFFICIAL SATMAE NORMALIZATION
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


class GeoBenchBigEarthNetDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, img_size: int = 96, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = SATMAE_BIGEARTHNET_BAND_INDICES
        self.img_size     = img_size
        self.augment      = augment
        self.normalizer   = SatMAENormalize(BIGEARTHNET_10BAND_MEAN, BIGEARTHNET_10BAND_STD)

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
        x_data = x_data[:, :, self.band_indices]

        x = torch.from_numpy(x_data.astype(np.float32)).permute(2, 0, 1)
        x = self.spatial_transform(x)
        x = self.normalizer(x)

        y   = torch.tensor(sample.label, dtype=torch.float32)  # (43,) multi-hot
        sid = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. SATMAE BACKBONE & LINEAR PROBE ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class SatMAEMultiLabelLinearProbe(nn.Module):
    def __init__(
        self,
        model_type: str = "vit_large_patch16",
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 10,
        num_classes: int = 43,
        checkpoint_path: str = None,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_type = model_type
        self.freeze_backbone = freeze_backbone

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

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

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
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v

        missing, unexpected = self.backbone.load_state_dict(filtered_dict, strict=False)
        print(f"[SatMAE] Loaded {len(filtered_dict)}/{len(model_dict)} layers successfully.")

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

def find_optimal_thresholds(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = np.full(num_classes, 0.5)
    candidates = np.linspace(0.10, 0.90, 17)
    for c in range(num_classes):
        yt_c = y_true[:, c]
        if yt_c.sum() == 0:
            continue
        best_f1 = -1.0
        best_th = 0.5
        pr_c = probs[:, c]
        for th in candidates:
            yp_c = (pr_c >= th).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = f1_score(yt_c, yp_c, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        thresholds[c] = best_th
    return thresholds


def compute_subgroup_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        yt_c = y_true[:, c]
        if yt_c.sum() == 0:
            continue
        yp_c = y_pred[:, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s.append(f1_score(yt_c, yp_c, zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    f1s = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        yt_c = y_true[:, c]
        if yt_c.sum() == 0:
            continue
        yp_c = y_pred[:, c]
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
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
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
        for b2 in biomes[i + 1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class SatMAEBCEProbeLightningModule(MethodTrainingMixin, L.LightningModule):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.loss_fn = nn.BCEWithLogitsLoss()

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
            pl_module._val_y_prob = []

        logits = outputs["logits"].detach().float()
        labels = outputs["y"].detach().cpu().numpy()
        probs  = torch.sigmoid(logits).cpu().numpy()

        pl_module._val_y_true.append(labels)
        pl_module._val_y_prob.append(probs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(pl_module, "_val_y_true") or pl_module._val_y_true is None:
            return
        y_true = np.concatenate(pl_module._val_y_true, axis=0)
        probs  = np.concatenate(pl_module._val_y_prob, axis=0)
        preds05 = (probs >= 0.5).astype(int)

        macro_f1 = float(macro_f1_present(y_true, preds05, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, preds05, average="micro", zero_division=0))
        acc      = float(((preds05 == y_true).all(axis=1)).mean())
        map_val  = float(average_precision_score(y_true, probs, average="macro"))

        pl_module.log("val/macro_f1", macro_f1, prog_bar=True, logger=True)
        pl_module.log("val/micro_f1", micro_f1, prog_bar=False, logger=True)
        pl_module.log("val/accuracy", acc, prog_bar=False, logger=True)
        pl_module.log("val/map", map_val, prog_bar=False, logger=True)

        pl_module._val_y_true = None
        pl_module._val_y_prob = None


class BiomeMultiLabelLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader: DataLoader = None):
        super().__init__()
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.class_names = class_names
        self.output_dir  = output_dir
        self.test_loader = test_loader

        self.val_history  = []
        self.test_history = []
        self.opt_thresholds = np.full(num_classes, 0.5)

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
        return defaultdict(lambda: {"true": [], "probs": [], "sample_ids": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if not hasattr(pl_module, "_biome_data_val") or pl_module._biome_data_val is None:
            pl_module._biome_data_val = self._init_biome_data()

        logits     = outputs["logits"].detach().float()
        labels     = outputs["y"].detach().cpu().numpy()
        sample_ids = outputs["sample_ids"]
        probs      = torch.sigmoid(logits).cpu().numpy()

        for i, sid in enumerate(sample_ids):
            group = self.biome_map.get(str(sid), "Unknown")
            pl_module._biome_data_val[group]["true"].append(labels[i])
            pl_module._biome_data_val[group]["probs"].append(probs[i])
            pl_module._biome_data_val[group]["sample_ids"].append(str(sid))

    def _compute_and_log(self, biome_data, epoch, tb, section_title, prefix, pl_module=None):
        return report_epoch(self, biome_data, epoch, tb, section_title, prefix, pl_module, task="bigearthnet", ignore_index=-1)

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
        with evaluating(pl_module):
            pl_module.eval()
            test_biome_data = self._init_biome_data()
            device = pl_module.device

            with torch.no_grad():
                for batch in self.test_loader:
                    x, y, sample_ids = batch
                    x = x.to(device)
                    logits = forward_with_sample_ids(pl_module, x, sample_ids)
                    labels = y.numpy()
                    probs  = torch.sigmoid(logits.float()).cpu().numpy()

                    for i, sid in enumerate(sample_ids):
                        group = self.biome_map.get(str(sid), "Unknown")
                        test_biome_data[group]["true"].append(labels[i])
                        test_biome_data[group]["probs"].append(probs[i])
                        test_biome_data[group]["sample_ids"].append(str(sid))

            test_res = self._compute_and_log(test_biome_data, epoch, tb, "group_metrics_test", "test", pl_module=None)

            print(f"\n── [Epoch {epoch:02d} Results] ──────────────────────────────────────────")
            print(f"   Val  mAP : {trainer.callback_metrics.get('val/map', 0.0):.4f} | Macro F1@0.5: {trainer.callback_metrics.get('val/macro_f1', 0.0):.4f} | Worst: {trainer.callback_metrics.get('val/worst_case_macro_f1', 0.0):.4f} | PER: {trainer.callback_metrics.get('val/performance_equity_ratio', 0.0):.4f}")
            if test_res:
                print(f"   Test mAP : {test_res['global_map']:.4f} | Macro F1 (Opt): {test_res['overall_macro_f1_opt']:.4f} | Worst: {test_res['worst_case_macro_f1']:.4f} | PER: {test_res['performance_equity_ratio']:.4f}")
            print("────────────────────────────────────────────────────────────────────────\n")


    # ─────────────────────────────────────────────────────────────────────────────
    # 9. MAIN EXECUTION
    # ─────────────────────────────────────────────────────────────────────────────

def main():
    global BATCH_SIZE
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="SatMAE BigEarthNet ERM Linear Probe Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay (default: 1e-4)")
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"], help="SatMAE model size (default: large)")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Custom path to pretrained SatMAE weights (.pth)")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume a Lightning checkpoint")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    BATCH_SIZE = args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    if args.model_size == "large":
        model_type = "vit_large_patch16"
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else DEFAULT_CKPT_LARGE
    else:
        model_type = "vit_base_patch16"
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path else DEFAULT_CKPT_BASE

    mode_str = f"satmae_bigearthnet_lp_erm_{args.model_size}"
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    else:
        OUT_DIR = CURRENT_DIR / f"runs/bigearthnet_lp_erm/seed_{seed}"

    CKPT_DIR = OUT_DIR / "checkpoints"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print("══════════════════════════════════════════════════════════════════════")
    print(f" SatMAE BigEarthNet ERM Linear Probe (Seed {seed})")
    print(f" Model Type  : {model_type} (patch_size=8, img_size=96, 10 Sentinel bands)")
    print(f" Checkpoint  : {ckpt_path}")
    print(f" Output Dir  : {OUT_DIR}")
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

    train_ds = GeoBenchBigEarthNetDataset(task, "train", img_size=96, augment=True)
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchBigEarthNetDataset(task, "valid", img_size=96, augment=False)
    test_ds  = GeoBenchBigEarthNetDataset(task, "test",  img_size=96, augment=False)

    print("\nSplit biome distributions:")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds,   biome_map)
    print_split_biome_distribution("test",  test_ds,  biome_map)
    print()

    train_loader = training_loader(train_ds, method='erm', groups=protocol_groups, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = SatMAEMultiLabelLinearProbe(
        model_type=model_type,
        img_size=96,
        patch_size=8,
        in_chans=10,
        num_classes=NUM_CLASSES,
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
        freeze_backbone=FREEZE_BACKBONE
    )

    pl_model = SatMAEBCEProbeLightningModule(
        model=base_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_classes=NUM_CLASSES
    )
    configure_training_protocol(pl_model, 'erm', 'bigearthnet', protocol_groups, args, ignore_index=-1, lookup=lambda sid: lookup_biome(biome_map, sid))
    args.method_protocol = pl_model.method_protocol.config

    CKPT_MACRO_DIR = CKPT_DIR / "best_macro_f1"
    CKPT_WORST_DIR = CKPT_DIR / "best_worst_f1"
    CKPT_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_WORST_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_macro_cb = ModelCheckpoint(
        monitor                 = 'val/benchmark_macro_f1',
        mode                    = "max",
        dirpath                 = CKPT_MACRO_DIR,
        filename                = f"{mode_str}-macro-epoch={{epoch:02d}}-val_f1={{val/benchmark_macro_f1:.4f}}",
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
        monitor   = 'val/benchmark_macro_f1',
        mode      = "max",
        patience  = EARLY_STOP_PATIENCE,
        min_delta = EARLY_STOP_MIN_DELTA,
        verbose   = False,
    )
    global_f1_cb = GlobalMacroF1Logger(num_classes=NUM_CLASSES)
    biome_cb     = BiomeMultiLabelLogger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = test_loader,
    )
    progress_bar_cb = CleanProgressBar(refresh_rate=20)
    tb_logger = TensorBoardLogger(save_dir=OUT_DIR / "tb_logs", name="tb_logs", version=f"bigearthnet_seed_{seed}")

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

    record_config(OUT_DIR, __file__, args)
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume_from)
    finalize_selected(trainer, pl_model)

    print(f"\n[Training Complete]")
    print(f" Best Macro F1 Checkpoint      : {ckpt_macro_cb.best_model_path}")
    print(f" Best Macro F1 Score           : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f" Best Worst F1 Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f" Best Worst F1 Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
