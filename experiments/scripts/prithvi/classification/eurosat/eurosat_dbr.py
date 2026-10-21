#!/usr/bin/env python3
"""
Prithvi EO V2 300M — EuroSAT Classification (DBR Method)
========================================================
Key Features:
  • Task: Single-label Multi-Class Classification (m-eurosat, 10 classes)
  • Method: Distributionally Balanced Reweighting (DBR) with Macro Grouping (eurosat_biome_mapping.json)
  • Unknowns Purged: Dataset strictly filters out samples without known biome mappings.
  • Dynamic Weights: Updates group loss reweighting factors w_b dynamically based on validation Macro F1.
  • Augmentations: Random horizontal/vertical flips active on train split.
  • Slim Checkpoints: Strips frozen encoder weights (~2.5 MB per .ckpt file)
"""

import sys
import os
import json
import logging
import argparse
import ctypes
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 0. SYSTEM SETUP
# ─────────────────────────────────────────────────────────────────────────────
os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

try:
    ctypes.CDLL("/usr/lib64/openmpi/lib/libmpi.so.40")
except OSError:
    try:
        ctypes.CDLL("libmpi.so.40")
    except OSError:
        pass

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark = True

import geobench
import geobench.task
from terratorch.models import PrithviModelFactory
from terratorch.models.backbones import prithvi_vit
from terratorch.models.backbones.prithvi_vit import HLSBands

from sklearn.metrics import average_precision_score, f1_score, accuracy_score


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

ROOT_DIR = Path(__file__).resolve()
while ROOT_DIR.parent != ROOT_DIR:
    if (ROOT_DIR / "pyproject.toml").is_file():
        break
    ROOT_DIR = ROOT_DIR.parent
BASE_DIR = ROOT_DIR

DATA_DIR = BASE_DIR / "Geobench"
if not DATA_DIR.exists() and os.environ.get("GEO_BENCH_DIR"):
    DATA_DIR = Path(os.environ["GEO_BENCH_DIR"])
if not DATA_DIR.exists():
    DATA_DIR = Path("/data1/aminur/dataset/geobench")

BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/eurosat_biome_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/eurosat_biome_mapping.json"

WEIGHTS_PT = BASE_DIR / "checkpoints/Prithvi_EO_V2_300M_TL.pt"
if not WEIGHTS_PT.exists():
    WEIGHTS_PT = BASE_DIR / "models/Prithvi_EO_V2_300M_TL.pt"
if not WEIGHTS_PT.exists():
    WEIGHTS_PT = Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/Prithvi_EO_V2_300M_TL.pt")

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

BANDS = [
    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2,
]
PRITHVI_BAND_INDICES = [1, 2, 3, 7, 11, 12]

BATCH_SIZE   = 64
EPOCHS = 50
LR           = 1e-4
WEIGHT_DECAY = 0.1
SEED         = 42
NUM_WORKERS  = 16
FREEZE_BACKBONE = True

EARLY_STOP_PATIENCE = 1000
EARLY_STOP_MIN_DELTA = 1e-4

# ── DBR Config ──
USE_DBR   = True
DBR_ALPHA = 0.5
DBR_EMA   = 0.9


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
        
    print(f"  [{split_name.upper()} Split] Total samples before filter: {len(sample_names):,}")
    for b in sorted(counts.keys()):
        print(f"    • {b:<44} : {counts[b]:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEOBENCH DATASET WRAPPER (FILTERED & AUGMENTED)
# ─────────────────────────────────────────────────────────────────────────────

class GeoBenchClassifDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, biome_map: dict, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = PRITHVI_BAND_INDICES
        self.augment      = augment
        self.biome_map    = biome_map
        self.mean = torch.tensor(prithvi_vit.PRITHVI_V2_MEAN, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(prithvi_vit.PRITHVI_V2_STD,  dtype=torch.float32)[:, None, None]

        # Purge Unknowns: Build a strict list of valid indices
        self.valid_indices = []
        try:
            split_ids = self.dataset.active_partition.partition_dict[split]
            id_to_index = {str(self.dataset[i].sample_name): i for i in range(len(self.dataset))}
            for sid in split_ids:
                if self.biome_map.get(str(sid), "Unknown") != "Unknown":
                    if str(sid) in id_to_index:
                        self.valid_indices.append(id_to_index[str(sid)])
        except Exception:
            self.valid_indices = []
            for i in range(len(self.dataset)):
                sid = str(self.dataset[i].sample_name)
                if self.biome_map.get(sid, "Unknown") != "Unknown":
                    self.valid_indices.append(i)
        
        print(f"[Dataset] {split.upper()}: Filtered down to {len(self.valid_indices):,} samples with known biomes.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.dataset[real_idx]
        
        x_data = sample.pack_to_3d(band_names=self.band_names)[0]
        x_data = x_data[:, :, self.band_indices]
        x      = torch.tensor(x_data, dtype=torch.float32).permute(2, 0, 1)
        x      = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        
        if self.augment:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                
        x      = (x - self.mean) / self.std
        y      = torch.tensor(int(sample.label), dtype=torch.long)
        sid    = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. DBR REWEIGHTER MODULE
# ─────────────────────────────────────────────────────────────────────────────

class DynamicBiomeReweighter:
    def __init__(self, train_biomes: list, alpha: float = DBR_ALPHA, ema: float = DBR_EMA):
        self.alpha = alpha
        self.ema   = ema

        counts = defaultdict(int)
        for b in train_biomes:
            counts[b] += 1
        self.biomes = sorted(counts.keys())
        total = sum(counts.values())

        inv_freq = {b: total / (len(self.biomes) * counts[b]) for b in self.biomes}
        mean_if  = np.mean(list(inv_freq.values()))
        self.w_current = {b: float(inv_freq[b] / mean_if) for b in self.biomes}

        print(f"\n[DBR] Initialized for {len(self.biomes)} groups:")
        for b in self.biomes:
            print(f"  • {b:<44} : count={counts[b]:,}, w_init={self.w_current[b]:.4f}")

    def get_weights_tensor(self, sample_ids: list, biome_map: dict, device: torch.device) -> torch.Tensor:
        w_list = [self.w_current.get(biome_map.get(str(sid), "Unknown"), 1.0) for sid in sample_ids]
        return torch.tensor(w_list, dtype=torch.float32, device=device)

    def update_from_val(self, val_group_f1s: dict):
        valid = {b: val for b, val in val_group_f1s.items() if b in self.w_current and val is not None and not np.isnan(val)}
        if not valid:
            return

        mean_f1 = np.mean(list(valid.values()))
        w_raw   = {}
        for b in self.biomes:
            if b in valid:
                rel_diff = (mean_f1 - valid[b]) / (mean_f1 + 1e-6)
                w_target = self.w_current[b] * (1.0 + self.alpha * rel_diff)
                w_raw[b] = max(0.1, min(10.0, w_target))
        else:
                w_raw[b] = self.w_current[b]

        mean_w = np.mean(list(w_raw.values()))
        w_norm = {b: w_raw[b] / mean_w for b in self.biomes}

        print("\n[DBR] Updating weights based on Val Macro F1:")
        for b in self.biomes:
            f1_str = f"{valid[b]:.4f}" if b in valid else "N/A"
            w_old  = self.w_current[b]
            w_new  = self.ema * w_old + (1 - self.ema) * w_norm[b]
            self.w_current[b] = w_new
            print(f"  • {b:<44} : F1={f1_str:>6} | w_old={w_old:.4f} → w_new={w_new:.4f}")


def build_dbr_weights(train_ds: GeoBenchClassifDataset, alpha: float = DBR_ALPHA, ema: float = DBR_EMA) -> DynamicBiomeReweighter:
    print("[DBR] Scanning train set for macro biome labels...")
    train_biomes = []
    for real_idx in train_ds.valid_indices:
        sid = str(train_ds.dataset[real_idx].sample_name)
        group = train_ds.biome_map.get(sid, "Unknown")
        if group != "Unknown":
            train_biomes.append(group)
    return DynamicBiomeReweighter(train_biomes, alpha=alpha, ema=ema)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FAIRNESS & METRIC COMPUTATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_gini(values: list) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    array = np.sort(np.array(vals))
    index = np.arange(1, len(array) + 1)
    n = len(array)
    sum_val = np.sum(array)
    if sum_val == 0:
        return 0.0
    return float(np.sum((2 * index - n - 1) * array) / (n * sum_val))

def compute_ece_classif(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies  = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

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

def compute_eqodds(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown"]
    if len(biomes) < 2:
        return float("nan")

    J = {}
    for b in biomes:
        data = biome_data[b]
        yt = np.array(data["true"])
        yp = np.array(data["pred"])
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

def compute_dpm(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown"]
    if len(biomes) < 2:
        return float("nan")

    PPR = {}
    for b in biomes:
        data = biome_data[b]
        yp   = np.array(data["pred"])
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = (yp == c).mean()
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class ClassifLinearProbe(L.LightningModule):
    def __init__(self, model, lr: float, weight_decay: float, num_classes: int,
                 class_weights=None, reweighter=None, use_dbr: bool = False, biome_map: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "reweighter"])
        self.model       = model
        self.num_classes = num_classes
        self.reweighter  = reweighter
        self.use_dbr     = use_dbr and (reweighter is not None)
        self.biome_map   = biome_map or {}

        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn      = torch.nn.CrossEntropyLoss(weight=w)
        self.loss_fn_none = torch.nn.CrossEntropyLoss(weight=w, reduction="none")

        if FREEZE_BACKBONE:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()
            frozen    = sum(p.numel() for p in self.model.encoder.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n[Model] Frozen    : {frozen/1e6:.1f}M  (encoder)")
            print(f"[Model] Trainable : {trainable/1e6:.1f}M  (head)")
            print(f"[DBR]   Active    : {self.use_dbr}")

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
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)

    def training_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        if FREEZE_BACKBONE:
            self.model.encoder.eval()
        logits = self(x)

        if self.use_dbr:
            loss_per_sample = self.loss_fn_none(logits, y)
            sample_w = self.reweighter.get_weights_tensor(sample_ids, self.biome_map, device=x.device)
            loss = (loss_per_sample * sample_w).mean()
        else:
            loss = self.loss_fn(logits, y)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss   = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def test_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        loss   = self.loss_fn(logits, y)
        self.log("test/loss", loss, add_dataloader_idx=False, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}}


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOGGERS & METRIC CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMacroF1Logger(Callback):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._gf1_labels = []
        pl_module._gf1_preds  = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        preds  = logits.argmax(dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        pl_module._gf1_labels.append(labels)
        pl_module._gf1_preds.append(preds)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._gf1_labels:
            return
        y_true    = np.concatenate(pl_module._gf1_labels)
        y_pred    = np.concatenate(pl_module._gf1_preds)
        global_present = [c for c in range(self.num_classes) if (y_true == c).sum() > 0]
        if global_present:
            global_f1 = float(f1_score(y_true, y_pred, labels=global_present, average="macro", zero_division=0))
        else:
            global_f1 = 0.0
        global_micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        global_micro_acc = float((y_pred == y_true).mean())
        pl_module.log("val/macro_f1", global_f1, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/micro_f1", global_micro_f1, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/micro_acc", global_micro_acc, prog_bar=False, add_dataloader_idx=False)
        print(f"\n  [GlobalMacroF1] epoch {trainer.current_epoch} → val/macro_f1 = {global_f1:.4f} | val/micro_f1 = {global_micro_f1:.4f} | val/micro_acc = {global_micro_acc:.4f}")
        pl_module._gf1_labels = None
        pl_module._gf1_preds  = None


class BiomeMacroF1Logger(Callback):
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

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label, pl_module=None):
        biome_data = {b: d for b, d in biome_data.items() if b != "Unknown"}
        if not biome_data:
            print(f"\n  Biome metrics [{label}] — epoch {epoch}: No known biomes found!")
            return

        print(f"\n{'═'*82}")
        print(f"  Macro Group Metrics [{label}] — epoch {epoch}")
        print(f"{'═'*82}")
        print(f"\n  ── Per-group Classification Metrics ───────────────────────────────────────")
        print(f"  {'Macro Group':<44}  {'Macro F1':>8}  {'mAP':>7}  {'Acc':>7}  {'ECE':>7}")
        print(f"  {'─'*44}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")

        group_f1s   = {}
        group_micro_f1s = {}
        group_maps  = {}
        group_accs  = {}
        group_eces  = {}
        group_counts = {}

        for group in sorted(biome_data.keys()):
            d      = biome_data[group]
            n      = len(d["true"])
            y_true = np.array(d["true"])
            y_pred = np.array(d["pred"])
            probs  = np.array(d["probs"])

            present_classes = [c for c in range(self.num_classes) if (y_true == c).sum() > 0]
            if present_classes:
                f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0))
            else:
                f1 = 0.0

            micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            map_score = compute_map_score(y_true, probs, self.num_classes)
            acc = float((y_pred == y_true).mean())
            ece = compute_ece_classif(y_true, probs)

            group_f1s[group]    = f1
            group_micro_f1s[group] = micro_f1
            group_maps[group]   = map_score
            group_accs[group]   = acc
            group_eces[group]   = ece
            group_counts[group] = n

            print(f"  {group:<44}  {f1:>8.4f}  {map_score:>7.4f}  {acc:>7.4f}  {ece:>7.4f}")


        # Option B (class-level disparities)
        biomes  = sorted(biome_data.keys())
        n_biome = len(biomes)
        if n_biome >= 2:
            f1_matrix = np.full((self.num_classes, n_biome), float("nan"))
            for j, biome in enumerate(biomes):
                data = biome_data[biome]
                yt   = np.array(data["true"])
                yp   = np.array(data["pred"])
                if len(yt) == 0:
                    continue
                for c in range(self.num_classes):
                    yt_c = (yt == c).astype(int)
                    yp_c = (yp == c).astype(int)
                    if yt_c.sum() > 0:
                        f1_matrix[c, j] = float(f1_score(yt_c, yp_c, zero_division=0))
            
            disparity = np.full(self.num_classes, float("nan"))
            for c in range(self.num_classes):
                valid = f1_matrix[c, ~np.isnan(f1_matrix[c])]
                if len(valid) >= 2:
                    disparity[c] = float(valid.max() - valid.min())
                    
            col_w  = 8
            name_w = 36
            sep    = "─"
            biome_short = [b[:col_w] for b in biomes]

            header = (f"  {'Class':<{name_w}}  "
                      + "  ".join(f"{b:>{col_w}}" for b in biome_short)
                      + f"  {'Disparity':>{col_w}}")
            rule   = f"  {sep*name_w}  " + f"  {sep*col_w}" * n_biome + f"  {sep*col_w}"

            print(f"\n  ── Option B: Class-level biome F1 (max-min disparity) {'─'*10}")
            print(header)
            print(rule)
            
            for c in range(self.num_classes):
                name = self.class_names[c] if c < len(self.class_names) else f"cls{c}"
                row  = f"  {name:<{name_w}}  "
                row += "  ".join(
                    f"{f1_matrix[c,j]:>{col_w}.3f}" if not np.isnan(f1_matrix[c, j])
                    else f"{'nan':>{col_w}}"
                    for j in range(n_biome)
                )
                disp_str = (f"{disparity[c]:>{col_w}.3f}"
                            if not np.isnan(disparity[c]) else f"{'nan':>{col_w}}")
                print(row + f"  {disp_str}")
            print(rule)

        valid_f1s  = [v for v in group_f1s.values() if v is not None and not np.isnan(v)]
        valid_maps = [v for v in group_maps.values() if v is not None and not np.isnan(v)]
        valid_accs = [v for v in group_accs.values() if v is not None and not np.isnan(v)]
        valid_eces = [v for v in group_eces.values() if v is not None and not np.isnan(v)]

        all_yt = np.concatenate([d["true"] for d in biome_data.values()])
        all_yp = np.concatenate([d["pred"] for d in biome_data.values()])
        all_pr = np.concatenate([d["probs"] for d in biome_data.values()])

        overall_present = list(np.unique(all_yt))
        overall_f1  = float(f1_score(all_yt, all_yp, labels=overall_present, average="macro", zero_division=0)) if overall_present else 0.0
        overall_micro_f1 = float(f1_score(all_yt, all_yp, average="micro", zero_division=0))
        overall_map = float(compute_map_score(all_yt, all_pr, self.num_classes))
        overall_acc = float((all_yp == all_yt).mean())

        if len(valid_f1s) >= 2:
            worst_f1  = float(np.min(valid_f1s))
            best_f1   = float(np.max(valid_f1s))
            mean_f1   = float(np.mean(valid_f1s))
            std_f1    = float((np.std(valid_f1s, ddof=1) if len(valid_f1s) > 1 else 0.0))
            per_ratio = float(worst_f1 / best_f1) if best_f1 > 0 else None
            gini_coef = compute_gini(valid_f1s)
            nfr       = float((best_f1 - worst_f1) / np.mean(valid_f1s)) if np.mean(valid_f1s) > 0 else float("nan")
        else:
            worst_f1 = best_f1 = mean_f1 = std_f1 = per_ratio = gini_coef = nfr = None

        worst_map = float(np.min(valid_maps)) if len(valid_maps) >= 2 else None
        worst_acc = float(np.min(valid_accs)) if len(valid_accs) >= 2 else None

        if len(valid_eces) >= 2:
            ece_disparity = float(np.max(valid_eces) - np.min(valid_eces))
        else:
            ece_disparity = None

        eqodds_disp = compute_eqodds(biome_data, self.num_classes)
        dpm_disp    = compute_dpm(biome_data, self.num_classes)

        if pl_module is not None and label == "val":
            pl_module.log("val/accuracy", overall_acc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("val/mAP", overall_map, on_step=False, on_epoch=True, sync_dist=True)
            if worst_f1 is not None:
                pl_module.log("val/worst_case_macro_f1", worst_f1, on_step=False, on_epoch=True, sync_dist=True)
            if worst_acc is not None:
                pl_module.log("val/worst_case_accuracy", worst_acc, on_step=False, on_epoch=True, sync_dist=True)

            if pl_module.use_dbr and pl_module.reweighter is not None:
                pl_module.reweighter.update_from_val(group_f1s)

        print(f"\n  ── Representational Fairness & Overall Performance ────────────────────────")
        print(f"  {'Overall Macro F1':<44}  {overall_f1:.4f}")
        print(f"  {'Worst-case Macro F1':<44}  {worst_f1 if worst_f1 is not None else float('nan'):.4f}")
        print(f"  {'Best-case Macro F1':<44}  {best_f1 if best_f1 is not None else float('nan'):.4f}")
        print(f"  {'Overall mAP':<44}  {overall_map:.4f}")
        print(f"  {'Worst-case mAP':<44}  {worst_map if worst_map is not None else float('nan'):.4f}")
        print(f"  {'Overall Accuracy':<44}  {overall_acc:.4f}")
        print(f"  {'Worst-case Accuracy':<44}  {worst_acc if worst_acc is not None else float('nan'):.4f}")
        print(f"  {'Performance Equity Ratio (Worst/Best)':<44}  {per_ratio if per_ratio is not None else float('nan'):.4f}")
        print(f"  {'Gini Coefficient (Inequality)':<44}  {gini_coef if gini_coef is not None else float('nan'):.4f}")
        print(f"  {'NFR (Normalized Fairness Range)':<44}  {nfr if nfr is not None else float('nan'):.4f}")
        print(f"  {'ECE Disparity (Max-Min ECE)':<44}  {ece_disparity if ece_disparity is not None else float('nan'):.4f}")
        print(f"  {'Group Macro F1 Std':<44}  {std_f1 if std_f1 is not None else float('nan'):.4f}")
        print(f"  {'Equalized Odds Disparity':<44}  {eqodds_disp if not np.isnan(eqodds_disp) else float('nan'):.4f}")
        print(f"  {'Demographic Parity Disparity':<44}  {dpm_disp if not np.isnan(dpm_disp) else float('nan'):.4f}")
        print(f"{'═'*82}\n")

        payload = {
            "epoch": epoch,
            "overall_macro_f1": overall_f1,
            "overall_micro_f1": overall_micro_f1,
            "overall_micro_acc": overall_acc,
            "overall_mAP": overall_map,
            "overall_accuracy": overall_acc,
            "worst_case_macro_f1": worst_f1,
            "best_case_macro_f1": best_f1,
            "worst_case_mAP": worst_map,
            "worst_case_accuracy": worst_acc,
            "performance_equity_ratio": per_ratio,
            "gini_coefficient": gini_coef,
            "normalized_fairness_range": nfr,
            "ece_disparity": ece_disparity,
            "group_macro_f1_std": std_f1,
            "equalized_odds_disparity": eqodds_disp,
            "demographic_parity_disparity": dpm_disp,
            "per_group_metrics": {
                g: {"macro_f1": group_f1s[g], "micro_f1": group_micro_f1s[g], "mAP": group_maps[g], "accuracy": group_accs[g], "ece": group_eces[g], "sample_count": group_counts[g]}
                for g in group_f1s
            }
        }

        history = self.val_history if label == "val" else self.test_history
        history.append(payload)
        json_file = self.output_dir / f"{label}_results.json"
        with open(json_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  [JSON] Results updated in {json_file}")

    def _eval_test_split_and_log(self, trainer, pl_module, epoch, tb):
        if self.test_loader is None:
            return
        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device
        store  = defaultdict(lambda: {"true": [], "pred": [], "probs": []})

        with torch.no_grad():
            for batch in self.test_loader:
                x, y, sample_ids = batch
                x = x.to(device)
                logits = pl_module(x)
                probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                preds  = probs.argmax(axis=-1)
                labels = y.numpy()

                for i, sid in enumerate(sample_ids):
                    group = self.biome_map.get(sid, "Unknown")
                    if group == "Unknown":
                        continue
                    store[group]["true"].append(labels[i])
                    store[group]["pred"].append(preds[i])
                    store[group]["probs"].append(probs[i])

        self._compute_and_log(store, epoch, tb, "group_metrics_test", "test")
        if was_training:
            pl_module.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._biome_data_val = defaultdict(lambda: {"true": [], "pred": [], "probs": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits     = outputs["logits"]
        y          = outputs["y"]
        sample_ids = outputs["sample_ids"]

        probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)
        labels = y.cpu().numpy()

        for i, sid in enumerate(sample_ids):
            group = self.biome_map.get(sid, "Unknown")
            if group == "Unknown":
                continue
            pl_module._biome_data_val[group]["true"].append(labels[i])
            pl_module._biome_data_val[group]["pred"].append(preds[i])
            pl_module._biome_data_val[group]["probs"].append(probs[i])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._biome_data_val:
            return
        epoch = trainer.current_epoch
        tb    = trainer.logger.experiment if trainer.logger else None
        self._compute_and_log(pl_module._biome_data_val, epoch, tb, "group_metrics_val", "val", pl_module=pl_module)
        self._eval_test_split_and_log(trainer, pl_module, epoch, tb)
        pl_module._biome_data_val = None


# ─────────────────────────────────────────────────────────────────────────────
# 8. BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(weights_path: Path, num_classes: int):
    factory = PrithviModelFactory()
    return factory.build_model(
        task="classification",
        backbone="prithvi_eo_v2_300_tl",
        decoder="MLPDecoder",
        bands=BANDS,
        num_classes=num_classes,
        pretrained=True,
        backbone_ckpt_path=str(weights_path)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Prithvi EuroSAT Classification DBR Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dbr_tag  = f"dbr_alpha{DBR_ALPHA}_ema{DBR_EMA}" if USE_DBR else "erm"
    mode_str = f"eurosat_lp_{dbr_tag}"
    OUT_DIR  = BASE_DIR / f"runs/{mode_str}/seed_{seed}"
    CKPT_DIR = OUT_DIR  / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset    : m-eurosat (10 classes)")
    print(f"[Config] Mode       : Linear Probe (DBR)")
    print(f"[Config] Seed       : {seed}")
    print(f"[Config] DBR        : active={USE_DBR}, alpha={DBR_ALPHA}, ema={DBR_EMA}")
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

    print("\n══════════════════════════════════════════════════════════════════════")
    print_split_biome_distribution("train", task.get_dataset(split="train"), biome_map)
    print_split_biome_distribution("valid", task.get_dataset(split="valid"), biome_map)
    print_split_biome_distribution("test",  task.get_dataset(split="test"),  biome_map)
    print("══════════════════════════════════════════════════════════════════════\n")

    train_ds = GeoBenchClassifDataset(task, "train", biome_map, augment=True)
    val_ds   = GeoBenchClassifDataset(task, "valid", biome_map, augment=False)
    test_ds  = GeoBenchClassifDataset(task, "test",  biome_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = build_model(WEIGHTS_PT, NUM_CLASSES)

    reweighter = None
    if USE_DBR:
        reweighter = build_dbr_weights(
            train_ds  = train_ds,
            alpha     = DBR_ALPHA,
            ema       = DBR_EMA,
        )

    lp_model = ClassifLinearProbe(
        model         = base_model,
        lr            = LR,
        weight_decay  = WEIGHT_DECAY,
        num_classes   = NUM_CLASSES,
        class_weights = CLASS_WEIGHTS,
        reweighter    = reweighter,
        use_dbr       = USE_DBR,
        biome_map     = biome_map,
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
    biome_cb      = BiomeMacroF1Logger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = None,
    )
    progress_bar_cb = CleanProgressBar(refresh_rate=20)

    tb_logger = TensorBoardLogger(save_dir=BASE_DIR / "runs", name=mode_str, version=0)

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

    trainer.fit(lp_model, train_loader, val_loader, ckpt_path=args.resume_from)

    print(f"\n[Done] Best Macro F1 Checkpoint : {ckpt_macro_cb.best_model_path}")
    print(f"[Done] Best Macro F1 Score      : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f"[Done] Best Worst F1 Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f"[Done] Best Worst F1 Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()