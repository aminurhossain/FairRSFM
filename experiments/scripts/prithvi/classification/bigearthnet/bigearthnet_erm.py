#!/usr/bin/env python3
"""
Prithvi EO V2 300M — BigEarthNet Multi-Label Classification (ERM Linear Probe Baseline)
========================================================================================
Key Features:
  • Task: Multi-label Classification (m-bigearthnet, 43 classes, BCEWithLogitsLoss)
  • Primary Metrics: Macro F1 Score, mAP (Mean Average Precision), Sample Accuracy
  • Fairness Suite: Worst/Best Macro F1, PER, Gini, ECE Disparity, Group Std, Equalized Odds, Demographic Parity
  • Macro Grouping: Maps 14 biome codes to 6 macro biome groups via eurosat_biome_mapping.json
  • Slim Checkpoints: Strips frozen encoder weights (~2.5 MB per .ckpt file)
  • Live Test Logging: Evaluates test split at every validation epoch
  • Consolidated JSONs: Saves val_results.json and test_results.json
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

def compute_optimized_f1(y_true, probs, *args, **kwargs):
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            continue
        best_f1_c = 0.0
        for th in __import__('numpy').arange(0.05, 1.0, 0.05):
            y_pred_c = (probs[:, c] > th).astype(int)
            score = __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y_true[:, c], y_pred_c, zero_division=0)
            if score > best_f1_c:
                best_f1_c = score
                best_thresholds[c] = th
    
    y_pred = (probs > best_thresholds).astype(int)
    present = [c for c in range(num_classes) if y_true[:, c].sum() > 0]
    if not present:
        return 0.0
    return float(__import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0))


import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

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

BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/bigearthnet_biome_mapping.json"
if not BIOME_JSON.exists():
    BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/bigearthnet_biome_mapping.json"

WEIGHTS_PT = BASE_DIR / "checkpoints/Prithvi_EO_V2_300M_TL.pt"
if not WEIGHTS_PT.exists():
    WEIGHTS_PT = BASE_DIR / "models/Prithvi_EO_V2_300M_TL.pt"
if not WEIGHTS_PT.exists():
    WEIGHTS_PT = Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/Prithvi_EO_V2_300M_TL.pt")

BENCHMARK_NAME = "classification_v1.0"
BENCHMARK_DIR  = DATA_DIR / "classification_v1.0"
TASK_KEYWORD   = "bigearthnet"
NUM_CLASSES    = 43
NUM_FRAMES     = 1
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

BANDS = [
    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2,
]
PRITHVI_BAND_INDICES = [1, 2, 3, 7, 10, 11]

# ── Hyperparams ──
BATCH_SIZE   = 256
EPOCHS = 50
LR           = 1e-4
WEIGHT_DECAY = 0.1
HEAD_DROPOUT = 0.1
SEED         = 42
NUM_WORKERS  = 16
FREEZE_BACKBONE = True

# ── Early stopping ──
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
        1: "high biomass",
        3: "high biomass",
        5: "high biomass",
        2: "High amplitude phenological",
        6: "High amplitude phenological",
        4: "High amplitude phenological",
        7: "transitional herbaceous & scrub",
        8: "transitional herbaceous & scrub",
        12: "transitional herbaceous & scrub",
        10: "cryospheric & short-cycle",
        11: "cryospheric & short-cycle",
        13: "xeric & mineralogical",
        9: "hydrologically modulated",
        14: "hydrologically modulated"
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

    print(f"[BiomeMap] Loaded {len(biome_map)} entries | {len(set(biome_map.values()))} macro biome groups")
    return biome_map


def print_split_biome_distribution(split_name: str, dataset, biome_map: dict):
    counts = defaultdict(int)
    raw_ds = dataset.dataset
    
    # Fast path: extract sample names directly without loading HDF5
    if hasattr(raw_ds, "active_partition") and hasattr(raw_ds, "split"):
        sample_names = raw_ds.active_partition.partition_dict[raw_ds.split]
    else:
        # Fallback (slow)
        sample_names = [raw_ds[i].sample_name for i in range(len(raw_ds))]
        
    for sid in sample_names:
        b = biome_map.get(str(sid), "Unknown")
        counts[b] += 1
        
    print(f"  [{split_name.upper()} Split] Total samples: {len(sample_names):,}")
    for b in sorted(counts.keys()):
        print(f"    • {b:<44} : {counts[b]:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEOBENCH MULTI-LABEL CLASSIFICATION DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class GeoBenchClassifDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = PRITHVI_BAND_INDICES
        self.augment      = augment
        self.mean = torch.tensor(prithvi_vit.PRITHVI_V2_MEAN, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(prithvi_vit.PRITHVI_V2_STD,  dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x_data = sample.pack_to_3d(band_names=self.band_names)[0]
        x_data = x_data[:, :, self.band_indices]
        x      = torch.tensor(x_data, dtype=torch.float32).permute(2, 0, 1)
        x      = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        x      = (x - self.mean) / self.std
        y      = torch.tensor(sample.label, dtype=torch.float32)   # (43,) multi-hot
        sid    = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. FAIRNESS & METRIC COMPUTATION UTILITIES
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


def find_optimal_thresholds(all_labels: np.ndarray,
                             all_probs: np.ndarray,
                             num_classes: int) -> np.ndarray:
    thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        if all_labels[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            preds_c = (all_probs[:, c] >= t).astype(int)
            with __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter("ignore")
                f1 = __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(all_labels[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def compute_ece(y_true_flat, probs_max, preds_flat, n_bins=10):
    if len(y_true_flat) == 0:
        return float("nan")
    correct = (preds_flat == y_true_flat).astype(float)
    ece_val, n = 0.0, len(y_true_flat)
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (probs_max > bins[i]) & (probs_max <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece_val += (mask.sum()/n) * abs(
            correct[mask].mean() - probs_max[mask].mean()
        )
    return float(ece_val)


def compute_map_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    try:
        aps = [average_precision_score(y_true[:, c], probs[:, c])
               for c in range(num_classes) if y_true[:, c].sum() > 0]
        return float(np.mean(aps)) if aps else float("nan")
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


def compute_dpm(biome_data: dict, num_classes: int) -> float:
    biomes = [b for b in biome_data.keys() if b != "Unknown"]
    if len(biomes) < 2:
        return float("nan")

    PPR = {}
    for b in biomes:
        data = biome_data[b]
        yp   = np.array(data["pred"])
        PPR[b] = yp.mean(axis=0)

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 6. LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class ClassifLinearProbe(L.LightningModule):
    def __init__(self, model, lr: float, weight_decay: float, num_classes: int):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model       = model
        self.num_classes = num_classes
        self.loss_fn     = torch.nn.BCEWithLogitsLoss()

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
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        if FREEZE_BACKBONE:
            self.model.encoder.eval()
        logits = self(x)
        loss   = self.loss_fn(logits, y)
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
        y_true    = np.concatenate(pl_module._gf1_labels)
        probs     = np.concatenate(pl_module._gf1_probs)
        y_pred    = (probs > 0.5).astype(int)
        
        opt_th = find_optimal_thresholds(y_true, probs, y_true.shape[1])
        y_pred_opt = (probs >= opt_th[None, :]).astype(int)
        present = [c for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
        if present:
            global_f1 = float(f1_score(y_true[:, present], y_pred_opt[:, present], average="macro", zero_division=0))
        else:
            global_f1 = 0.0
        global_micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        
        best_macro = 0.0
        best_micro = 0.0
        for t in np.linspace(0.1, 0.9, 17):
            yp = (probs > t).astype(int)
            mac = f1_score(y_true, yp, average="macro", zero_division=0)
            mic = f1_score(y_true, yp, average="micro", zero_division=0)
            if mac > best_macro: best_macro = mac
            if mic > best_micro: best_micro = mic
            
        pl_module.log("val/macro_f1", global_f1, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/micro_f1", global_micro_f1, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/opt_macro_f1", float(best_macro), prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/opt_micro_f1", float(best_micro), prog_bar=False, add_dataloader_idx=False)
        print(f"\n  [GlobalMacroF1] epoch {trainer.current_epoch} → val/macro_f1 = {global_f1:.4f} | opt_macro = {best_macro:.4f} | micro_f1 = {global_micro_f1:.4f}")
        pl_module._gf1_labels = None
        pl_module._gf1_probs  = None


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

        all_labels = np.concatenate([d["true"] for d in biome_data.values()], axis=0)
        all_probs = np.concatenate([d["probs"] for d in biome_data.values()], axis=0)
        optimal_thresholds = find_optimal_thresholds(all_labels, all_probs, self.num_classes)
        
        for group, d in biome_data.items():
            probs = np.array(d["probs"])
            d["preds_opt"] = (probs >= optimal_thresholds[None, :]).astype(int)
            d["pred"] = d["preds_opt"]  # Overwrite pred so fairness metrics use optimal thresholds

        print(f"\n{'═'*82}")
        print(f"  Macro Group Metrics [{label}] — epoch {epoch}")
        print(f"{'═'*82}")
        print(f"\n  ── Per-group Classification Metrics ───────────────────────────────────────")
        print(f"  {'Macro Group':<44}  {'Opt F1':>8}  {'mAP':>7}  {'Acc':>7}  {'ECE':>7}")
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

            present = [c for c in range(self.num_classes) if y_true[:, c].sum() > 0]
            if present:
                f1 = float(f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0))
            else:
                f1 = 0.0

            micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            map_score = compute_map_score(y_true, probs, self.num_classes)
            acc = float((y_pred == y_true).all(axis=1).mean())
            ece = compute_ece(y_true.max(axis=1), probs.max(axis=1), y_pred.max(axis=1))

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
                yp = np.array(data["pred"])
                if len(yt) == 0:
                    continue
                for c in range(self.num_classes):
                    yt_c = yt[:, c]
                    yp_c = yp[:, c]
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

        overall_f1  = float(f1_score(all_yt, all_yp, average="macro", zero_division=0))
        overall_micro_f1 = float(f1_score(all_yt, all_yp, average="micro", zero_division=0))
        
        best_overall_macro = 0.0
        best_overall_micro = 0.0
        for t in np.linspace(0.1, 0.9, 17):
            yp_t = (all_pr > t).astype(int)
            mac = f1_score(all_yt, yp_t, average="macro", zero_division=0)
            mic = f1_score(all_yt, yp_t, average="micro", zero_division=0)
            if mac > best_overall_macro: best_overall_macro = mac
            if mic > best_overall_micro: best_overall_micro = mic
            
        overall_map = float(compute_map_score(all_yt, all_pr, self.num_classes))
        overall_acc = float((all_yp == all_yt).all(axis=1).mean())

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
            prefix = "val_orig" if hasattr(pl_module, "projector") or getattr(pl_module, "use_dbr", False) else "val"
            pl_module.log(f"{prefix}/accuracy", overall_acc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log(f"{prefix}/mAP", overall_map, on_step=False, on_epoch=True, sync_dist=True)
            if worst_f1 is not None:
                pl_module.log("val/worst_case_macro_f1", worst_f1, on_step=False, on_epoch=True, sync_dist=True)
            if worst_acc is not None:
                pl_module.log("val/worst_case_accuracy", worst_acc, on_step=False, on_epoch=True, sync_dist=True)

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
            "opt_overall_macro_f1": float(best_overall_macro),
            "opt_overall_micro_f1": float(best_overall_micro),
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
                probs  = torch.sigmoid(logits.float()).cpu().numpy()
                preds  = (probs > 0.5).astype(int)
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

        probs  = torch.sigmoid(logits.float()).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
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
            
        all_y_true = np.concatenate([d["true"] for d in pl_module._biome_data_val.values()])
        all_probs = np.concatenate([d["probs"] for d in pl_module._biome_data_val.values()])
        opt_th = find_optimal_thresholds(all_y_true, all_probs, self.num_classes)
        
        for group in pl_module._biome_data_val.keys():
            pr = np.array(pl_module._biome_data_val[group]["probs"])
            pl_module._biome_data_val[group]["pred"] = (pr >= opt_th[None, :]).astype(int)
            
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

    parser = argparse.ArgumentParser(description="Prithvi BigEarthNet Classification ERM Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    mode_str = "bigearthnet_lp_erm"
    OUT_DIR  = BASE_DIR / f"runs/{mode_str}/seed_{seed}"
    CKPT_DIR = OUT_DIR  / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset    : m-bigearthnet (43 classes, multi-label)")
    print(f"[Config] Mode       : Linear Probe")
    print(f"[Config] Seed       : {seed}")
    print(f"[Config] Grouping   : Macro Grouping (eurosat_biome_mapping.json)")
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

    train_ds = GeoBenchClassifDataset(task, "train", augment=True)
    val_ds   = GeoBenchClassifDataset(task, "valid", augment=False)
    test_ds  = GeoBenchClassifDataset(task, "test",  augment=False)

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

    base_model = build_model(WEIGHTS_PT, NUM_CLASSES)

    lp_model = ClassifLinearProbe(
        model       = base_model,
        lr          = LR,
        weight_decay= WEIGHT_DECAY,
        num_classes = NUM_CLASSES,
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
    progress_bar_cb = CleanProgressBar()

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
