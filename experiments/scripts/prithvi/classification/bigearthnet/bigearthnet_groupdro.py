#!/usr/bin/env python3
"""
Prithvi EO V2 300M — BigEarthNet Multi-Label Classification (Group DRO Method)
==============================================================================
Key Features:
  • Task: Multi-label Classification (m-bigearthnet, 43 classes, BCEWithLogitsLoss)
  • Architecture: Prithvi-EO-V2-300M (Terratorch PrithviModelFactory + MLPDecoder)
  • Method: Group Distributionally Robust Optimization (Group DRO - Sagawa et al., ICLR 2020)
  • Dynamic Group Weights: q_g <- q_g * exp(eta_q * loss_g) normalized over the probability simplex.
  • Biome Macro Groups: Evaluated across macro biome groups from bigearthnet_biome_mapping.json.
  • Unknowns Purged: Dataset strictly filters out samples without known biome mappings.
  • Subgroup Unique-Class Evaluation: Macro F1 computed over classes present in each biome.
  • Optimal Validation Threshold Sweep: Grid search t in [0.05, 0.95] for calibrated multi-label F1.
  • Metrics Suite: Full overall & per-biome Macro F1, mAP, Exact Acc, ECE, PER, Gini, NFR, EqOdds, DPM.
  • JSON Outputs: Saves val_results.json and test_results.json at every epoch.
  • Slim Checkpoints: Strips frozen encoder weights (~2.5 MB per .ckpt file).
"""

from fairrsfm.training.protocol import (MethodTrainingMixin, known_training_dataset, lookup_biome, training_loader, configure_training_protocol, dataset_sample_ids, balanced_group_loader, forward_with_sample_ids)
import sys
import os
import json
import logging
import argparse
import ctypes
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 0. SYSTEM & ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

try:
    import pyproj.datadir
    _pyproj_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_DATA"] = _pyproj_dir
    os.environ["PROJ_LIB"]  = _pyproj_dir
except Exception:
    pass

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
import torch.nn as nn
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
from terratorch.models.backbones.prithvi_vit import HLSBands, checkpoint_filter_fn_vit

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
from fairrsfm.training.reporting import report_epoch, evaluating, macro_f1_present, write_json, finalize_selected, record_config
from fairrsfm.evaluation.serialization import json_safe
sys.path.insert(0, str(ROOT_DIR / "src"))
from fairrsfm.evaluation.metrics import compute_ece_multilabel

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
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-4
ETA_Q        = 0.01
WEIGHT_DECAY = 0.1
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
    for sid in dataset_sample_ids(dataset):
        counts[lookup_biome(biome_map, sid)] += 1
    print(f"[{split_name}] {dict(counts)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEOBENCH MULTI-LABEL CLASSIFICATION DATASET WRAPPER (FILTERING UNKNOWNS)
# ─────────────────────────────────────────────────────────────────────────────

class GeoBenchClassifDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, biome_map: dict, augment: bool = False):
        super().__init__()
        raw_ds = task.get_dataset(split=split)
        self.band_names   = raw_ds[0].band_names
        self.band_indices = PRITHVI_BAND_INDICES
        self.augment      = augment
        self.mean = torch.tensor(prithvi_vit.PRITHVI_V2_MEAN, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(prithvi_vit.PRITHVI_V2_STD,  dtype=torch.float32)[:, None, None]

        # Strictly filter out unknown biomes
        if hasattr(raw_ds, "active_partition") and hasattr(raw_ds, "split") and raw_ds.split in raw_ds.active_partition.partition_dict:
            sample_names = raw_ds.active_partition.partition_dict[raw_ds.split]
        elif hasattr(raw_ds, "active_partition") and split in raw_ds.active_partition.partition_dict:
            sample_names = raw_ds.active_partition.partition_dict[split]
        elif hasattr(raw_ds, "_sample_name_list"):
            sample_names = raw_ds._sample_name_list
        else:
            sample_names = [raw_ds[i].sample_name for i in range(len(raw_ds))]

        self.valid_indices = [
            i for i, sid in enumerate(sample_names)
            if biome_map.get(str(sid), "Unknown") != "Unknown"
        ]

        self.dataset = raw_ds
        print(f"[Dataset] {split.upper()}: Filtered down to {len(self.valid_indices):,} samples with known biomes.")
        self.valid_indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample   = self.dataset[real_idx]
        x_data   = sample.pack_to_3d(band_names=self.band_names)[0]
        x_data   = x_data[:, :, self.band_indices]
        x        = torch.tensor(x_data, dtype=torch.float32).permute(2, 0, 1)
        x        = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])

        x = (x - self.mean) / self.std
        label = getattr(sample.label, "data", sample.label)
        y = torch.tensor(label, dtype=torch.float32)
        sid = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXACT METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_optimal_thresholds(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            continue
        best_th, best_f1 = 0.5, 0.0
        for th in np.arange(0.1, 0.95, 0.05):
            preds_c = (probs[:, c] >= th).astype(int)
            f1_c = f1_score(y_true[:, c], preds_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_th = th
        thresholds[c] = best_th
    return thresholds


def compute_macro_f1_multilabel(y_true: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    present_classes = [c for c in range(num_classes) if y_true[:, c].sum() > 0]
    if not present_classes:
        return float("nan")
    f1s = [f1_score(y_true[:, c], preds[:, c], zero_division=0) for c in present_classes]
    return float(np.mean(f1s))


def compute_per_class_f1(y_true: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    f1s = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            continue
        f1s[c] = float(f1_score(y_true[:, c], preds[:, c], zero_division=0))
    return f1s


def compute_map(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    present_classes = [c for c in range(num_classes) if y_true[:, c].sum() > 0]
    if not present_classes:
        return float("nan")
    aps = [average_precision_score(y_true[:, c], probs[:, c]) for c in present_classes]
    return float(np.mean(aps))




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


def compute_eqodds_multilabel(biome_data: dict, num_classes: int) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.vstack(biome_data[b]["true"])
        yp = np.vstack(biome_data[b]["pred_opt"])
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            tp = ((yp[:, c] == 1) & (yt[:, c] == 1)).sum()
            fn = ((yp[:, c] == 0) & (yt[:, c] == 1)).sum()
            fp = ((yp[:, c] == 1) & (yt[:, c] == 0)).sum()
            tn = ((yp[:, c] == 0) & (yt[:, c] == 0)).sum()
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


def compute_demographic_parity_multilabel(biome_data: dict, num_classes: int) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.vstack(biome_data[b]["pred_opt"])
        PPR[b] = yp.mean(axis=0)

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i+1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 6. LIGHTNING MODULE (GROUP DRO MULTI-LABEL)
# ─────────────────────────────────────────────────────────────────────────────

class PrithviBigEarthNetGroupDRO(MethodTrainingMixin, L.LightningModule):
    def __init__(self, model: nn.Module, biome_map: dict, lr: float,
                 eta_q: float = ETA_Q, weight_decay: float = WEIGHT_DECAY,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "biome_map"])
        self.model       = model
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.eta_q       = eta_q

        known_biomes = sorted(set(biome_map.values()) - {"Unknown"})
        self.biome_to_idx = {b: i for i, b in enumerate(known_biomes)}
        self.num_biomes = len(known_biomes)

        self._last_group_losses = None
        self._last_active_mask = None

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





    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}}


# ─────────────────────────────────────────────────────────────────────────────
# 7. METRIC CALLBACKS & CONSOLIDATED LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMetricsLogger(Callback):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._g_labels = []
        pl_module._g_probs  = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        probs  = torch.sigmoid(logits.float()).cpu().numpy()
        labels = y.cpu().numpy()
        pl_module._g_labels.append(labels)
        pl_module._g_probs.append(probs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._g_labels:
            return
        y_true = np.concatenate(pl_module._g_labels)
        probs  = np.concatenate(pl_module._g_probs)

        opt_th   = compute_optimal_thresholds(y_true, probs, self.num_classes)
        preds_05 = (probs >= 0.5).astype(int)
        preds_op = (probs >= opt_th).astype(int)

        macro_f1_05 = compute_macro_f1_multilabel(y_true, preds_05, self.num_classes)
        macro_f1_op = compute_macro_f1_multilabel(y_true, preds_op, self.num_classes)
        map_score   = compute_map(y_true, probs, self.num_classes)
        exact_acc   = float(accuracy_score(y_true, preds_op))

        pl_module.log("val/macro_f1", macro_f1_op, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/macro_f1_05", macro_f1_05, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/map", map_score, prog_bar=False, add_dataloader_idx=False)
        pl_module.log("val/exact_acc", exact_acc, prog_bar=False, add_dataloader_idx=False)

        print(f"\n  [GlobalMetrics] epoch {trainer.current_epoch} → val/macro_f1(opt) = {macro_f1_op:.4f} | val/macro_f1(0.5) = {macro_f1_05:.4f} | val/mAP = {map_score:.4f} | val/exact_acc = {exact_acc*100:.2f}%")
        pl_module._g_labels = None
        pl_module._g_probs  = None


class BiomeMetricsLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader=None):
        super().__init__()
        self.biome_map   = biome_map
        self.num_classes = num_classes
        self.class_names = class_names
        self.output_dir  = Path(output_dir)
        self.test_loader = test_loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opt_thresholds = np.full(num_classes, 0.5)

        val_json = self.output_dir / "val_results.json"
        if val_json.exists():
            try:
                with open(val_json, "r") as fp:
                    self.val_history = json.load(fp)
            except Exception:
                self.val_history = []
        else:
            self.val_history = []

        test_json = self.output_dir / "test_results.json"
        if test_json.exists():
            try:
                with open(test_json, "r") as fp:
                    self.test_history = json.load(fp)
            except Exception:
                self.test_history = []
        else:
            self.test_history = []

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label, pl_module=None):
        return report_epoch(self, biome_data, epoch, tb, tb_prefix, label, pl_module, task="bigearthnet", ignore_index=-1)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_data = defaultdict(lambda: {"true": [], "probs": [], "sample_ids": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        sids   = outputs["sample_ids"]

        probs = torch.sigmoid(logits.float()).cpu().numpy()
        labels_np = y.cpu().numpy()

        for sid, label_arr, prob_arr in zip(sids, labels_np, probs):
            biome = self.biome_map.get(str(sid), "Unknown")
            self._val_data[biome]["true"].append(label_arr)
            self._val_data[biome]["probs"].append(prob_arr)
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
        with evaluating(pl_module):
            pl_module.eval()
            test_data = defaultdict(lambda: {"true": [], "probs": [], "sample_ids": []})
            device = pl_module.device

            with torch.no_grad():
                for batch in self.test_loader:
                    x, y, sids = batch
                    x = x.to(device)
                    logits = forward_with_sample_ids(pl_module, x, sids)
                    probs = torch.sigmoid(logits.float()).cpu().numpy()
                    labels_np = y.numpy()

                    for sid, label_arr, prob_arr in zip(sids, labels_np, probs):
                        biome = self.biome_map.get(str(sid), "Unknown")
                        test_data[biome]["true"].append(label_arr)
                        test_data[biome]["probs"].append(prob_arr)
                        test_data[biome]["sample_ids"].append(sid)

            self._compute_and_log(test_data, epoch, tb, "test", "test", pl_module=None)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global BATCH_SIZE
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Prithvi BigEarthNet Group DRO Training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-4)")
    parser.add_argument("--eta-q", type=float, default=ETA_Q, help="Group DRO step size eta_q (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: 50)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch")
    parser.add_argument("--out-dir", type=str, default=None, help="Isolated output directory")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    BATCH_SIZE = args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    mode_str = "bigearthnet_groupdro"
    if args.output_dir:
        OUT_DIR = Path(args.out_dir) if args.out_dir else (Path(args.output_dir))
    else:
        OUT_DIR = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / f"runs/{mode_str}/seed_{seed}")
    CKPT_DIR = OUT_DIR / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset      : m-bigearthnet (43 classes)")
    print(f"[Config] Architecture : Prithvi-EO-V2-300M (Terratorch)")
    print(f"[Config] Mode         : Group DRO (Distributionally Robust Optimization)")
    print(f"[Config] LR           : {args.lr}")
    print(f"[Config] Eta Q        : {args.eta_q}")
    print(f"[Config] Seed         : {seed}")
    print(f"[Config] Grouping     : Macro Grouping (bigearthnet_biome_mapping.json)")
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

    train_ds = GeoBenchClassifDataset(task, "train", biome_map, augment=True)
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchClassifDataset(task, "valid", biome_map, augment=False)
    test_ds  = GeoBenchClassifDataset(task, "test",  biome_map, augment=False)

    print("\n══════════════════════════════════════════════════════════════════════")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds,   biome_map)
    print_split_biome_distribution("test",  test_ds,  biome_map)
    print("══════════════════════════════════════════════════════════════════════\n")

    train_loader = training_loader(train_ds, method='groupdro', groups=protocol_groups, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    factory = PrithviModelFactory()
    base_model = factory.build_model(
        task                 = "classification",
        backbone             = "prithvi_eo_v2_300_tl",
        decoder              = "MLPDecoder",
        bands                = BANDS,
        num_classes          = NUM_CLASSES,
        pretrained           = False,
    )

    if Path(WEIGHTS_PT).exists():
        print(f"[Model] Loading local weights from: {WEIGHTS_PT}")
        state_dict = torch.load(str(WEIGHTS_PT), map_location="cpu", weights_only=True)
        clean_sd = checkpoint_filter_fn_vit(state_dict, base_model.encoder, BANDS, BANDS)
        missing, unexpected = base_model.encoder.load_state_dict(clean_sd, strict=False)
        print(f"[Model] Successfully loaded Prithvi backbone weights! (missing: {len(missing)}, unexpected: {len(unexpected)})")

    gdro_model = PrithviBigEarthNetGroupDRO(
        model        = base_model,
        biome_map    = biome_map,
        lr           = args.lr,
        eta_q        = args.eta_q,
        weight_decay = WEIGHT_DECAY,
        num_classes  = NUM_CLASSES,
    )
    configure_training_protocol(gdro_model, 'groupdro', 'bigearthnet', protocol_groups, args, ignore_index=-1, lookup=lambda sid: lookup_biome(biome_map, sid))
    args.method_protocol = gdro_model.method_protocol.config

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
        monitor                 = "val/worst_case_f1",
        mode                    = "max",
        dirpath                 = CKPT_WORST_DIR,
        filename                = f"{mode_str}-worst-epoch={{epoch:02d}}-worst_f1={{val/worst_case_f1:.4f}}",
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
    global_cb       = GlobalMetricsLogger(num_classes=NUM_CLASSES)
    biome_cb        = BiomeMetricsLogger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = test_loader,
    )
    progress_bar_cb = CleanProgressBar()

    tb_logger = TensorBoardLogger(save_dir=OUT_DIR / "tb_logs", name="tb_logs", version=0)

    trainer = L.Trainer(
        accelerator             = "gpu",
        devices                 = 1,
        strategy                = "auto",
        precision               = "bf16-mixed",
        max_epochs              = args.epochs,
        check_val_every_n_epoch = 1,
        log_every_n_steps       = 10,
        logger                  = tb_logger,
        callbacks               = [global_cb, ckpt_macro_cb, ckpt_worst_cb, early_stop_cb, biome_cb, progress_bar_cb],
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

    record_config(OUT_DIR, __file__, args)
    trainer.fit(gdro_model, train_loader, val_loader, ckpt_path=resume_path)
    finalize_selected(trainer, gdro_model)

    print(f"\n[Done] Best Macro F1 Checkpoint : {ckpt_macro_cb.best_model_path}")
    print(f"[Done] Best Macro F1 Score      : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f"[Done] Best Worst F1 Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f"[Done] Best Worst F1 Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
