"""Prithvi 2.0 — Linear Probing / Fine-tuning: m-SA-crop-type (Segmentation)

Dynamic Biome Reweighting uses unweighted epoch group training losses,
EMA beta=0.9, alpha=0.5, epsilon=1e-6, and mean-one weights.
Initialize inverse group frequencies on the common known-biome training
subset. Backbones stay frozen; validation and test run every epoch and
do not update DBR. See docs/methods.md and fairrsfm.training.protocol.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. SYSTEM SETUP  ── PASTE FROM LP TRAIN FILE ──
# ─────────────────────────────────────────────────────────────────────────────
from fairrsfm.training.protocol import (MethodTrainingMixin, known_training_dataset, lookup_biome, training_loader, configure_training_protocol, dataset_sample_ids, balanced_group_loader, forward_with_sample_ids)
import os
import ctypes
try:
    import segmentation_models_pytorch.base.model as smp_model
    smp_model.SegmentationModel.__new__ = lambda cls, *args, **kwargs: object.__new__(cls)
except ImportError:
    pass

os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/lib64/openmpi/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)
try:
    ctypes.CDLL("/usr/lib64/openmpi/lib/libmpi.so.40")
except OSError:
    try:
        ctypes.CDLL("libmpi.so.40")
    except OSError:
        pass

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True


# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json
import math
import random
import sys
import warnings
from collections import defaultdict, Counter
from pathlib import Path

class TeeLogger:
    """
    Duplicates stdout and stderr output to a text log file automatically.
    Filters out tqdm progress bar carriage-return updates from train.log
    so the log file stays tiny and clean while terminal shows live progress.
    """
    def __init__(self, log_file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if "\r" in message or "%|" in message or "it/s]" in message:
            return
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

import numpy as np
import albumentations
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger


class CleanProgressBar(TQDMProgressBar):
    """
    Ultra-clean progress bar:
    Shows ONLY active training step loss in tqdm bar during iterations.
    Removes stale validation metrics, train/miou, and v_num so the progress bar stays ultra-short and never wraps lines.
    """
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        for k in list(items.keys()):
            if k not in ("loss", "train/loss"):
                items.pop(k, None)
        return items
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex

import geobench
import geobench.task
from terratorch.models import PrithviModelFactory
from terratorch.models.backbones import prithvi_vit
from terratorch.models.backbones.prithvi_vit import HLSBands


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# ── Experiment switches ──
FREEZE_BACKBONE   = True     # True = linear probe, False = fine-tune

# ── Dynamic Base Directory Resolution ──
ROOT_DIR = Path(__file__).resolve()
while ROOT_DIR.parent != ROOT_DIR:
    if (ROOT_DIR / "pyproject.toml").is_file():
        break
    ROOT_DIR = ROOT_DIR.parent
BASE_DIR = ROOT_DIR
from fairrsfm.training.reporting import report_epoch, evaluating, macro_f1_present, write_json, finalize_selected, record_config
from fairrsfm.evaluation.serialization import json_safe
# ── Dynamic Dataset Directory Resolution ──
DATA_DIR = None
for _candidate in [
    Path("/data1/aminur/dataset/geobench"),
    Path("/mnt/DATA1/geobench"),
    BASE_DIR / "data/geobench"
]:
    if _candidate.exists():
        DATA_DIR = _candidate
        break
if DATA_DIR is None:
    DATA_DIR = Path("/data1/aminur/dataset/geobench")
BIOME_JSON = BASE_DIR / "experiments/configs/data_mappings/sa_crop_biome_mapping.json"
WEIGHTS_PT = BASE_DIR / "checkpoints/Prithvi_EO_V2_300M_TL.pt"

# ── Dataset ──
BENCHMARK_NAME = "segmentation_v1.0"          # adjust if needed
BENCHMARK_DIR  = DATA_DIR / "segmentation_v1.0"  # adjust if needed
TASK_KEYWORD   = "crop"           # matched against dataset_name
NUM_CLASSES  = 10
NUM_FRAMES   = 1               # single temporal frame for this dataset
IGNORE_INDEX = -1

# ── Bands: Blue=0,Green=1,Red=2,NIR_N=6,SWIR1=9,SWIR2=10 from 13-band HDF5 ──
BAND_INDICES = [0, 1, 2, 7, 10, 11]
BANDS = [
    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2,
]

# ── Class names (update after confirming from dataset) ──
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# ── Class weights (None = uniform) ──
CLASS_WEIGHTS = None

# ── Hyperparams ──
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-4
WEIGHT_DECAY = 0.1
HEAD_DROPOUT = 0.1
SEED         = 42
NUM_WORKERS  = 16

# ── Early stopping ──
EARLY_STOP_PATIENCE  = 1000
EARLY_STOP_MIN_DELTA = 1e-4

# ── Biome ──
SAMPLE_WARN_THRESHOLD = 50

# ── DBR config ──
USE_DBR                     = True   # enable dynamic biome reweighting
DBR_ALPHA                   = 0.5    # feedback strength
DBR_EMA                     = 0.9    # metric EMA coefficient
DBR_UPDATE_EVERY            = 1      # update weights every epoch

# ── Neck indices for prithvi_eo_v2_300 ──
NECK_INDICES = [5, 11, 17, 23]   # use [7,15,23,31] for _600


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class GeoBenchSegDataset(Dataset):
    """
    GeoBench wrapper for m-SA-crop-type.
    Returns (image, mask, sample_id) — NOT a dict, so terratorch never sees
    sample_id and can't choke on it.

    image : (6, 224, 224) float32  — 6 selected bands, resized
    mask  : (224, 224)    int64    — per-pixel class IDs
    """
    def __init__(self, task, split: str, augment: bool = False):
        self.dataset   = task.get_dataset(split=split)
        self.augment   = augment

        self._spatial_aug = albumentations.Compose([
            albumentations.Flip(),
            albumentations.Resize(224, 224),
        ])
        self._spatial_val = albumentations.Compose([
            albumentations.Resize(224, 224),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample     = self.dataset[idx]
        band_names = self.dataset[0].band_names

        # (H, W, 13) — all bands
        image = sample.pack_to_3d(band_names=band_names)[0].astype(np.float32)
        # select 6 bands → (H, W, 6)
        image = image[:, :, BAND_INDICES]
        # mask (H, W)
        mask  = sample.label.data.astype(np.int64)

        # spatial transforms (albumentations on numpy HWC)
        tfm   = self._spatial_aug if self.augment else self._spatial_val
        out   = tfm(image=image, mask=mask)
        image = out["image"]   # (224, 224, 6) numpy
        mask  = out["mask"]    # (224, 224) numpy

        # to tensor
        image = torch.tensor(image).permute(2, 0, 1).float()  # (6, 224, 224)
        mask  = torch.tensor(mask).long()                      # (224, 224)

        sample_id = str(sample.sample_name)
        return image, mask, sample_id





# ─────────────────────────────────────────────────────────────────────────────
# 5. METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(y_true_flat: np.ndarray,
                 y_pred_flat: np.ndarray,
                 num_classes: int,
                 ignore_index: int = -1) -> float:
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


def compute_per_class_iou(y_true_flat: np.ndarray,
                           y_pred_flat: np.ndarray,
                           num_classes: int,
                           ignore_index: int = -1) -> np.ndarray:
    mask   = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    ious   = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        if (y_true == c).sum() == 0:
            continue
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else float("nan")
    return ious


def compute_ece_seg(y_true_flat, probs_max, preds_flat,
                    ignore_index=-1, n_bins=10):
    mask       = y_true_flat != ignore_index
    y_true     = y_true_flat[mask]
    probs_max  = probs_max[mask]
    preds_flat = preds_flat[mask]
    if len(y_true) == 0:
        return float("nan")
    correct = (preds_flat == y_true).astype(float)
    ece, n  = 0.0, len(y_true)
    for lo, hi in zip(np.linspace(0,1,n_bins+1)[:-1],
                      np.linspace(0,1,n_bins+1)[1:]):
        m = (probs_max >= lo) & (probs_max < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum()/n) * abs(correct[m].mean() - probs_max[m].mean())
    return float(ece)


def compute_segmentation_metrics_from_biome_data(biome_data: dict,
                                                 num_classes: int,
                                                 ignore_index: int):
    def _valid_pixels(yt_flat, yp_flat, ignore):
        mask = yt_flat != ignore
        return yt_flat[mask], yp_flat[mask]

    def compute_miou_local(yt_flat, yp_flat, num_classes, ignore):
        yt, yp = _valid_pixels(yt_flat, yp_flat, ignore)
        present = [c for c in range(num_classes) if (yt == c).sum() > 0]
        if not present:
            return float("nan")
        ious = []
        for c in present:
            tp = ((yp == c) & (yt == c)).sum()
            fp = ((yp == c) & (yt != c)).sum()
            fn = ((yp != c) & (yt == c)).sum()
            denom = tp + fp + fn
            ious.append(float(tp / denom) if denom > 0 else float("nan"))
        valid = [v for v in ious if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    def compute_mean_dice_local(yt_flat, yp_flat, num_classes, ignore):
        yt, yp = _valid_pixels(yt_flat, yp_flat, ignore)
        present = [c for c in range(num_classes) if (yt == c).sum() > 0]
        if not present:
            return float("nan"), np.full(num_classes, float("nan"))
        per_class = np.full(num_classes, float("nan"))
        for c in present:
            tp = ((yp == c) & (yt == c)).sum()
            fp = ((yp == c) & (yt != c)).sum()
            fn = ((yp != c) & (yt == c)).sum()
            denom = 2 * tp + fp + fn
            per_class[c] = float(2 * tp / denom) if denom > 0 else float("nan")
        valid = per_class[~np.isnan(per_class)]
        return (float(np.mean(valid)) if len(valid) > 0 else float("nan")), per_class

    def compute_pixel_accuracy_local(yt_flat, yp_flat, ignore):
        yt, yp = _valid_pixels(yt_flat, yp_flat, ignore)
        if len(yt) == 0:
            return float("nan")
        return float((yp == yt).sum() / len(yt))

    rows = []
    for biome, data in biome_data.items():
        if not data["true"]:
            continue
        yt = np.concatenate(data["true"])
        yp = np.concatenate(data["pred"])
        rows.append((biome, yt, yp))

    if not rows:
        return None

    all_true = np.concatenate([yt for _, yt, _ in rows])
    all_pred = np.concatenate([yp for _, _, yp in rows])

    global_miou = compute_miou_local(all_true, all_pred, num_classes, ignore_index)
    global_f1, _ = compute_mean_dice_local(all_true, all_pred, num_classes, ignore_index)
    pixel_acc = compute_pixel_accuracy_local(all_true, all_pred, ignore_index)

    biome_mious = {}
    biome_f1s = {}
    for biome, yt, yp in rows:
        biome_mious[biome] = compute_miou_local(yt, yp, num_classes, ignore_index)
        biome_f1s[biome], _ = compute_mean_dice_local(yt, yp, num_classes, ignore_index)

    valid_mious = [v for v in biome_mious.values() if not np.isnan(v)]
    valid_f1s = [v for v in biome_f1s.values() if not np.isnan(v)]
    if len(valid_mious) >= 2:
        worst_miou = float(np.min(valid_mious))
        mean_miou = float(np.mean(valid_mious))
        std_miou = float((np.std(valid_mious, ddof=1) if len(valid_mious) > 1 else 0.0))
        nfr_miou = (max(valid_mious) - min(valid_mious)) / mean_miou if mean_miou > 0 else float("nan")
        dev = sum(abs(v - global_miou) for v in valid_mious)
        es_miou = float(global_miou / (dev + 1))
        es_std_miou = float(global_miou / (std_miou + 1))
    else:
        worst_miou = std_miou = nfr_miou = es_miou = es_std_miou = float("nan")

    if len(valid_f1s) >= 2:
        worst_f1 = float(np.min(valid_f1s))
        mean_f1 = float(np.mean(valid_f1s))
        std_f1 = float((np.std(valid_f1s, ddof=1) if len(valid_f1s) > 1 else 0.0))
        nfr_f1 = (max(valid_f1s) - min(valid_f1s)) / mean_f1 if mean_f1 > 0 else float("nan")
    else:
        worst_f1 = mean_f1 = std_f1 = nfr_f1 = float("nan")

    def compute_eqodds_local(biome_data, num_classes, ignore):
        biomes = sorted(biome_data.keys())
        if len(biomes) < 2:
            return float("nan")
        J = {}
        for b, data in biome_data.items():
            yt = np.concatenate(data["true"])
            yp = np.concatenate(data["pred"])
            m = yt != ignore
            yt_valid = yt[m]
            yp_valid = yp[m]
            j_b = np.full(num_classes, float("nan"))
            for c in range(num_classes):
                yt_c = (yt_valid == c).astype(int)
                yp_c = (yp_valid == c).astype(int)
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
            for b2 in biomes[i + 1:]:
                diffs = [abs(J[b1][c] - J[b2][c])
                         for c in range(num_classes)
                         if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
                if diffs:
                    worst = max(worst, float(np.mean(diffs)))
        return worst

    eqodds = compute_eqodds_local(biome_data, num_classes, ignore_index)

    return {
        "global_miou": global_miou,
        "global_f1": global_f1,
        "pixel_acc": pixel_acc,
        "es_miou": es_miou,
        "es_std_miou": es_std_miou,
        "worst_biome_miou": worst_miou,
        "worst_biome_f1": worst_f1,
        "mean_biome_f1": mean_f1,
        "std_biome_f1": std_f1,
        "nfr_miou": nfr_miou,
        "nfr_f1": nfr_f1,
        "biome_std_miou": std_miou,
        "eqodds": eqodds,
        "biome_wise_miou_table": biome_mious,
        "biome_wise_f1_table": biome_f1s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. DBR MODULE
# ─────────────────────────────────────────────────────────────────────────────

from fairrsfm.methods.dbr import DynamicBiomeReweighter




# ─────────────────────────────────────────────────────────────────────────────
# 7. GLOBAL mIoU CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMIoULogger(Callback):
    """
    Accumulates pixel predictions from original val loader (dataloader_idx=0)
    and logs val_orig/miou at epoch end — drives checkpointing.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._gmiou_true = []
        pl_module._gmiou_pred = []

    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]                       # (B, C, H, W)
        masks  = outputs["masks"]                        # (B, H, W)
        preds  = logits.argmax(dim=1).cpu().numpy()
        masks  = masks.cpu().numpy()
        pl_module._gmiou_true.append(masks.reshape(-1))
        pl_module._gmiou_pred.append(preds.reshape(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module._gmiou_true:
            return
        y_true      = np.concatenate(pl_module._gmiou_true)
        y_pred      = np.concatenate(pl_module._gmiou_pred)
        global_miou = compute_miou(y_true, y_pred,
                                   self.num_classes, self.ignore_index)
        pl_module.log("val_orig/miou", global_miou,
                      prog_bar=False, add_dataloader_idx=False)
        print(f"\n  [GlobalMIoU] epoch {trainer.current_epoch} "
              f"→ val_orig/miou = {global_miou:.4f}")
        pl_module._gmiou_true = None
        pl_module._gmiou_pred = None


# ─────────────────────────────────────────────────────────────────────────────
# 7. BIOME FAIRNESS CALLBACK  (Option C = A + B)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gini(values: list) -> float:
    """Compute the Gini coefficient of inequality for a list of non-negative values."""
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


def print_split_biome_distribution(split_name: str, dataset, biome_map: dict):
    counts = defaultdict(int)
    for sid in dataset_sample_ids(dataset):
        counts[lookup_biome(biome_map, sid)] += 1
    print(f"[{split_name}] {dict(counts)}")



def compute_macro_f1_seg(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, num_classes: int, ignore_index: int = -1) -> float:
    mask   = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    if not present: return float("nan")
    f1s = []
    for c in present:
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in f1s if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")

def compute_eqodds_seg(biome_data: dict, num_classes: int, ignore_index: int = -1) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown"])
    if len(biomes) < 2: return float("nan")
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
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes) if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs: worst = max(worst, float(np.mean(diffs)))
    return worst

def compute_dpm_seg(biome_data: dict, num_classes: int, ignore_index: int = -1) -> float:
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown"])
    if len(biomes) < 2: return float("nan")
    PPR = {}
    for b in biomes:
        yt = np.concatenate(biome_data[b]["true"])
        yp = np.concatenate(biome_data[b]["pred"])
        mask = yt != ignore_index
        yp_v = yp[mask]
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = (yp_v == c).mean() if len(yp_v) > 0 else float("nan")
        PPR[b] = ppr_b
    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diffs = [abs(PPR[b1][c] - PPR[b2][c]) for c in range(num_classes) if not (np.isnan(PPR[b1][c]) or np.isnan(PPR[b2][c]))]
            if diffs: worst = max(worst, float(np.mean(diffs)))
    return worst

class BiomeMacroF1Logger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, output_dir: Path, ignore_index: int = -1):
        super().__init__()
        self.biome_map = biome_map
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.ignore_index = ignore_index

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label):
        biome_data = {b: d for b, d in biome_data.items() if b != "Unknown"}
        if not biome_data: return
        print(f"\n  ── Per-biome Macro F1 ─────────────────────────────────────")
        print(f"  {'Biome':<44}  {'Macro F1':>8}")
        print(f"  {'─'*44}  {'─'*8}")
        f1s = []
        for biome in sorted(biome_data.keys()):
            y_true = np.concatenate(biome_data[biome]["true"])
            y_pred = np.concatenate(biome_data[biome]["pred"])
            f1 = compute_macro_f1_seg(y_true, y_pred, self.num_classes, self.ignore_index)
            print(f"  {biome:<44}  {f1:>8.4f}")
            f1s.append(f1)
            bk = biome.replace(" ","_").replace("&","and").replace(",","")
            if tb: tb.add_scalar(f"{tb_prefix}/f1/{bk}", f1, epoch)
        valid = [v for v in f1s if not np.isnan(v)]
        if len(valid) >= 2:
            print(f"\n  Fairness Metrics [F1]:")
            print(f"  Worst-case F1 : {np.min(valid):.4f}")
            print(f"  Std F1        : {(np.std(valid, ddof=1) if len(valid) > 1 else 0.0):.4f}")
            print(f"  PER F1        : {np.min(valid)/np.max(valid):.4f}")
        print(f"  {'─'*54}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not hasattr(pl_module, "_val_biome_data"): pl_module._val_biome_data = {}
        for b, d in outputs.get("biome_data", {}).items():
            if b not in pl_module._val_biome_data: pl_module._val_biome_data[b] = {"true": [], "pred": [], "probs": []}
            pl_module._val_biome_data[b]["true"].extend(d["true"])
            pl_module._val_biome_data[b]["pred"].extend(d["pred"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "_val_biome_data"):
            tb = pl_module.logger.experiment if pl_module.logger else None
            self._compute_and_log(pl_module._val_biome_data, trainer.current_epoch, tb, "val", "val")
            pl_module._val_biome_data.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not hasattr(pl_module, "_test_biome_data"): pl_module._test_biome_data = {}
        for b, d in outputs.get("biome_data", {}).items():
            if b not in pl_module._test_biome_data: pl_module._test_biome_data[b] = {"true": [], "pred": [], "probs": []}
            pl_module._test_biome_data[b]["true"].extend(d["true"])
            pl_module._test_biome_data[b]["pred"].extend(d["pred"])

    def on_test_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "_test_biome_data"):
            tb = pl_module.logger.experiment if pl_module.logger else None
            self._compute_and_log(pl_module._test_biome_data, trainer.current_epoch, tb, "test", "test")
            pl_module._test_biome_data.clear()

class BiomeMIoULogger(Callback):

    def __init__(self, biome_map: dict, num_classes: int,
                 class_names: list, output_dir: Path, ignore_index: int = -1,
                 test_loader=None):
        super().__init__()
        self.biome_map             = biome_map
        self.num_classes           = num_classes
        self.class_names           = class_names
        self.output_dir            = Path(output_dir)
        self.ignore_index          = ignore_index
        self.test_loader           = test_loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_history           = []
        self.test_history          = []

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label, pl_module=None):
        # Exclude Unknown from all metrics completely
        return report_epoch(self, biome_data, epoch, tb, tb_prefix, label, pl_module, task="sa_crop", ignore_index=getattr(self, "ignore_index", IGNORE_INDEX))

    def _eval_test_split_and_log(self, trainer, pl_module, epoch, tb):
        with evaluating(pl_module):
            was_training = pl_module.training
            pl_module.eval()
            device = pl_module.device
            store  = defaultdict(lambda: {"true": [], "pred": [], "probs": []})

            with torch.no_grad():
                for batch in self.test_loader:
                    x, masks, sample_ids = batch
                    x     = x.to(device)
                    masks = masks.to(device)
                    logits = forward_with_sample_ids(pl_module, x, sample_ids)
                    probs  = torch.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).cpu().numpy()
                    preds  = probs.argmax(axis=-1)
                    masks  = masks.cpu().numpy()

                    for i, sid in enumerate(sample_ids):
                        biome = (self.biome_map.get(sid)
                                 or self.biome_map.get(Path(sid).stem, "Unknown"))
                        store[biome]["true"].append(masks[i].reshape(-1))
                        store[biome]["pred"].append(preds[i].reshape(-1))
                        store[biome]["probs"].append(probs[i].reshape(-1, self.num_classes))

            self._compute_and_log(store, epoch, tb, "biome_metrics_test", "test", pl_module=pl_module)
            if was_training:
                pl_module.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._biome_orig = defaultdict(lambda: {"true":[],"pred":[],"probs":[]})

    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or outputs is None:
            return
        store      = pl_module._biome_orig
        logits     = outputs["logits"]           # (B, C, H, W)
        masks      = outputs["masks"]            # (B, H, W)
        sample_ids = outputs["sample_ids"]       # list of str

        probs = torch.softmax(logits.float(), dim=1).permute(0,2,3,1).cpu().numpy()
        preds = probs.argmax(axis=-1)            # (B, H, W)
        masks = masks.cpu().numpy()

        for i, sid in enumerate(sample_ids):
            biome = (self.biome_map.get(sid)
                     or self.biome_map.get(Path(sid).stem, "Unknown"))
            store[biome]["true"].append(masks[i].reshape(-1))
            store[biome]["pred"].append(preds[i].reshape(-1))
            store[biome]["probs"].append(probs[i].reshape(-1, self.num_classes))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        tb    = trainer.logger.experiment if hasattr(trainer.logger, "experiment") else None

        # 1. Compute & save validation metrics to val_results.json
        if pl_module._biome_orig:
            self._compute_and_log(pl_module._biome_orig, epoch, tb,
                                  "biome_metrics", "val", pl_module=pl_module)
        pl_module._biome_orig = None

        # 2. Evaluate test split & save test metrics to test_results.json
        if self.test_loader is not None:
            self._eval_test_split_and_log(trainer, pl_module, epoch, tb)


# ─────────────────────────────────────────────────────────────────────────────
# 8. HELPERS  ── PASTE _normalize_id() + load_biome_map() FROM LP TRAIN FILE ──
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_id(sid: str) -> str:
    return Path(sid).stem

def load_biome_map(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"[BiomeMap] WARNING: {json_path} not found.")
        return {}
    with open(json_path) as f:
        records = json.load(f)
    biome_map: dict = {}
    for r in records:
        if r["biome_name"] == "Unknown":
            continue
        raw = str(r["geobench_id"])
        biome_map[raw]            = r["biome_name"]
        biome_map[Path(raw).stem] = r["biome_name"]
    n = len({str(r["geobench_id"]) for r in records})
    print(f"[BiomeMap] Loaded {n} entries | "
          f"{len(set(biome_map.values()))} biomes")
    return biome_map

def print_biome_distribution(dataset, name: str, biome_map: dict):
    dist  = defaultdict(int)
    # handle Subset
    if hasattr(dataset, "indices"):
        base = dataset.dataset.dataset
        ids  = [str(base[i].sample_name) for i in dataset.indices]
    else:
        ids  = [str(dataset.dataset[i].sample_name) for i in range(len(dataset))]
    for sid in ids:
        biome = biome_map.get(sid) or biome_map.get(Path(sid).stem)
        if biome is None or biome == "Unknown":
            continue
        dist[biome] += 1
    total = len(ids)
    print(f"\n  Biome distribution — {name} ({total} samples)")
    for biome, count in sorted(dist.items()):
        print(f"    {biome:<45} {count:>6}  ({100*count/total:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 9. LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class SegLinearProbe(MethodTrainingMixin, L.LightningModule):
    """
    Owns the model directly (no terratorch task wrapper) to avoid
    self.log control-flow issues.

    Uses PrithviModelFactory with UperNetDecoder + necks, same as notebook.
    All metrics (loss, mIoU) logged here — callbacks only accumulate arrays.
    """

    def __init__(self, model, lr: float, weight_decay: float,
                 num_classes: int, ignore_index: int,
                 class_weights=None,
                 reweighter=None,
                 use_dbr: bool = False,
                 biome_map: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "reweighter"])
        self.model       = model
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reweighter   = reweighter
        self.use_dbr      = use_dbr and (reweighter is not None)
        self.biome_map   = biome_map or {}

        # set by BiomeMIoULogger after each val epoch
        self.val_biome_metrics: dict = {}

        # loss
        w = torch.tensor(class_weights, dtype=torch.float32) \
            if class_weights else None
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=w, ignore_index=ignore_index
        )
        self.loss_fn_none = torch.nn.CrossEntropyLoss(
            weight=w, ignore_index=ignore_index, reduction="none"
        )

        # torchmetrics mIoU for train/val logging
        self.train_miou = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average="macro"
        )
        self.val_miou = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average="macro"
        )

        # freeze encoder if LP
        if FREEZE_BACKBONE:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()
            frozen    = sum(p.numel() for p in self.model.encoder.parameters())
            trainable = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)
            print(f"\n[Model] Frozen    : {frozen/1e6:.1f}M  (encoder)")
            print(f"[Model] Trainable : {trainable/1e6:.1f}M  (decoder+head)")
            print(f"[DBR]   Active    : {self.use_dbr}")
        else:
            total = sum(p.numel() for p in self.model.parameters())
            print(f"\n[Model] Full fine-tune: {total/1e6:.1f}M params")

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

    def forward(self, x, sample_ids=None):
        out = self.model(x)
        return out.output if hasattr(out, "output") else out   # (B, C, H, W)





    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_orig/miou"}}


# ─────────────────────────────────────────────────────────────────────────────
# 10. BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(weights_path: Path, num_classes: int):
    factory = PrithviModelFactory()
    return factory.build_model(
        task                          = "segmentation",
        backbone                      = "prithvi_eo_v2_300_tl",
        decoder                       = "UperNetDecoder",
        bands                         = BANDS,
        num_classes                   = num_classes,
        pretrained                    = True,
        backbone_ckpt_path=str(weights_path),
        decoder_channels              = 256,
        decoder_scale_modules         = True,
        head_dropout                  = HEAD_DROPOUT,
    #     necks                         = [
    #     {"name": "SelectIndices",        "indices": NECK_INDICES},
    #     {"name": "ReshapeTokensToImage", "effective_time_dim": NUM_FRAMES},
    # ],
        backbone_num_frames           = NUM_FRAMES,
        rescale                       = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. TEST METRICS EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test_split(lp_model, test_loader, biome_map, device):
    training_state = lp_model.training
    lp_model.eval()
    lp_model.to(device)

    all_true = []
    all_pred = []
    biome_data = defaultdict(lambda: {"true": [], "pred": []})

    with torch.no_grad():
        for x, masks, sample_ids in test_loader:
            x = x.to(device)
            logits = lp_model(x, sample_ids)
            preds = logits.argmax(dim=1).cpu().numpy()
            masks = masks.numpy()

            for i, sid in enumerate(sample_ids):
                biome = biome_map.get(sid) or biome_map.get(Path(sid).stem)
                if biome is None or biome == "Unknown":
                    continue
                flat_true = masks[i].reshape(-1)
                flat_pred = preds[i].reshape(-1)
                biome_data[biome]["true"].append(flat_true)
                biome_data[biome]["pred"].append(flat_pred)
                all_true.append(flat_true)
                all_pred.append(flat_pred)

    if training_state:
        lp_model.train()

    if not all_true:
        print("[Metrics Evaluation] No valid test samples found!")
        return None

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # Helper for valid pixels
    def _valid_pixels(yt, yp, ignore):
        m = yt != ignore
        return yt[m], yp[m]

    # Helper to compute IoU
    def compute_miou_local(yt_flat, yp_flat, num_classes, ignore):
        yt, yp = _valid_pixels(yt_flat, yp_flat, ignore)
        present = [c for c in range(num_classes) if (yt == c).sum() > 0]
        if not present:
            return float("nan")
        ious = []
        for c in present:
            tp = ((yp == c) & (yt == c)).sum()
            fp = ((yp == c) & (yt != c)).sum()
            fn = ((yp != c) & (yt == c)).sum()
            denom = tp + fp + fn
            ious.append(float(tp / denom) if denom > 0 else float("nan"))
        valid = [v for v in ious if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    # Helper to compute Dice (F1)
    def compute_mean_dice_local(yt_flat, yp_flat, num_classes, ignore):
        yt, yp = _valid_pixels(yt_flat, yp_flat, ignore)
        present = [c for c in range(num_classes) if (yt == c).sum() > 0]
        if not present:
            return float("nan"), np.full(num_classes, float("nan"))
        per_class = np.full(num_classes, float("nan"))
        for c in present:
            tp = ((yp == c) & (yt == c)).sum()
            fp = ((yp == c) & (yt != c)).sum()
            fn = ((yp != c) & (yt == c)).sum()
            denom = 2 * tp + fp + fn
            per_class[c] = float(2 * tp / denom) if denom > 0 else float("nan")
        valid = per_class[~np.isnan(per_class)]
        return (float(np.mean(valid)) if len(valid) > 0 else float("nan")), per_class

    # Helper to compute Pixel Accuracy
    def compute_pixel_accuracy_local(yt_flat, yp_flat, ignore):
        yt, yp = _valid_pixels(yt_flat, yp_flat, ignore)
        if len(yt) == 0:
            return float("nan")
        return float((yp == yt).sum() / len(yt))

    # Compute overall metrics
    global_miou = compute_miou_local(all_true, all_pred, NUM_CLASSES, IGNORE_INDEX)
    global_dice, _ = compute_mean_dice_local(all_true, all_pred, NUM_CLASSES, IGNORE_INDEX)
    pix_acc = compute_pixel_accuracy_local(all_true, all_pred, IGNORE_INDEX)

    # Biome-wise metrics
    biome_mious = {}
    biome_dices = {}

    for biome, data in biome_data.items():
        yt = np.concatenate(data["true"])
        yp = np.concatenate(data["pred"])
        # mIoU
        biome_mious[biome] = compute_miou_local(yt, yp, NUM_CLASSES, IGNORE_INDEX)
        # Dice (F1)
        dice_val, _ = compute_mean_dice_local(yt, yp, NUM_CLASSES, IGNORE_INDEX)
        biome_dices[biome] = dice_val

    # Fairness metrics for mIoU
    valid_mious = [v for v in biome_mious.values() if not np.isnan(v)]
    if len(valid_mious) >= 2:
        worst_miou = float(np.min(valid_mious))
        mean_miou  = float(np.mean(valid_mious))
        std_miou   = float((np.std(valid_mious, ddof=1) if len(valid_mious) > 1 else 0.0))
        nfr_miou   = (max(valid_mious) - min(valid_mious)) / mean_miou if mean_miou > 0 else float("nan")

        # Equity scaled miou
        dev = sum(abs(v - global_miou) for v in valid_mious)
        es_miou = float(global_miou / (dev + 1))
        # Equity scaled std miou
        es_std_miou = float(global_miou / (std_miou + 1))
    else:
        worst_miou = std_miou = nfr_miou = es_miou = es_std_miou = float("nan")

    # Helper to compute Equalized Odds
    def compute_eqodds_local(biome_data, num_classes, ignore):
        biomes = sorted(biome_data.keys())
        if len(biomes) < 2:
            return float("nan")

        J = {}
        for b, data in biome_data.items():
            yt = np.concatenate(data["true"])
            yp = np.concatenate(data["pred"])

            m = yt != ignore
            yt_valid = yt[m]
            yp_valid = yp[m]

            j_b = np.full(num_classes, float("nan"))
            for c in range(num_classes):
                yt_c = (yt_valid == c).astype(int)
                yp_c = (yp_valid == c).astype(int)

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
            for b2 in biomes[i + 1:]:
                diffs = [abs(J[b1][c] - J[b2][c])
                         for c in range(num_classes)
                         if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
                if diffs:
                    worst = max(worst, float(np.mean(diffs)))
        return worst


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    global BATCH_SIZE
    import logging
    import argparse
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Prithvi DBR Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    BATCH_SIZE = args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dbr_tag = f"dbr_alpha{DBR_ALPHA}_ema{DBR_EMA}" if USE_DBR else "erm"
    mode_str = f"sa_crop_{'lp' if FREEZE_BACKBONE else 'ft'}_{dbr_tag}"
    OUT_DIR  = Path(args.out_dir) if args.out_dir else BASE_DIR / f"runs/{mode_str}/seed_{seed}"
    CKPT_DIR = OUT_DIR  / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Redirect stdout & stderr to train.log ──
    log_file = OUT_DIR / "train.log"
    sys.stdout = TeeLogger(log_file)
    sys.stderr = sys.stdout
    print(f"\n[Logging] Output mirrored to: {log_file}")

    print(f"[Config] Dataset    : m-SA-crop-type  ({NUM_CLASSES} classes)")
    print(f"[Config] Mode       : {'Linear Probe' if FREEZE_BACKBONE else 'Fine-tune'}")
    print(f"[Config] DBR        : active={USE_DBR}, alpha={DBR_ALPHA}, ema={DBR_EMA}")
    print(f"[Config] Seed       : {seed}")

    biome_map = load_biome_map(BIOME_JSON)

    geobench.GEO_BENCH_DIR         = str(DATA_DIR)
    geobench.task.GEO_BENCH_DIR    = DATA_DIR
    os.environ["GEO_BENCH_DIR"]    = str(DATA_DIR)

    tasks = list(geobench.task_iterator(
        benchmark_name=BENCHMARK_NAME,
        benchmark_dir=str(BENCHMARK_DIR)
    ))
    task  = None
    for t in tasks:
        if TASK_KEYWORD in t.dataset_name.lower():
            task = t
            break
    assert task is not None, \
        f"'{TASK_KEYWORD}' not found in {[t.dataset_name for t in tasks]}"
    print(f"[Data] Task: {task.dataset_name}")

    # ── ID diagnostic ──
    _raw = task.get_dataset(split="test")
    _sids = [str(_raw[i].sample_name) for i in range(min(5, len(_raw)))]
    _hits = sum(1 for s in _sids
                if biome_map.get(s) or biome_map.get(Path(s).stem))
    print(f"[IDDiag] Sample IDs : {_sids}")
    print(f"[IDDiag] Hits 5/5   : {_hits}/5",
          "OK" if _hits > 0 else "<-- MISMATCH!")

    # ── Datasets ──
    train_ds = GeoBenchSegDataset(task, "train", augment=True)
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchSegDataset(task, "valid", augment=False)
    test_ds  = GeoBenchSegDataset(task, "test",  augment=False)

    # ── Print Dataset Biome Sample Distribution at Startup ──
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  Dataset Biome Sample Counts Summary")
    print("══════════════════════════════════════════════════════════════════════")
    print_split_biome_distribution("train", train_ds, biome_map)
    print_split_biome_distribution("valid", val_ds, biome_map)
    print_split_biome_distribution("test", test_ds, biome_map)
    print("══════════════════════════════════════════════════════════════════════\n")

    # ── DataLoaders ──
    train_loader    = training_loader(train_ds, method='dbr', groups=protocol_groups, batch_size=BATCH_SIZE,
                                 shuffle=True,  num_workers=NUM_WORKERS,
                                 pin_memory=True, persistent_workers=True)
    val_loader      = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True, persistent_workers=True)
    test_loader     = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True, persistent_workers=True)

    # ── Model ──
    base_model = build_model(WEIGHTS_PT, NUM_CLASSES)

    reweighter = None

    lp_model = SegLinearProbe(
        model         = base_model,
        lr            = LR,
        weight_decay  = WEIGHT_DECAY,
        num_classes   = NUM_CLASSES,
        ignore_index  = IGNORE_INDEX,
        class_weights = CLASS_WEIGHTS,
        reweighter    = reweighter,
        use_dbr       = USE_DBR,
        biome_map     = biome_map,
    )
    configure_training_protocol(lp_model, 'dbr', 'sa_crop', protocol_groups, args, ignore_index=IGNORE_INDEX, lookup=lambda sid: lookup_biome(biome_map, sid))
    args.method_protocol = lp_model.method_protocol.config

    # ── Separate Checkpoint Subdirectories ──
    CKPT_MACRO_DIR = CKPT_DIR / "best_macro_miou"
    CKPT_WORST_DIR = CKPT_DIR / "best_worst_miou"
    CKPT_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_WORST_DIR.mkdir(parents=True, exist_ok=True)

    # ── Dual Checkpoint Callbacks ──
    ckpt_macro_cb = ModelCheckpoint(
        monitor                 = 'val/benchmark_miou',
        mode                    = "max",
        dirpath                 = CKPT_MACRO_DIR,
        filename                = f"{mode_str}-macro-epoch={{epoch:02d}}-val_miou={{val/benchmark_miou:.4f}}",
        auto_insert_metric_name = False,
        save_top_k              = 10,
        save_last               = True,
        verbose                 = False,
    )
    ckpt_worst_cb = ModelCheckpoint(
        monitor                 = "val/worst_case_miou",
        mode                    = "max",
        dirpath                 = CKPT_WORST_DIR,
        filename                = f"{mode_str}-worst-epoch={{epoch:02d}}-worst_miou={{val/worst_case_miou:.4f}}",
        auto_insert_metric_name = False,
        save_top_k              = 10,
        save_last               = False,
        verbose                 = False,
    )

    early_stop_cb = EarlyStopping(
        monitor   = 'val/benchmark_miou',
        mode      = "max",
        patience  = EARLY_STOP_PATIENCE,
        min_delta = EARLY_STOP_MIN_DELTA,
        verbose   = False,
    )
    global_miou_cb = GlobalMIoULogger(
        num_classes  = NUM_CLASSES,
        ignore_index = IGNORE_INDEX,
    )
    biome_cb = BiomeMIoULogger(
        biome_map             = biome_map,
        num_classes           = NUM_CLASSES,
        class_names           = CLASS_NAMES,
        output_dir            = OUT_DIR,
        ignore_index          = IGNORE_INDEX,
        test_loader           = test_loader,
    )

    tb_logger = TensorBoardLogger(
        save_dir = OUT_DIR / "tb_logs",
        name     = mode_str,
        version  = 0,
    )

    progress_bar_cb = CleanProgressBar()

    f1_cb = BiomeMacroF1Logger(biome_map, num_classes=NUM_CLASSES, output_dir=OUT_DIR, ignore_index=IGNORE_INDEX)

    trainer = L.Trainer(
        accelerator             = "gpu",
        devices                 = 1,
        strategy                = "auto",
        precision               = "bf16-mixed",
        max_epochs              = args.epochs,
        check_val_every_n_epoch = 1,
        log_every_n_steps       = 10,
        logger                  = tb_logger,
        callbacks               = [global_miou_cb, ckpt_macro_cb,
                                   ckpt_worst_cb, early_stop_cb, biome_cb,
                                   progress_bar_cb],
        enable_checkpointing    = True,
        enable_progress_bar     = True,
        deterministic           = "warn",
        benchmark               = False,
        num_sanity_val_steps    = 2,
    )

    record_config(OUT_DIR, __file__, args)
    trainer.fit(lp_model, train_loader, val_loader, ckpt_path=args.resume_from)
    finalize_selected(trainer, lp_model)

    ckpt_path = CKPT_DIR / f"{mode_str}_weights.pt"
    torch.save({
        "decoder"        : lp_model.model.decoder.state_dict(),
        "head"           : lp_model.model.head.state_dict(),
        "epoch"          : trainer.current_epoch,
        "val_miou"       : ckpt_macro_cb.best_model_score,
        "bands"          : BAND_INDICES,
        "num_classes"    : NUM_CLASSES,
        "num_frames"     : NUM_FRAMES,
        "freeze_backbone": True,
    }, ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
