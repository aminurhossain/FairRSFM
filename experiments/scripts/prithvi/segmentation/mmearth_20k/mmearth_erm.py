"""
Prithvi 2.0 — MMEarth Dynamic World Segmentation (Super Training File)
=======================================================================
Three training modes — set exactly ONE to True:

  USE_NORMAL_LP = True   # Standard linear probe (backbone frozen, CE loss)
  USE_DBR       = False  # Dynamic Biome Reweighting with Validation Feedback
  USE_BOLP      = False  # Biome-Orthogonal Linear Probing (feature-space debiasing)

Biome grouping: 14 RESOLVE biomes → 5 macro groups (from tile_info.json, 1-indexed):
  "high biomass"                    → 1, 3, 5
  "High amplitude phenological"     → 2, 4, 6
  "transitional herbaceous & scrub" → 7, 8, 12
  "cryospheric & short-cycle"       → 10, 11
  "xeric & mineralogical"           → 13
  "hydrologically modulated"        → 9, 14

JSON per-epoch metrics saved to <out_dir>/test_metrics.json:
  epoch_N → { val: { global_miou, worst_biome_miou, nfr, biome_std,
                     eqodds, macro_f1, biome_wise_miou_table },
              test: { same keys } }
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 0. SYSTEM SETUP
# ─────────────────────────────────────────────────────────────────────────────
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
torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from lightning.pytorch.callbacks import TQDMProgressBar

class TeeLogger:
    def __init__(self, log_file_path: Path, stream):
        self.terminal = stream
        self.log_file_path = log_file_path
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")

    def write(self, message):
        # ALWAYS show everything in terminal
        self.terminal.write(message)
        self.terminal.flush()

        # But do NOT save tqdm redraw/progress-bar noise
        if "\r" in message or "%|" in message or "it/s]" in message:
            return

        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

class CleanProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)

        for k in list(items.keys()):
            if k.startswith("val/"):
                items.pop(k, None)

        return items
    
def strip_encoder_from_checkpoint(ckpt_path: Path):
    if not ckpt_path.exists(): return
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        keys = list(checkpoint["state_dict"].keys())
        for k in keys:
            if k.startswith("model.encoder.") or k.startswith("model.backbone."):
                del checkpoint["state_dict"][k]
        torch.save(checkpoint, ckpt_path)

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F

from lightning.pytorch.callbacks import (
    Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from terratorch.tasks import SemanticSegmentationTask

from mmearth_dynamic_world_datamodule import (
    DYNAMIC_WORLD_CLASSES,
    MMEarthDynamicWorldSegmentationDataModule,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIG  ← only block to edit between runs
# ─────────────────────────────────────────────────────────────────────────────

# ── Training mode — set exactly ONE to True ──
USE_NORMAL_LP = True    # Standard LP: backbone frozen, plain CE loss
USE_DBR       = False   # Dynamic Biome Reweighting with Validation Feedback
USE_BOLP      = False   # Biome-Orthogonal Linear Probing

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get("MMEARTH_DATA_DIR", "/data1/aminur/om/MMearth20k/hardcoded")
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "output")

# ── Bands ──
PRITHVI_BANDS   = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
MMEARTH_S2_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]

# ── Hyperparams ──
BATCH_SIZE   = 122
EPOCHS = 50
LR           = 1.0e-3
WEIGHT_DECAY = 0.01
SEED         = 0
NUM_WORKERS  = 4

# ── Trainer ──
ACCELERATOR = "gpu"
DEVICES     = 1
PRECISION   = "16-mixed"

# ── Decoder ──
DECODER = "linear"   # "linear", "upernet", "unet"

# ── Val / test split fractions (used when dataset has no explicit split) ──
VAL_FRACTION  = 0.1
TEST_FRACTION = 0.1

# ── DBR config ──
DBR_ALPHA          = 0.5   # feedback strength
DBR_EMA            = 0.5   # metric EMA coefficient
DBR_UPDATE_EVERY   = 1     # update weights every N epochs

# ── BOLP config ──
BOLP_K = 5   # number of top singular vectors to project out (rank of null-space)

# ── Early stopping ──
EARLY_STOP_PATIENCE  = 30
EARLY_STOP_MIN_DELTA = 1e-4

# ── Misc ──
CONSOLE_LOG_EVERY_N_STEPS =400
SAMPLE_WARN_THRESHOLD     = 10   # warn if biome group has fewer samples than this


# ─────────────────────────────────────────────────────────────────────────────
# 3. BIOME MACRO-GROUP MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# tile_info.json stores biome as 0-indexed int → add 1 to get 1-indexed key below
BIOME_MACRO_GROUPS: dict[int, str] = {
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

# All unique macro group names
MACRO_GROUP_NAMES = sorted(set(BIOME_MACRO_GROUPS.values()))

BIOME_NAMES_14 = [
    "Tropical and Subtropical Moist Broadleaf Forests",       # 1
    "Tropical and Subtropical Dry Broadleaf Forests",         # 2
    "Tropical and Subtropical Coniferous Forests",            # 3
    "Temperate Broadleaf and Mixed Forests",                  # 4
    "Temperate Coniferous Forests",                           # 5
    "Boreal forests/Taiga",                                   # 6
    "Tropical and Subtropical Grasslands, Savannas",          # 7
    "Temperate Grasslands, Savannas, and Shrublands",         # 8
    "Flooded Grasslands and Savannas",                        # 9
    "Montane Grasslands and Shrublands",                      # 10
    "Tundra",                                                 # 11
    "Mediterranean Forests, Woodlands, and Scrub",            # 12
    "Deserts and Xeric Shrublands",                           # 13
    "Mangroves",                                              # 14
]


def biome_int_to_macro_group(biome_0idx: int) -> str | None:
    """Convert 0-indexed biome int from tile_info.json to macro group name."""
    key = biome_0idx + 1  # tile_info uses 0-based; macro dict uses 1-based
    return BIOME_MACRO_GROUPS.get(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TILE INFO LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_tile_info(data_root: str | Path) -> dict:
    """Load tile_info.json from data root. Returns {tile_id_str: {biome: int, ...}}."""
    data_root = Path(data_root)
    matches = sorted(data_root.glob("*tile_info.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one *tile_info.json in {data_root}, found {len(matches)}."
        )
    return json.loads(matches[0].read_text(encoding="utf-8"))


def tile_id_to_macro_group(tile_id: str, tile_info: dict) -> str | None:
    """Look up macro group for a tile ID string."""
    info = tile_info.get(str(tile_id))
    if info is None:
        return None
    return biome_int_to_macro_group(int(info["biome"]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def compute_gini(values: list) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2: return 0.0
    array = np.sort(np.array(vals))
    n = len(array)
    sum_val = np.sum(array)
    if sum_val == 0: return 0.0
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * array) / (n * sum_val))

def compute_dpm_seg(biome_data: dict, num_classes: int) -> float:
    biomes = sorted(biome_data.keys())
    if len(biomes) < 2: return float("nan")
    PPR = {}
    for b, data in biome_data.items():
        if not data["pred"]: continue
        yp = np.concatenate(data["pred"])
        valid_mask = np.concatenate(data["true"]) != -1
        yp_valid = yp[valid_mask]
        ppr_b = np.zeros(num_classes)
        total = len(yp_valid)
        if total > 0:
            for c in range(num_classes):
                ppr_b[c] = (yp_valid == c).sum() / total
        PPR[b] = ppr_b
    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diffs = [abs(PPR[b1][c] - PPR[b2][c]) for c in range(num_classes)]
            worst = max(worst, float(np.mean(diffs)))
    return worst

def compute_miou(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    mask   = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
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
    valid = [v for v in ious if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_macro_f1_seg(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    """Macro-averaged F1 (Dice) over classes present in y_true (pixel-level)."""
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
    valid = [v for v in f1s if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_eqodds_seg(
    biome_data: dict,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    """
    Equalized Odds Difference over macro biome groups at the pixel level.

    For each class c and biome b, compute Youden's J = TPR_bc - FPR_bc.
    EqOdds = max over biome pairs of mean |J_b1_c - J_b2_c| over classes.
    """
    biomes = sorted(biome_data.keys())
    if len(biomes) < 2:
        return float("nan")

    J: dict[str, np.ndarray] = {}
    for b, data in biome_data.items():
        if not data["true"]:
            continue
        yt = np.concatenate(data["true"])
        yp = np.concatenate(data["pred"])
        mask      = yt != ignore_index
        yt_valid  = yt[mask]
        yp_valid  = yp[mask]
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
            if not (math.isnan(tpr) or math.isnan(fpr)):
                j_b[c] = tpr - fpr
        J[b] = j_b

    worst = 0.0
    biome_list = sorted(J.keys())
    for i, b1 in enumerate(biome_list):
        for b2 in biome_list[i + 1:]:
            diffs = [
                abs(J[b1][c] - J[b2][c])
                for c in range(num_classes)
                if not (math.isnan(J[b1][c]) or math.isnan(J[b2][c]))
            ]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
    return worst


def compute_fairness_metrics(
    biome_mious: dict[str, float],
    global_miou: float,
) -> dict[str, float]:
    """NFR, biome_std, worst_biome_miou from a {group: miou} dict."""
    valid = [v for v in biome_mious.values() if not math.isnan(v)]
    if len(valid) < 2:
        return {
            "worst_biome_miou": float("nan"),
            "nfr": float("nan"),
            "biome_std": float("nan"),
        }
    worst = float(np.min(valid))
    mean  = float(np.mean(valid))
    std   = float(np.std(valid))
    nfr   = (max(valid) - min(valid)) / mean if mean > 0 else float("nan")
    return {
        "worst_biome_miou": worst,
        "nfr": nfr,
        "biome_std": std,
    }


def evaluate_split(
    model: SemanticSegmentationTask,
    dataloader: DataLoader,
    tile_info: dict,
    num_classes: int,
    device: torch.device,
    ignore_index: int = -1,
    bolp_module: "BOLPModule | None" = None,
) -> dict:
    model.eval()
    biome_data: dict[str, dict] = {}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images    = batch["image"].to(device)
            masks     = batch["mask"].squeeze(1)
            filenames = batch["filename"]

            if bolp_module is not None:
                logits = bolp_module.forward_with_bolp(model, images)
            else:
                logits = model.model(images).output

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i, tile_id in enumerate(filenames):
                group = tile_id_to_macro_group(str(tile_id), tile_info)
                flat_true = masks_np[i].reshape(-1)
                flat_pred = preds[i].reshape(-1)
                all_true.append(flat_true)
                all_pred.append(flat_pred)
                if group is not None:
                    if group not in biome_data:
                        biome_data[group] = {"true": [], "pred": []}
                    biome_data[group]["true"].append(flat_true)
                    biome_data[group]["pred"].append(flat_pred)

    if not all_true:
        return {}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    global_miou = compute_miou(y_true, y_pred, num_classes, ignore_index)
    macro_f1    = compute_macro_f1_seg(y_true, y_pred, num_classes, ignore_index)
    eqodds      = compute_eqodds_seg(biome_data, num_classes, ignore_index)
    dpm         = compute_dpm_seg(biome_data, num_classes)

    biome_mious: dict[str, float] = {}
    biome_f1s: dict[str, float] = {}
    for group, data in biome_data.items():
        yt = np.concatenate(data["true"])
        yp = np.concatenate(data["pred"])
        biome_mious[group] = compute_miou(yt, yp, num_classes, ignore_index)
        biome_f1s[group] = compute_macro_f1_seg(yt, yp, num_classes, ignore_index)

    fairness = compute_fairness_metrics(biome_mious, global_miou)
    
    valid_mious = [v for v in biome_mious.values() if not math.isnan(v)]
    gini = compute_gini(valid_mious)
    per = (min(valid_mious) / max(valid_mious)) if valid_mious and max(valid_mious) > 0 else float('nan')

    return {
        "global_miou":         global_miou,
        "macro_f1":            macro_f1,
        "eqodds":              eqodds,
        "dpm":                 dpm,
        "gini":                gini,
        "per":                 per,
        "worst_biome_miou":    fairness["worst_biome_miou"],
        "nfr":                 fairness["nfr"],
        "biome_std":           fairness["biome_std"],
        "biome_wise_miou_table": biome_mious,
        "biome_wise_f1_table": biome_f1s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. JSON METRICS SAVER
# ─────────────────────────────────────────────────────────────────────────────

def save_epoch_metrics(
    out_dir: Path,
    epoch: int,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    json_path = out_dir / "test_metrics.json"
    history: dict = {}
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                history = json.load(f)
        except Exception as e:
            print(f"[MetricsSaver] Could not load existing file: {e}")

    history[f"epoch_{epoch}"] = {
        "epoch": epoch,
        "val":   val_metrics,
        "test":  test_metrics,
    }

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[MetricsSaver] Saved epoch_{epoch} metrics → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSOLE LOGGER CALLBACK  (unchanged from original file)
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleMetricsLogger(Callback):
    def __init__(self, every_n_steps: int = 20) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps

    @staticmethod
    def _to_float(value: object) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @classmethod
    def _format_metrics(cls, metrics: dict, prefixes: tuple) -> str:
        parts = []
        for key in sorted(metrics):
            if not key.startswith(prefixes):
                continue
            value = cls._to_float(metrics[key])
            if value is not None:
                parts.append(f"{key}={value:.4f}")
        return " ".join(parts)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Filter out intermediate step metrics to keep the epoch summary clean
        filtered_metrics = {
            k: v for k, v in trainer.callback_metrics.items() 
            if k.startswith("train/") and not k.endswith("_step")
        }
        metrics = self._format_metrics(filtered_metrics, ("train/",))
        if metrics:
            print(f"[train] epoch={trainer.current_epoch} {metrics}", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = self._format_metrics(trainer.callback_metrics, ("val/",))
        if metrics:
            print(f"[val] epoch={trainer.current_epoch} step={trainer.global_step} {metrics}", flush=True)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        metrics = self._format_metrics(trainer.callback_metrics, ("test/",))
        if metrics:
            print(f"[test] step={trainer.global_step} {metrics}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PER-EPOCH METRICS CALLBACK  (val + test JSON export every epoch)
# ─────────────────────────────────────────────────────────────────────────────

class EpochMetricsSaverCallback(Callback):
    """
    After every validation epoch (skipping sanity check):
      1. Compute val metrics over val_loader
      2. Compute test metrics over test_loader
      3. Append both to test_metrics.json under key epoch_N
    Also stores val biome mious on pl_module for DBR to consume.
    """

    def __init__(
        self,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        tile_info:    dict,
        num_classes:  int,
        out_dir:      Path,
        device:       torch.device,
        ignore_index: int = -1,
        bolp_module:  "BOLPModule | None" = None,
    ) -> None:
        super().__init__()
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.tile_info    = tile_info
        self.num_classes  = num_classes
        self.out_dir      = out_dir
        self.device       = device
        self.ignore_index = ignore_index
        self.bolp_module  = bolp_module

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch

        val_metrics  = evaluate_split(
            pl_module, self.val_loader,  self.tile_info,
            self.num_classes, self.device, self.ignore_index, self.bolp_module,
        )
        test_metrics = evaluate_split(
            pl_module, self.test_loader, self.tile_info,
            self.num_classes, self.device, self.ignore_index, self.bolp_module,
        )

        # expose val biome mious to pl_module so DBR callback can read them
        pl_module.val_biome_metrics = val_metrics.get("biome_wise_miou_table", {})

        save_epoch_metrics(self.out_dir, epoch, val_metrics, test_metrics)

        # print summary
        print(
            f"\\n"
            f"  ── Representational Fairness & Overall Performance ────────────────────────\\n"
            f"  Overall mIoU                                  {val_metrics.get('global_miou', float('nan')):.4f}\\n"
            f"  Worst-case mIoU                               {val_metrics.get('worst_biome_miou', float('nan')):.4f}\\n"
            f"  Overall Macro F1                              {val_metrics.get('macro_f1', float('nan')):.4f}\\n"
            f"  Performance Equity Ratio (Worst/Best mIoU)    {val_metrics.get('per', float('nan')):.4f}\\n"
            f"  Gini Coefficient (Inequality)                 {val_metrics.get('gini', float('nan')):.4f}\\n"
            f"  Group mIoU Std                                {val_metrics.get('biome_std', float('nan')):.4f}\\n"
            f"  Equalized Odds Disparity                      {val_metrics.get('eqodds', float('nan')):.4f}\\n"
            f"  Demographic Parity Disparity                  {val_metrics.get('dpm', float('nan')):.4f}\\n"
            f"  [Test mIoU]                                   {test_metrics.get('global_miou', float('nan')):.4f}\\n"
            f"══════════════════════════════════════════════════════════════════════════════════\\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. DBR  (Dynamic Biome Reweighting with Validation Feedback)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicBiomeReweighter:
    """
    Initialized with static inverse-frequency weights from the training set.
    After each validation epoch, weights are updated with:
        w_b ← w_b * (1 + α * (mean_M - M_b) / (mean_M + ε))
    then renormalized so the average weight = 1.
    EMA is applied to running per-biome metrics for smoother trajectories.
    """

    def __init__(
        self,
        train_biome_groups: list[str],   # macro group name per training sample
        alpha: float = DBR_ALPHA,
        ema:   float = DBR_EMA,
    ) -> None:
        self.alpha = alpha
        self.ema   = ema

        counts = Counter(train_biome_groups)
        total  = sum(counts.values())
        raw    = {b: total / c for b, c in counts.items()}
        norm   = sum(raw.values()) / len(raw)
        self.weights: dict[str, float] = {b: w / norm for b, w in raw.items()}
        self.running_metrics: dict[str, float] = {b: 0.5 for b in self.weights}

        print("[DBR] Initial biome weights (inverse-frequency):")
        for b, w in sorted(self.weights.items()):
            print(f"  {b:<45}  w={w:.4f}  (n={counts[b]})")

    def update(self, val_biome_metrics: dict[str, float]) -> None:
        for b, m in val_biome_metrics.items():
            if b not in self.running_metrics:
                continue
            self.running_metrics[b] = (
                self.ema * self.running_metrics[b] + (1 - self.ema) * m
            )
        mean_m = float(np.mean(list(self.running_metrics.values())))
        for b in self.weights:
            gap = (mean_m - self.running_metrics[b]) / (mean_m + 1e-8)
            self.weights[b] *= (1 + self.alpha * gap)
        avg = float(np.mean(list(self.weights.values())))
        for b in self.weights:
            self.weights[b] /= avg

    def get_weight(self, group: str | None) -> float:
        if group is None:
            return 1.0
        return self.weights.get(group, 1.0)


class DBRWeightUpdateCallback(Callback):
    """
    At the start of each training epoch (after epoch 0):
      - Read pl_module.val_biome_metrics (set by EpochMetricsSaverCallback)
      - Update the reweighter
      - Print updated weights
    """

    def __init__(self, reweighter: DynamicBiomeReweighter, update_every: int = 1) -> None:
        super().__init__()
        self.reweighter   = reweighter
        self.update_every = update_every

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if epoch == 0:
            return
        if epoch % self.update_every != 0:
            return
        biome_metrics = getattr(pl_module, "val_biome_metrics", {})
        if not biome_metrics:
            return
        self.reweighter.update(biome_metrics)
        print(f"\n[DBR] Updated weights at epoch {epoch}:")
        for b, w in sorted(self.reweighter.weights.items()):
            m = self.reweighter.running_metrics.get(b, float("nan"))
            print(f"  {b:<45}  w={w:.4f}  ema_miou={m:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. DBR-AWARE SemanticSegmentationTask WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class DBRSemanticSegmentationTask(SemanticSegmentationTask):
    """
    Thin subclass of SemanticSegmentationTask that overrides training_step
    to apply per-sample biome loss weighting when use_dbr=True.

    The standard SemanticSegmentationTask computes CE loss internally.
    Here we recompute it with reduction='none', apply biome weights, then mean.
    """

    def __init__(
        self,
        *args,
        reweighter:   DynamicBiomeReweighter | None = None,
        tile_info:    dict | None = None,
        use_dbr:      bool = False,
        ignore_index: int  = -1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reweighter   = reweighter
        self.tile_info    = tile_info or {}
        self.use_dbr      = use_dbr and (reweighter is not None)
        self._ignore_index = ignore_index
        # will be set by EpochMetricsSaverCallback each epoch
        self.val_biome_metrics: dict = {}

    def training_step(self, batch, batch_idx):
        if not self.use_dbr:
            return super().training_step(batch, batch_idx)

        # ── DBR path: compute per-sample weighted loss ──
        images    = batch["image"]
        masks     = batch["mask"].squeeze(1)   # (B, H, W) long
        filenames = batch.get("filename", [None] * images.size(0))

        logits = self.model(images).output     # (B, C, H, W)

        # per-pixel loss → (B, H, W)
        loss_pixel = F.cross_entropy(
            logits, masks,
            ignore_index=self._ignore_index,
            reduction="none",
        )

        # mean over valid pixels per sample → (B,)
        valid_mask  = (masks != self._ignore_index).float()
        loss_sum    = (loss_pixel * valid_mask).sum(dim=(1, 2))
        valid_count = valid_mask.sum(dim=(1, 2)).clamp(min=1)
        per_sample_loss = loss_sum / valid_count  # (B,)

        # biome weights per sample
        weights = torch.tensor(
            [
                self.reweighter.get_weight(
                    tile_id_to_macro_group(str(tid), self.tile_info)
                )
                for tid in filenames
            ],
            dtype=torch.float32,
            device=logits.device,
        )

        loss = (per_sample_loss * weights).mean()

        # log manually (mirrors what SemanticSegmentationTask does)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=images.size(0))
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 11. BOLP MODULE
# ─────────────────────────────────────────────────────────────────────────────

class BOLPModule:
    """
    Biome-Orthogonal Linear Probing preprocessing.

    Phase A (precompute, call once before training):
      1. Extract frozen encoder features for all training samples
         → mean-pool patch tokens to get one vector per image
      2. Compute per-biome centroids µ_b and global mean µ_0
      3. Form M = [µ_b - µ_0, ...], compute thin SVD, keep top-k left singular vectors U_k
      4. Build projection matrix P = I - U_k @ U_k.T

    Phase B (forward_with_bolp, called at training/val/test time):
      z     = mean-pooled encoder features
      z_res = z - µ_biome + µ_0           (Step 1: residualize)
      z_hat = P @ z_res                   (Step 2: project out biome directions)
      logits = decoder(z_hat)             ← NOT YET: for segmentation we inject
                                            z_hat back as the token sequence
                                            before the decoder neck.

    Note on segmentation BOLP:
      Prithvi outputs patch token sequences of shape (B, N_tokens, d).
      We apply:
        (a) mean-pool across tokens → (B, d) for centroid computation
        (b) at inference, residualize and project EACH token independently
            using the same µ_b (image-level) and P (shared), then pass the
            projected token sequence to the neck + decoder.
    """

    def __init__(self, k: int = BOLP_K, device: torch.device = torch.device("cpu")) -> None:
        self.k      = k
        self.device = device
        self.mu:    dict[str, torch.Tensor] = {}   # group → centroid (d,)
        self.mu0:   torch.Tensor | None = None     # global mean (d,)
        self.P:     torch.Tensor | None = None     # projection matrix (d, d)
        self._fitted = False

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_encoder(model: SemanticSegmentationTask):
        """
        Safely retrieve the encoder from SemanticSegmentationTask.
        Tries common attribute paths used by terratorch.
        """
        inner = getattr(model, "model", model)
        for attr in ("encoder", "backbone", "encoder_backbone"):
            enc = getattr(inner, attr, None)
            if enc is not None:
                return enc
        raise AttributeError(
            "Could not find encoder inside SemanticSegmentationTask. "
            "Tried: model.model.encoder, model.model.backbone, "
            "model.model.encoder_backbone. "
            "Please set the correct attribute path in BOLPModule._get_encoder()."
        )

    @staticmethod
    def _extract_patch_tokens(
        encoder,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run encoder forward and extract patch token embeddings.
        Returns mean-pooled token vector per image: (B, d).

        Prithvi ViT returns a list/tuple of feature maps or a tensor.
        We handle both cases.
        """
        with torch.no_grad():
            out = encoder(images)

        # terratorch Prithvi returns a list of tensors from intermediate layers
        # The last element is the final encoder output
        if isinstance(out, (list, tuple)):
            tokens = out[-1]   # (B, N_tokens, d)  or  (B, d, H, W)
        else:
            tokens = out

        if tokens.dim() == 4:
            # spatial feature map (B, d, H, W) → mean pool
            return tokens.flatten(2).mean(dim=2)   # (B, d)
        elif tokens.dim() == 3:
            # token sequence (B, N, d) → mean pool over tokens
            return tokens.mean(dim=1)              # (B, d)
        elif tokens.dim() == 2:
            return tokens                          # already (B, d)
        else:
            raise ValueError(f"Unexpected encoder output shape: {tokens.shape}")

    # ── Phase A ──────────────────────────────────────────────────────────────

    def fit(
        self,
        model:      SemanticSegmentationTask,
        dataloader: DataLoader,
        tile_info:  dict,
    ) -> None:
        """
        Precompute biome centroids and projection matrix P from training set.
        Call this ONCE before trainer.fit().
        """
        print("[BOLP] Phase A: extracting frozen encoder features...")
        encoder = self._get_encoder(model)
        encoder.eval()

        group_features: dict[str, list[torch.Tensor]] = defaultdict(list)

        for batch in dataloader:
            images    = batch["image"].to(self.device)
            filenames = batch["filename"]

            pooled = self._extract_patch_tokens(encoder, images)  # (B, d)

            for i, tile_id in enumerate(filenames):
                group = tile_id_to_macro_group(str(tile_id), tile_info)
                if group is not None:
                    group_features[group].append(pooled[i].cpu())

        if not group_features:
            raise RuntimeError(
                "[BOLP] No samples with known biome groups found in training set. "
                "Check tile_info.json and BIOME_MACRO_GROUPS mapping."
            )

        # biome centroids
        self.mu = {
            g: torch.stack(feats).mean(0).to(self.device)
            for g, feats in group_features.items()
        }
        self.mu0 = torch.stack(list(self.mu.values())).mean(0).to(self.device)

        # M matrix: columns = µ_b - µ_0  → shape (d, |B|)
        d = self.mu0.shape[0]
        M = torch.stack(
            [self.mu[g] - self.mu0 for g in sorted(self.mu.keys())],
            dim=1,
        )  # (d, |B|)

        # thin SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k = min(self.k, U.shape[1])
        Uk = U[:, :k]  # (d, k)

        # projection matrix P = I - U_k U_k^T
        self.P = torch.eye(d, device=self.device) - Uk @ Uk.T  # (d, d)
        self._fitted = True

        print(f"[BOLP] Fitted on {len(group_features)} biome groups:")
        for g in sorted(group_features.keys()):
            print(f"  {g:<45}  n={len(group_features[g])}")
        print(f"[BOLP] Projection rank k={k}, feature dim d={d}")

    # ── Phase B ──────────────────────────────────────────────────────────────

    def project_tokens(
        self,
        tokens:     torch.Tensor,    # (B, N, d) or (B, d, H, W) or (B, d)
        tile_ids:   list[str],
        tile_info:  dict,
    ) -> torch.Tensor:
        """
        Apply BOLP to a batch of encoder token sequences.
        Works for both token sequences (3D) and spatial maps (4D).
        """
        assert self._fitted, "Call BOLPModule.fit() before using project_tokens()."

        original_shape = tokens.shape
        is_spatial = tokens.dim() == 4  # (B, d, H, W)

        if is_spatial:
            B, d, H, W = tokens.shape
            tokens = tokens.permute(0, 2, 3, 1).reshape(B, H * W, d)  # (B, HW, d)
        elif tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)  # (B, 1, d)

        B, N, d = tokens.shape

        # Step 1: residualize — subtract biome centroid, add global mean (image-level)
        for i, tile_id in enumerate(tile_ids):
            group = tile_id_to_macro_group(str(tile_id), tile_info)
            if group is not None and group in self.mu:
                tokens[i] = tokens[i] - self.mu[group] + self.mu0
            # if group unknown: skip residualization, still apply projection below

        # Step 2: null-space projection — apply P to each token vector
        # tokens: (B, N, d),  P: (d, d)
        # z_hat = z @ P.T  (since P is symmetric, P.T = P, but keep general)
        tokens_proj = tokens @ self.P.T  # (B, N, d)

        if is_spatial:
            tokens_proj = tokens_proj.reshape(B, H, W, d).permute(0, 3, 1, 2)
        elif original_shape.dim() == 2 if hasattr(original_shape, 'dim') else len(original_shape) == 2:
            tokens_proj = tokens_proj.squeeze(1)

        return tokens_proj

    def forward_with_bolp(
        self,
        model:      SemanticSegmentationTask,
        images:     torch.Tensor,
        tile_ids:   list[str] | None = None,
        tile_info:  dict | None = None,
    ) -> torch.Tensor:
        """
        Full BOLP forward pass for evaluation / inference.
        Uses a hook to intercept encoder output, project it, then continue
        through neck + decoder + head.

        Returns logits (B, num_classes, H, W).
        """
        assert self._fitted, "Call BOLPModule.fit() before inference."

        if tile_ids is None:
            # no residualization possible — still apply projection
            tile_ids = [None] * images.size(0)
        if tile_info is None:
            tile_info = {}

        encoder = self._get_encoder(model)
        inner_model = getattr(model, "model", model)

        # We hook the encoder output and project before it reaches the decoder
        projected_storage: list[torch.Tensor] = []

        def _hook(module, input, output):
            if isinstance(output, (list, tuple)):
                # project the last element (final encoder features)
                out_list = list(output)
                last = out_list[-1]
                if last.dim() == 3:
                    # token sequence (B, N, d)
                    projected = self.project_tokens(
                        last.clone(), tile_ids, tile_info or {}
                    )
                elif last.dim() == 4:
                    projected = self.project_tokens(
                        last.clone(), tile_ids, tile_info or {}
                    )
                else:
                    projected = last
                out_list[-1] = projected
                projected_storage.append(projected)
                return type(output)(out_list) if isinstance(output, tuple) else out_list
            elif isinstance(output, torch.Tensor):
                projected = self.project_tokens(
                    output.clone(), tile_ids, tile_info or {}
                )
                projected_storage.append(projected)
                return projected

        handle = encoder.register_forward_hook(_hook)
        try:
            result = inner_model(images)
            logits = result.output if hasattr(result, "output") else result
        finally:
            handle.remove()

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 12. BOLP TRAINING STEP WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class BOLPSemanticSegmentationTask(SemanticSegmentationTask):
    """
    Thin subclass that injects the BOLP hook during training_step, validation_step,
    and test_step so the projected features flow through the decoder.
    """

    def __init__(
        self,
        *args,
        bolp_module: BOLPModule,
        tile_info:   dict,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bolp_module = bolp_module
        self.tile_info   = tile_info
        self.val_biome_metrics: dict = {}

    def _bolp_forward(self, batch: dict) -> torch.Tensor:
        images    = batch["image"]
        filenames = list(batch.get("filename", [None] * images.size(0)))
        return self.bolp_module.forward_with_bolp(
            self, images, tile_ids=filenames, tile_info=self.tile_info
        )

    def training_step(self, batch, batch_idx):
        masks  = batch["mask"].squeeze(1)
        logits = self._bolp_forward(batch)
        loss   = F.cross_entropy(logits, masks, ignore_index=-1)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=logits.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        masks  = batch["mask"].squeeze(1)
        logits = self._bolp_forward(batch)
        loss   = F.cross_entropy(logits, masks, ignore_index=-1)
        preds  = logits.argmax(dim=1)
        self.log("val/loss", loss, prog_bar=True, batch_size=logits.size(0))
        miou_metric = getattr(self, "val_metrics", None)
        if miou_metric is not None:
            try:
                miou_metric.update(preds, masks)
                self.log("val/mIoU", miou_metric, prog_bar=True,
                         on_epoch=True, batch_size=logits.size(0))
            except Exception:
                pass
        return loss

    def test_step(self, batch, batch_idx):
        masks  = batch["mask"].squeeze(1)
        logits = self._bolp_forward(batch)
        loss   = F.cross_entropy(logits, masks, ignore_index=-1)
        self.log("test/loss", loss, batch_size=logits.size(0))
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 13. MODEL BUILDER  (unchanged logic from original file)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    decoder:      str,
    num_classes:  int,
    lr:           float,
    weight_decay: float,
    reweighter:   DynamicBiomeReweighter | None = None,
    tile_info:    dict | None = None,
    bolp_module:  BOLPModule | None = None,
) -> SemanticSegmentationTask:

    decoder_type = decoder.lower()

    base_args = {
        "backbone":            "prithvi_eo_v2_300_tl",
        "backbone_pretrained": True,
        "backbone_ckpt_path":  str(SCRIPT_DIR.parents[3] / "models/Prithvi_EO_V2_300M_TL.pt"),
        "backbone_bands":      PRITHVI_BANDS,
        "backbone_num_frames": 1,
        "num_classes":         num_classes,
    }

    if decoder_type == "linear":
        decoder_args = {
            "decoder":               "LinearDecoder",
            "decoder_num_classes":   num_classes,
            "decoder_upsampling_size": 16,
            "decoder_in_index":      -1,
            "rescale":               False,
            "necks": [
                {"name": "SelectIndices", "indices": [23]},
                {"name": "ReshapeTokensToImage"},
            ],
        }
    elif decoder_type == "upernet":
        decoder_args = {
            "decoder":          "UperNetDecoder",
            "decoder_channels": 256,
            "head_dropout":     0.1,
            "rescale":          True,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
        }
    elif decoder_type == "unet":
        decoder_args = {
            "decoder":          "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "head_dropout":     0.1,
            "rescale":          True,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
        }
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    model_args = {**base_args, **decoder_args}

    common_kwargs = dict(
        model_args    = model_args,
        model_factory = "EncoderDecoderFactory",
        loss          = "ce",
        ignore_index  = -1,
        lr            = lr,
        optimizer     = "AdamW",
        optimizer_hparams = {"weight_decay": weight_decay},
        freeze_backbone   = True,
        freeze_decoder    = False,
        freeze_head       = False,
        plot_on_val       = False,
        class_names       = DYNAMIC_WORLD_CLASSES,
    )

    if bolp_module is not None:
        return BOLPSemanticSegmentationTask(
            **common_kwargs,
            bolp_module = bolp_module,
            tile_info   = tile_info or {},
        )
    elif reweighter is not None:
        return DBRSemanticSegmentationTask(
            **common_kwargs,
            reweighter   = reweighter,
            tile_info    = tile_info or {},
            use_dbr      = True,
        )
    else:
        return SemanticSegmentationTask(**common_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 14. DATAMODULE BUILDER  (unchanged from original file)
# ─────────────────────────────────────────────────────────────────────────────

def int_or_float(value: str) -> int | float:
    return float(value) if "." in value else int(value)


def build_datamodule(args: argparse.Namespace) -> MMEarthDynamicWorldSegmentationDataModule:
    return MMEarthDynamicWorldSegmentationDataModule(
        data_root       = args.data_root,
        sentinel2_bands = MMEARTH_S2_BANDS,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        image_size      = 224,
        normalize       = True,
        fallback_val_fraction  = args.val_fraction,
        fallback_test_fraction = args.test_fraction,
        seed            = args.seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 15. BIOME METRICS CSV  (post-training, mirrors original evaluate_biome_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_biome_metrics_csv(
    model:      SemanticSegmentationTask,
    datamodule: MMEarthDynamicWorldSegmentationDataModule,
    out_dir:    Path,
    tile_info:  dict,
    num_classes: int,
    device:     torch.device,
    bolp_module: BOLPModule | None = None,
    max_batches: int | None = None,
) -> None:
    """
    Compute per-macro-group mIoU on test set and save CSV + JSON.
    Mirrors the original evaluate_biome_metrics() but uses macro groups.
    """
    model.eval()
    biome_data: dict[str, dict] = defaultdict(lambda: {
        "samples": 0,
        "valid_pixels": 0,
        "correct_pixels": 0,
        "intersection": np.zeros(num_classes, dtype=np.int64),
        "union":        np.zeros(num_classes, dtype=np.int64),
    })

    with torch.no_grad():
        for batch_idx, batch in enumerate(datamodule.test_dataloader()):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images    = batch["image"].to(device)
            masks     = batch["mask"].squeeze(1).cpu()
            filenames = batch["filename"]

            if bolp_module is not None:
                logits = bolp_module.forward_with_bolp(
                    model, images,
                    tile_ids=list(filenames),
                    tile_info=tile_info,
                )
            else:
                logits = model.model(images).output

            predictions = logits.argmax(dim=1).detach().cpu()

            for i, tile_id in enumerate(filenames):
                group = tile_id_to_macro_group(str(tile_id), tile_info)
                if group is None:
                    continue
                target = masks[i]
                pred   = predictions[i]
                valid  = target != -1
                vp     = int(valid.sum().item())
                d = biome_data[group]
                d["samples"] += 1
                d["valid_pixels"] += vp
                if vp == 0:
                    continue
                d["correct_pixels"] += int((pred[valid] == target[valid]).sum().item())
                for c in range(num_classes):
                    pc = pred == c
                    tc = target == c
                    d["intersection"][c] += int((pc & tc & valid).sum().item())
                    d["union"][c]        += int(((pc | tc) & valid).sum().item())

    rows = []
    for group, d in sorted(biome_data.items()):
        intersections = d["intersection"]
        unions        = d["union"]
        present       = unions > 0
        per_class_iou = np.divide(
            intersections, unions,
            out  = np.full(num_classes, np.nan, dtype=np.float64),
            where= present,
        )
        miou     = float(np.nanmean(per_class_iou)) if np.any(present) else float("nan")
        micro    = float(intersections.sum() / unions.sum()) if int(unions.sum()) > 0 else float("nan")
        pix_acc  = float(d["correct_pixels"] / d["valid_pixels"]) if d["valid_pixels"] > 0 else float("nan")
        row = {
            "macro_group":    group,
            "samples":        int(d["samples"]),
            "valid_pixels":   int(d["valid_pixels"]),
            "pixel_accuracy": pix_acc,
            "mIoU":           miou,
            "mIoU_micro":     micro,
        }
        for c, cls_name in enumerate(DYNAMIC_WORLD_CLASSES):
            row[f"IoU_{cls_name}"] = float(per_class_iou[c]) if not np.isnan(per_class_iou[c]) else ""
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "test_biome_metrics.csv"
    json_path = out_dir / "test_biome_metrics.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nFinal test metrics by macro biome group:")
    print(f"{'group':<45}  {'samples':>7}  {'pix_acc':>9}  {'mIoU':>6}  {'mIoU_micro':>10}")
    for row in rows:
        print(
            f"{row['macro_group']:<45}  "
            f"{row['samples']:>7}  "
            f"{row['pixel_accuracy']:>9.4f}  "
            f"{row['mIoU']:>6.4f}  "
            f"{row['mIoU_micro']:>10.4f}"
        )
    print(f"Wrote biome metrics CSV : {csv_path}")
    print(f"Wrote biome metrics JSON: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 16. CHECKPOINT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint_weights(
    model:           SemanticSegmentationTask,
    checkpoint_path: str | Path,
    device:          torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="gpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 17. SCAN TRAINING SET FOR DBR BIOME GROUPS
# ─────────────────────────────────────────────────────────────────────────────

def collect_train_biome_groups(
    datamodule: MMEarthDynamicWorldSegmentationDataModule,
    tile_info:  dict,
) -> list[str]:
    """Scan training dataloader (no image loading overhead — just filenames)."""
    print("[DBR] Scanning training set for macro biome groups...")
    groups = []
    for batch in datamodule.train_dataloader():
        for tile_id in batch["filename"]:
            g = tile_id_to_macro_group(str(tile_id), tile_info)
            if g is not None:
                groups.append(g)
    print(f"[DBR] Found {len(groups)} samples with known biome group.")
    dist = Counter(groups)
    for g, n in sorted(dist.items()):
        print(f"  {g:<45}  n={n}")
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 18. ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMEarth Dynamic World segmentation — Normal LP / DBR / BOLP"
    )
    p.add_argument("--data-root",    default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-dir",      default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size",   type=int,         default=BATCH_SIZE)
    p.add_argument("--num-workers",  type=int,         default=NUM_WORKERS)
    p.add_argument("--max-epochs", "--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    p.add_argument("--lr",           type=float,       default=LR)
    p.add_argument("--weight-decay", type=float,       default=WEIGHT_DECAY)
    p.add_argument("--val-fraction", type=float,       default=VAL_FRACTION)
    p.add_argument("--test-fraction",type=float,       default=TEST_FRACTION)
    p.add_argument("--seed",         type=int,         default=SEED)
    p.add_argument("--resume-from",  type=str,         default=None, help="Path to checkpoint to resume")
    p.add_argument("--precision",    default=PRECISION)
    p.add_argument("--accelerator",  default=ACCELERATOR)
    p.add_argument("--devices",      default=DEVICES)
    p.add_argument("--decoder",      default=DECODER,
                   choices=["linear", "unet", "upernet"])
    p.add_argument("--limit-train-batches", type=int_or_float, default=1.0)
    p.add_argument("--limit-val-batches",   type=int_or_float, default=1.0)
    p.add_argument("--limit-test-batches",  type=int_or_float, default=1.0)
    p.add_argument("--console-log-every-n-steps", type=int, default=CONSOLE_LOG_EVERY_N_STEPS)
    p.add_argument("--fast-dev-run",         action="store_true")
    p.add_argument("--skip-test",            action="store_true")
    p.add_argument("--skip-biome-metrics",   action="store_true")
    p.add_argument("--limit-biome-metric-batches", type=int, default=None)
    p.add_argument("--disable-checkpointing",action="store_true")
    p.add_argument("--tensorboard",          action="store_true")
    p.add_argument("--test-only",            action="store_true")
    p.add_argument("--ckpt-path",            default=None)
    # DBR overrides
    p.add_argument("--dbr-alpha",   type=float, default=DBR_ALPHA)
    p.add_argument("--dbr-ema",     type=float, default=DBR_EMA)
    # BOLP overrides
    p.add_argument("--bolp-k",      type=int,   default=BOLP_K)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 19. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── validate CONFIG switches ──
    active_modes = sum([USE_NORMAL_LP, USE_DBR, USE_BOLP])
    
    args = parse_args()
    
    mode_tag = (
        "normal_lp" if USE_NORMAL_LP
        else f"dbr_alpha{DBR_ALPHA}_ema{DBR_EMA}" if USE_DBR
        else f"bolp_k{BOLP_K}"
    )
    out_dir = Path(args.out_dir) / mode_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "terminal_output.log"

    sys.stdout = TeeLogger(log_path, sys.stdout)
    sys.stderr = TeeLogger(log_path, sys.stderr)
    if active_modes != 1:
        raise SystemExit(
            f"Exactly ONE of USE_NORMAL_LP / USE_DBR / USE_BOLP must be True. "
            f"Currently {active_modes} are True."
        )

    mode_tag = (
        "normal_lp" if USE_NORMAL_LP
        else f"dbr_alpha{DBR_ALPHA}_ema{DBR_EMA}" if USE_DBR
        else f"bolp_k{BOLP_K}"
    )
    print(f"\n{'='*60}")
    print(f"  Mode : {mode_tag}")
    print(f"{'='*60}\n")

    args = parse_args()

    if not args.test_only and args.decoder in {"upernet", "unet"} and args.batch_size < 2:
        raise SystemExit(f"--decoder {args.decoder} requires --batch-size >= 2.")

    pl.seed_everything(args.seed)

    out_dir        = Path(args.out_dir) / mode_tag
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.accelerator in {"gpu", "auto"} and torch.cuda.is_available()
        else "cpu"
    )

    # ── tile info ──
    tile_info   = load_tile_info(args.data_root)
    num_classes = len(DYNAMIC_WORLD_CLASSES)

    # ── datamodule ──
    datamodule = build_datamodule(args)
    datamodule.setup("fit")
    datamodule.setup("test")

    val_loader  = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # ── logger ──
    logger = (
        TensorBoardLogger(save_dir=str(out_dir), name="logs")
        if args.tensorboard else False
    )

    # ── reweighter / BOLP (built before model so we can pass them in) ──
    reweighter  = None
    bolp_module = None

    if USE_DBR:
        train_groups = collect_train_biome_groups(datamodule, tile_info)
        reweighter   = DynamicBiomeReweighter(
            train_biome_groups = train_groups,
            alpha = args.dbr_alpha,
            ema   = args.dbr_ema,
        )

    if USE_BOLP:
        bolp_module = BOLPModule(k=args.bolp_k, device=device)

    # ── model ──
    model = build_model(
        decoder      = args.decoder,
        num_classes  = num_classes,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        reweighter   = reweighter,
        tile_info    = tile_info,
        bolp_module  = bolp_module,
    )

    # ── BOLP Phase A: fit projection matrix ──
    if USE_BOLP:
        model.to(device)
        bolp_module.fit(
            model      = model,
            dataloader = datamodule.train_dataloader(),
            tile_info  = tile_info,
        )

    # ── callbacks ──
    callbacks: list[Callback] = [
        CleanProgressBar(),
        ConsoleMetricsLogger(every_n_steps=args.console_log_every_n_steps)
    ]

    checkpoint_callback = None
    if not args.disable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor  = "val/mIoU",
            mode     = "max",
            dirpath  = str(checkpoint_dir),
            filename = f"{mode_tag}" + "-{epoch:02d}-{val_mIoU:.4f}",
            save_top_k = 1,
            save_last  = True,
        )
        callbacks.append(checkpoint_callback)

    if args.tensorboard:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    callbacks.append(
        EarlyStopping(monitor="val/loss", mode="min",
                      patience=EARLY_STOP_PATIENCE,
                      min_delta=EARLY_STOP_MIN_DELTA)
    )

    # per-epoch JSON metrics saver (also feeds val_biome_metrics to DBR)
    epoch_metrics_cb = EpochMetricsSaverCallback(
        val_loader   = val_loader,
        test_loader  = test_loader,
        tile_info    = tile_info,
        num_classes  = num_classes,
        out_dir      = out_dir,
        device       = device,
        ignore_index = -1,
        bolp_module  = bolp_module,
    )
    callbacks.append(epoch_metrics_cb)

    if USE_DBR and reweighter is not None:
        callbacks.append(
            DBRWeightUpdateCallback(
                reweighter   = reweighter,
                update_every = args.dbr_ema and DBR_UPDATE_EVERY or DBR_UPDATE_EVERY,
            )
        )

    # ── trainer ──
    trainer = pl.Trainer(
        accelerator          = args.accelerator,
        devices              = args.devices,
        precision            = args.precision,
        max_epochs           = args.max_epochs,
        logger               = logger,
        callbacks            = callbacks,
        check_val_every_n_epoch = 1,
        log_every_n_steps    = args.console_log_every_n_steps,
        enable_checkpointing = not args.disable_checkpointing,
        fast_dev_run         = args.fast_dev_run,
        limit_train_batches  = args.limit_train_batches,
        limit_val_batches    = args.limit_val_batches,
        limit_test_batches   = args.limit_test_batches,
    )

    # ── test-only path ──
    if args.test_only:
        if args.ckpt_path is None:
            raise SystemExit("--test-only requires --ckpt-path.")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        if not args.skip_biome_metrics:
            load_checkpoint_weights(model, args.ckpt_path, device)
            evaluate_biome_metrics_csv(
                model, datamodule, out_dir, tile_info, num_classes, device,
                bolp_module=bolp_module,
                max_batches=args.limit_biome_metric_batches,
            )
        return

    # ── training ──
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from)

    best_path = checkpoint_callback.best_model_path  if checkpoint_callback else ""
    last_path = checkpoint_callback.last_model_path  if checkpoint_callback else ""
    if best_path:
        print(f"Best checkpoint : {best_path}")
    if last_path:
        print(f"Last checkpoint : {last_path}")

    # ── final test ──
    if not args.skip_test:
        test_ckpt = best_path or last_path or None
        print(f"Testing checkpoint: {test_ckpt or 'current weights'}")
        trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt)

        if test_ckpt:
            strip_encoder_from_checkpoint(Path(test_ckpt))

        if test_ckpt:
            load_checkpoint_weights(model, test_ckpt, device)

        if not args.skip_biome_metrics:
            evaluate_biome_metrics_csv(
                model, datamodule, out_dir, tile_info, num_classes, device,
                bolp_module=bolp_module,
                max_batches=args.limit_biome_metric_batches,
            )


if __name__ == "__main__":
    # ── Usage examples ───────────────────────────────────────────────────────
    #
    # Normal LP — MMEarth20K
    #
    # Set:
    #   USE_NORMAL_LP = True
    #   USE_DBR       = False
    #   USE_BOLP      = False
    
    # python train_mmearth_dynamic_world_prithvi_super.py \
    #     --data-root "/data1/aminur/om/MMearth20k/hardcoded" \
    #     --out-dir "/data1/aminur/om/MMearth20k/output" \
    #     --batch-size 16 \
    #     --num-workers 4 \
    #     --max-epochs 100 \
    #     --accelerator gpu \
    #     --devices 1 \
    #     --precision 16-mixed
    #
    # ------------------------------------------------------------------------
    # DBR — MMEarth20K
    #
    # Set:
    #   USE_NORMAL_LP = False
    #   USE_DBR       = True
    #   USE_BOLP      = False
    #
    # python train_mmearth_dynamic_world_prithvi_super.py \
    #     --data-root "/data1/aminur/om/MMearth20k/hardcoded" \
    #     --out-dir "/data1/aminur/om/MMearth20k/output" \
    #     --batch-size 1 \
    #     --num-workers 8 \
    #     --max-epochs 100 \
    #     --accelerator gpu \
    #     --devices 1 \
    #     --precision 16-mixed \
    #     --dbr-alpha 0.5 \
    #     --dbr-ema 0.8
    #
    # ------------------------------------------------------------------------
    # BOLP — MMEarth20K
    #
    # Set:
    #   USE_NORMAL_LP = False
    #   USE_DBR       = False
    #   USE_BOLP      = True
    #
    # python train_mmearth_dynamic_world_prithvi_super.py \
    #     --data-root "/data1/aminur/om/MMearth20k/hardcoded" \
    #     --out-dir "/data1/aminur/om/MMearth20k/output" \
    #     --batch-size 102 \
    #     --num-workers 6 \
    #     --max-epochs 100 \
    #     --accelerator gpu \
    #     --devices 1 \
    #     --precision 16-mixed \
    #     --bolp-k 5
    #
    # ------------------------------------------------------------------------
    # Smoke test
    #
    # python train_mmearth_dynamic_world_prithvi_super.py \
    #     --data-root "/data1/aminur/om/MMearth20k/hardcoded" \
    #     --out-dir "/data1/aminur/om/MMearth20k/output" \
    #     --accelerator gpu \
    #     --devices 1 \
    #     --precision 16-mixed \
    #     --limit-train-batches 2 \
    #     --limit-val-batches 1 \
    #     --limit-test-batches 1 \
    #     --disable-checkpointing \
    #     --fast-dev-run

    main()