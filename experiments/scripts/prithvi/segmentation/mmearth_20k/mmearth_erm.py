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

Validation metrics are saved each epoch to <out_dir>/val_results.json.
Selected-checkpoint test metrics are saved to <out_dir>/test_selected.json.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 0. SYSTEM SETUP
# ─────────────────────────────────────────────────────────────────────────────
from fairrsfm.methods.bolp import PrithviFinalLayerBOLP
from fairrsfm.training.protocol import (MethodTrainingMixin, known_training_dataset, lookup_biome, training_loader, configure_training_protocol, dataset_sample_ids, balanced_group_loader, forward_with_sample_ids, update_projection_protocol)
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
os.environ["PROJ_DATA"] = "/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_LIB"]  = "/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"

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
from types import SimpleNamespace
from fairrsfm.training.reporting import report_epoch, write_json, evaluating, record_config
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

class ProtocolSemanticSegmentationTask(MethodTrainingMixin, SemanticSegmentationTask):
    pass


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
_data_cands = [
    Path(os.environ.get("MMEARTH_DATA_DIR", "")),
    Path("/data1/aminur/om/MMearth20k/hardcoded"),
    SCRIPT_DIR.parents[4] / "methods/prithvi/segmentation/mmearth_20k/hardcoded",
    SCRIPT_DIR.parents[4] / "data/mmearth",
]
DEFAULT_DATA_ROOT = "/data1/aminur/om/MMearth20k/hardcoded"
for _dc in _data_cands:
    if _dc.is_dir():
        DEFAULT_DATA_ROOT = str(_dc)
        break
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "output")

# ── Bands ──
PRITHVI_BANDS   = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
MMEARTH_S2_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]

# ── Hyperparams ──
BATCH_SIZE   = 32
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
DBR_EMA            = 0.9   # metric EMA coefficient
DBR_UPDATE_EVERY   = 1     # update weights every N epochs

# ── BOLP config ──
BOLP_K = None   # number of top singular vectors to project out (rank of null-space)

# ── Early stopping ──
EARLY_STOP_PATIENCE  = 1000
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
    std   = float((np.std(valid, ddof=1) if len(valid) > 1 else 0.0))
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
    epoch: int = 0,
    label: str = "val",
    bootstrap_replicates: int = 0,
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
                logits = bolp_module.forward_with_bolp(
                    model, images, tile_ids=list(filenames), tile_info=tile_info
                )
            else:
                logits = model.model(images).output

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i, tile_id in enumerate(filenames):
                group = tile_id_to_macro_group(str(tile_id), tile_info) or "Unknown"
                flat_true = masks_np[i].reshape(-1)
                flat_pred = preds[i].reshape(-1)
                all_true.append(flat_true)
                all_pred.append(flat_pred)
                if group is not None:
                    if group not in biome_data:
                        biome_data[group] = {"true": [], "pred": []}
                    biome_data[group]["true"].append(flat_true)
                    biome_data[group]["pred"].append(flat_pred)

    if not biome_data:
        raise ValueError("No observations were collected for this split")
    callback = SimpleNamespace(num_classes=num_classes,
                               bootstrap_replicates=bootstrap_replicates)
    record = report_epoch(callback, biome_data, epoch, None, "metrics", label,
                          task="mmearth", ignore_index=ignore_index, persist=False)
    record.update(macro_f1=record["overall_macro_f1"],
                  biome_wise_miou_table=record["per_biome_miou"],
                  worst_biome_miou=record["worst_case_miou"],
                  biome_std=record["group_std"], per=record["performance_equity_ratio"],
                  gini=record["gini_coefficient"], eqodds=record["equalized_odds_disparity"],
                  dpm=record["demographic_parity_disparity"])
    return record


# ─────────────────────────────────────────────────────────────────────────────
# 6. JSON METRICS SAVER
# ─────────────────────────────────────────────────────────────────────────────

def save_epoch_metrics(
    out_dir: Path,
    epoch: int,
    val_metrics: dict,
) -> None:
    path = out_dir / "val_results.json"
    history = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(history, list):
        history = [row.get("val", row) for row in history.values()]
    history = [r for r in history if r.get("epoch") != epoch]
    history.append({**val_metrics, "epoch": int(epoch), "split": "val"})
    write_json(path, history)
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
# 8. PER-EPOCH VALIDATION METRICS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class EpochMetricsSaverCallback(Callback):
    """
    After every validation epoch (skipping sanity check):
      1. Compute val metrics over val_loader
      2. Append validation metrics to val_results.json under key epoch_N
    Also stores val biome mious on pl_module for DBR to consume.
    """

    def __init__(
        self,
        val_loader:   DataLoader,
        tile_info:    dict,
        num_classes:  int,
        out_dir:      Path,
        device:       torch.device,
        test_loader: DataLoader,
        ignore_index: int = -1,
        bolp_module:  "BOLPModule | None" = None,
    ) -> None:
        super().__init__()
        self.test_loader = test_loader
        self.val_loader   = val_loader
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
        # expose val biome mious to pl_module so DBR callback can read them
        pl_module.val_biome_metrics = val_metrics.get("biome_wise_miou_table", {})
        # Select checkpoints with the same present-class mIoU saved in val_results.
        pl_module.log("val/benchmark_miou", val_metrics["global_miou"],
                      on_step=False, on_epoch=True, prog_bar=True,
                      add_dataloader_idx=False)

        save_epoch_metrics(self.out_dir, epoch, val_metrics)
        with evaluating(pl_module):
            test_metrics = evaluate_split(
                pl_module, self.test_loader, self.tile_info, self.num_classes,
                self.device, self.ignore_index, self.bolp_module, epoch=epoch, label="test",
            )
        path = self.out_dir / "test_results.json"
        history = json.loads(path.read_text()) if path.exists() else []
        history = [r for r in history if r.get("epoch") != epoch]
        history.append({**test_metrics, "epoch": int(epoch), "split": "test"})
        write_json(path, history)

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
            f"══════════════════════════════════════════════════════════════════════════════════\\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. DBR  (Dynamic Biome Reweighting with Validation Feedback)
# ─────────────────────────────────────────────────────────────────────────────

from fairrsfm.methods.dbr import DynamicBiomeReweighter




# ─────────────────────────────────────────────────────────────────────────────
# 10. DBR-AWARE SemanticSegmentationTask WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class DBRSemanticSegmentationTask(MethodTrainingMixin, SemanticSegmentationTask):
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



# ─────────────────────────────────────────────────────────────────────────────
# 11. BOLP MODULE
# ─────────────────────────────────────────────────────────────────────────────

class BOLPModule(PrithviFinalLayerBOLP):
    def __init__(self, k=None, device=torch.device("cpu")):
        super().__init__(k=k, device=device, tile_lookup=tile_id_to_macro_group)


# ─────────────────────────────────────────────────────────────────────────────
# 12. BOLP TRAINING STEP WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class BOLPSemanticSegmentationTask(MethodTrainingMixin, SemanticSegmentationTask):
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

    _ckpt_cands = [
        SCRIPT_DIR.parents[4] / "checkpoints/Prithvi_EO_V2_300M_TL.pt",
        SCRIPT_DIR.parents[4] / "models/Prithvi_EO_V2_300M_TL.pt",
        Path("/data1/aminur/om/prithvi/FairRSFM_unified/checkpoints/Prithvi_EO_V2_300M_TL.pt"),
        Path("/data1/aminur/om/prithvi/FairRSFM/checkpoints/Prithvi_EO_V2_300M_TL.pt"),
    ]
    _ckpt_path = None
    for _c in _ckpt_cands:
        if _c.exists():
            _ckpt_path = str(_c)
            break
    if _ckpt_path is None:
        _ckpt_path = str(SCRIPT_DIR.parents[4] / "checkpoints/Prithvi_EO_V2_300M_TL.pt")

    base_args = {
        "backbone":            "prithvi_eo_v2_300_tl",
        "backbone_pretrained": True,
        "backbone_ckpt_path":  _ckpt_path,
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
        return ProtocolSemanticSegmentationTask(**common_kwargs)


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
        train_samples   = args.train_samples,
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
    write_json(json_path, rows)

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
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)


def strip_encoder_from_checkpoint(ckpt_path: Path | str | None) -> None:
    if not ckpt_path:
        return
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            keys = list(checkpoint["state_dict"].keys())
            stripped = False
            for k in keys:
                if k.startswith("model.encoder.") or k.startswith("model.backbone."):
                    del checkpoint["state_dict"][k]
                    stripped = True
            if stripped:
                torch.save(checkpoint, ckpt_path)
    except Exception as e:
        print(f"[Warning] strip_encoder_from_checkpoint failed: {e}")


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
    p.add_argument("--train-samples",        type=int,   default=None, help="Limit number of training samples for smoke test")
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
    datamodule.train_dataset, protocol_groups = known_training_dataset(datamodule.train_dataset, lambda sid: tile_id_to_macro_group(sid, tile_info))
    datamodule.known_training_ids = tuple(protocol_groups)
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
    configure_training_protocol(model, 'erm', 'mmearth', protocol_groups, args, ignore_index=-1, lookup=lambda sid: tile_id_to_macro_group(sid, tile_info))
    args.method_protocol = model.method_protocol.config
    record_config(out_dir, __file__, args)
    model.strict_loading = False

    # ── BOLP Phase A: fit projection matrix ──
    if USE_BOLP:
        model.to(device)
        bolp_module.fit(
            model      = model,
            dataloader = datamodule.train_dataloader(),
            tile_info  = tile_info,
        )
        update_projection_protocol(model, args)
        record_config(out_dir, __file__, args)

    # ── callbacks ──
    callbacks: list[Callback] = [
        CleanProgressBar(),
        ConsoleMetricsLogger(every_n_steps=args.console_log_every_n_steps)
    ]

    checkpoint_callback = None
    if not args.disable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor  = "val/benchmark_miou",
            mode     = "max",
            dirpath  = str(checkpoint_dir),
            filename = f"{mode_tag}" + "-{epoch:02d}-{val/benchmark_miou:.4f}",
            auto_insert_metric_name = False,
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
            load_checkpoint_weights(model, test_ckpt, device)

        selected_test_metrics = evaluate_split(
            model, test_loader, tile_info, num_classes, device,
            ignore_index=-1, bolp_module=bolp_module,
            epoch=int(torch.load(test_ckpt, map_location="cpu", weights_only=False).get("epoch", -1)) if test_ckpt else -1,
            label="test", bootstrap_replicates=1000,
        )
        write_json(out_dir / "test_selected.json", {
            "checkpoint": str(test_ckpt) if test_ckpt else None,
            "checkpoint_epoch": selected_test_metrics["epoch"],
            "selection": "validation checkpoint",
            "metrics": selected_test_metrics,
        })

        if not args.skip_biome_metrics:
            evaluate_biome_metrics_csv(
                model, datamodule, out_dir, tile_info, num_classes, device,
                bolp_module=bolp_module,
                max_batches=args.limit_biome_metric_batches,
            )

        if test_ckpt:
            strip_encoder_from_checkpoint(test_ckpt)
        if last_path and last_path != test_ckpt:
            strip_encoder_from_checkpoint(last_path)


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
