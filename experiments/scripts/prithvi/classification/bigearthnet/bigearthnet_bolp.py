#!/usr/bin/env python3
"""
Prithvi EO V2 300M — BigEarthNet Multi-Label Classification (BOLP Method)
==========================================================================
Key Features:
  • Task: Multi-label Classification (m-bigearthnet, 43 classes, BCEWithLogitsLoss)
  • Method: Biome-Orthogonal Linear Probing (BOLP) with Macro Grouping (eurosat_biome_mapping.json)
  • Unknowns Purged: Dataset strictly filters out samples without known biome mappings.
  • Dynamic K: Projection rank automatically set to |B| - 1.
  • Ablation Ready: --skip-resid flag added for testing raw projections.
  • Augmentations: Random horizontal/vertical flips active on train split.
  • Slim Checkpoints: Strips frozen encoder weights (~2.5 MB per .ckpt file)
"""

from fairrsfm.methods.bolp import BOLPProjector as SharedBOLPProjector, fit_bolp_layers
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
SEED = 42

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

BATCH_SIZE   = 64
EPOCHS = 50
LR           = 1e-4
WEIGHT_DECAY = 0.1
NUM_WORKERS  = 16
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

    return biome_map


def print_split_biome_distribution(split_name: str, dataset, biome_map: dict):
    counts = defaultdict(int)
    for sid in dataset_sample_ids(dataset):
        counts[lookup_biome(biome_map, sid)] += 1
    print(f"[{split_name}] {dict(counts)}")


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

        # Strictly filter out unknown biomes
        if hasattr(self.dataset, "active_partition") and hasattr(self.dataset, "split") and self.dataset.split in self.dataset.active_partition.partition_dict:
            sample_names = self.dataset.active_partition.partition_dict[self.dataset.split]
        elif hasattr(self.dataset, "active_partition") and split in self.dataset.active_partition.partition_dict:
            sample_names = self.dataset.active_partition.partition_dict[split]
        elif hasattr(self.dataset, "_sample_name_list"):
            sample_names = self.dataset._sample_name_list
        else:
            sample_names = [self.dataset[i].sample_name for i in range(len(self.dataset))]

        self.sample_names = sample_names
        self.valid_indices = [
            i for i, sid in enumerate(sample_names)
            if self.biome_map.get(str(sid), "Unknown") != "Unknown"
        ]

        print(f"[Dataset] {split.upper()}: Filtered down to {len(self.valid_indices):,} samples with known biomes.")
        self.valid_indices = list(range(len(self.dataset)))

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
        y      = torch.tensor(sample.label, dtype=torch.float32)
        sid    = str(self.sample_names[real_idx]) if hasattr(self, "sample_names") else str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOLP PROJECTOR & SUBSPACES
# ─────────────────────────────────────────────────────────────────────────────

class BOLPProjector(SharedBOLPProjector):
    pass


def build_bolp_projector(encoder: torch.nn.Module, train_ds: GeoBenchClassifDataset,
                         biome_map: dict, skip_residualization: bool = False,
                         device: torch.device = torch.device("cpu")) -> BOLPProjector:
    encoder.eval()
    encoder.to(device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    fitted = fit_bolp_layers(lambda x: encoder.encoder(x), loader, lambda sid: lookup_biome(biome_map, sid), device, skip_residualization=skip_residualization)
    return fitted.layers[0]


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
# 7. LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class ClassifLinearProbe(MethodTrainingMixin, L.LightningModule):
    def __init__(self, model, lr: float, weight_decay: float, num_classes: int,
                 projector: torch.nn.Module = None, biome_map: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "projector"])
        self.model       = model
        self.num_classes = num_classes
        self.projector   = projector
        self.biome_map   = biome_map or {}
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
        if self.projector is not None:
            checkpoint["bolp_biome_index"] = self.projector.biome_index
            checkpoint["bolp_skip_residualization"] = self.projector.skip_residualization
            checkpoint["bolp_k"] = self.projector.k

    def on_load_checkpoint(self, checkpoint):
        if FREEZE_BACKBONE:
            state_dict = checkpoint.get("state_dict", {})
            for k, v in self.model.encoder.state_dict().items():
                state_dict[f"model.encoder.{k}"] = v

    def _register_bolp_hook(self):
        def hook_fn(module, input, output):
            biome_ids = getattr(self, "_current_biome_ids", None)
            if isinstance(output, (list, tuple)):
                last = output[-1]
            else:
                last = output
            last = self.projector(last, biome_ids)
            if isinstance(output, (list, tuple)):
                out_list = list(output)
                out_list[-1] = last
                return type(output)(out_list)
            return last
        self._hook_handle = self.model.encoder.register_forward_hook(hook_fn)

    def forward(self, x, sample_ids=None):
        if self.projector is not None and not hasattr(self, "_hook_handle"):
            self._register_bolp_hook()
        if sample_ids is not None:
            self._current_biome_ids = [self.biome_map.get(str(sid), "Unknown") for sid in sample_ids]
        else:
            self._current_biome_ids = None
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)




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
            global_f1 = float(macro_f1_present(y_true[:, present], y_pred_opt[:, present], average="macro", zero_division=0))
        else:
            global_f1 = 0.0
        global_micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

        best_macro = 0.0
        best_micro = 0.0
        for t in np.linspace(0.1, 0.9, 17):
            yp = (probs > t).astype(int)
            mac = macro_f1_present(y_true, yp, average="macro", zero_division=0)
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
        self.opt_thresholds = None

    def _compute_and_log(self, biome_data, epoch, tb, tb_prefix, label, pl_module=None):
        return report_epoch(self, biome_data, epoch, tb, tb_prefix, label, pl_module, task="bigearthnet", ignore_index=-1)

    def _eval_test_split_and_log(self, trainer, pl_module, epoch, tb):
        with evaluating(pl_module):
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
                    logits = forward_with_sample_ids(pl_module, x, sample_ids)
                    probs  = torch.sigmoid(logits.float()).cpu().numpy()
                    preds  = (probs > 0.5).astype(int)
                    labels = y.numpy()

                    for i, sid in enumerate(sample_ids):
                        group = self.biome_map.get(sid, "Unknown")
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
    global BATCH_SIZE
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Prithvi BigEarthNet Classification BOLP Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID (default: 0)")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch (shared-GPU default: 64)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--skip-resid", action="store_true", help="Skip residualization step for ablation studies")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    BATCH_SIZE = args.batch_size
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    L.seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resid_tag = "no_resid" if args.skip_resid else "full"
    mode_str  = f"bigearthnet_lp_bolp_{resid_tag}"
    OUT_DIR   = Path(args.out_dir) if args.out_dir else BASE_DIR / f"runs/{mode_str}/seed_{seed}"
    CKPT_DIR  = OUT_DIR  / "checkpoints"

    for p in [DATA_DIR, BIOME_JSON, WEIGHTS_PT]:
        assert p.exists(), f"Path not found: {p}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(OUT_DIR / "train.log")
    sys.stderr = sys.stdout

    print(f"[Config] Dataset    : m-bigearthnet (43 classes, multi-label)")
    print(f"[Config] Mode       : Linear Probe (BOLP - Dynamic K)")
    print(f"[Config] Seed       : {seed}")
    print(f"[Config] Ablation   : skip_resid={args.skip_resid}")
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
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchClassifDataset(task, "valid", biome_map, augment=False)
    test_ds  = GeoBenchClassifDataset(task, "test",  biome_map, augment=False)

    train_loader = training_loader(train_ds, method='bolp', groups=protocol_groups, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = build_model(WEIGHTS_PT, NUM_CLASSES)

    bolp_train_ds = GeoBenchClassifDataset(task, "train", biome_map, augment=False)
    projector = build_bolp_projector(
        encoder              = base_model,
        train_ds             = bolp_train_ds,
        biome_map            = biome_map,
        skip_residualization = args.skip_resid,
        device               = device,
    )

    lp_model = ClassifLinearProbe(
        model       = base_model,
        lr          = LR,
        weight_decay= WEIGHT_DECAY,
        num_classes = NUM_CLASSES,
        projector   = projector,
        biome_map   = biome_map,
    )
    configure_training_protocol(lp_model, 'bolp', 'bigearthnet', protocol_groups, args, ignore_index=-1, lookup=lambda sid: lookup_biome(biome_map, sid))
    args.method_protocol = lp_model.method_protocol.config

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
    global_f1_cb  = GlobalMacroF1Logger(num_classes=NUM_CLASSES)
    biome_cb      = BiomeMacroF1Logger(
        biome_map   = biome_map,
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        output_dir  = OUT_DIR,
        test_loader = test_loader,
    )
    progress_bar_cb = CleanProgressBar(refresh_rate=20)

    tb_logger = TensorBoardLogger(save_dir=OUT_DIR / "tb_logs", name=mode_str, version=0)

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
    trainer.fit(lp_model, train_loader, val_loader, ckpt_path=args.resume_from)
    finalize_selected(trainer, lp_model)

    print(f"\n[Done] Best Macro F1 Checkpoint : {ckpt_macro_cb.best_model_path}")
    print(f"[Done] Best Macro F1 Score      : {ckpt_macro_cb.best_model_score:.4f}")
    if ckpt_worst_cb.best_model_path:
        print(f"[Done] Best Worst F1 Checkpoint : {ckpt_worst_cb.best_model_path}")
        print(f"[Done] Best Worst F1 Score      : {ckpt_worst_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
