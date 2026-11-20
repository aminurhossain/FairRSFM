#!/usr/bin/env python3
"""Prithvi EO V2 300M — EuroSAT Classification (DBR Method)

Dynamic Biome Reweighting uses unweighted epoch group training losses,
EMA beta=0.9, alpha=0.5, epsilon=1e-6, and mean-one weights.
Initialize inverse group frequencies on the common known-biome training
subset. Backbones stay frozen; validation and test run every epoch and
do not update DBR. See docs/methods.md and fairrsfm.training.protocol.
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
from fairrsfm.training.reporting import report_epoch, evaluating, macro_f1_present, write_json, finalize_selected, record_config
from fairrsfm.evaluation.serialization import json_safe
sys.path.insert(0, str(BASE_DIR / "src"))
from fairrsfm.evaluation.metrics import single_label_group_diagnostics

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
        y      = torch.tensor(int(sample.label), dtype=torch.long)
        sid    = str(sample.sample_name)
        return x, y, sid


# ─────────────────────────────────────────────────────────────────────────────
# 5. DBR REWEIGHTER MODULE
# ─────────────────────────────────────────────────────────────────────────────

from fairrsfm.methods.dbr import DynamicBiomeReweighter




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

class ClassifLinearProbe(MethodTrainingMixin, L.LightningModule):
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
            global_f1 = float(macro_f1_present(y_true, y_pred, labels=global_present, average="macro", zero_division=0))
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
        return report_epoch(self, biome_data, epoch, tb, tb_prefix, label, pl_module, task="eurosat", ignore_index=-1)

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
                    probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                    preds  = probs.argmax(axis=-1)
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

        probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)
        labels = y.cpu().numpy()

        for i, sid in enumerate(sample_ids):
            group = self.biome_map.get(sid, "Unknown")
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
    global BATCH_SIZE
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Prithvi EuroSAT Classification DBR Training")
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

    dbr_tag  = f"dbr_alpha{DBR_ALPHA}_ema{DBR_EMA}" if USE_DBR else "erm"
    mode_str = f"eurosat_lp_{dbr_tag}"
    OUT_DIR  = Path(args.out_dir) if args.out_dir else BASE_DIR / f"runs/{mode_str}/seed_{seed}"
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
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchClassifDataset(task, "valid", biome_map, augment=False)
    test_ds  = GeoBenchClassifDataset(task, "test",  biome_map, augment=False)

    train_loader = training_loader(train_ds, method='dbr', groups=protocol_groups, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    base_model = build_model(WEIGHTS_PT, NUM_CLASSES)

    reweighter = None

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
    configure_training_protocol(lp_model, 'dbr', 'eurosat', protocol_groups, args, ignore_index=-1, lookup=lambda sid: lookup_biome(biome_map, sid))
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
