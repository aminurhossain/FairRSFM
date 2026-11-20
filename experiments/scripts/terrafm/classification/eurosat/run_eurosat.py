#!/usr/bin/env python3
"""
TerraFM ViT-Base — EuroSAT Single-Label Multi-Class Classification (m-eurosat)
==============================================================================
Universal FairRSFM runner for TerraFM on m-eurosat (10 classes):
Supports methods: ERM, Group DRO, DBR, BOLP.

Protocol Guarantees:
  • Frozen ViT-Base Backbone + Linear Probe Head (embed_dim -> 10)
  • 12 Sentinel-2 Optical Channels (indices [0..9, 11, 12] from 13-band S2)
  • Full 50 Epochs Training (NO early stopping)
  • Full Macro-Biome Validation & Test evaluation saved every epoch from Epoch 0
  • Outputs: val_results.json, test_results.json, test_selected.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Path resolution
ROOT_DIR = Path(__file__).resolve()
while ROOT_DIR.parent != ROOT_DIR:
    if (ROOT_DIR / "pyproject.toml").is_file() or (ROOT_DIR / "src/fairrsfm").is_dir():
        break
    ROOT_DIR = ROOT_DIR.parent

for p in [ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "src/fairrsfm"]:
    if str(p) not in sys.path and p.is_dir():
        sys.path.insert(0, str(p))

os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

import pyproj.datadir
proj_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"]  = proj_dir

import numpy as np
import albumentations
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.metrics import f1_score, accuracy_score, average_precision_score

import geobench
import geobench.task
from fairrsfm.models.terrafm_wrapper import TerraFMClassificationClassifier
from fairrsfm.training.protocol import (
    MethodTrainingMixin,
    known_training_dataset,
    lookup_biome,
    training_loader,
    configure_training_protocol,
    dataset_sample_ids,
)

# Constants & Hyperparameters
TASK_KEYWORD = "eurosat"
NUM_CLASSES  = 10
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 0.01
SEED         = 42
NUM_WORKERS  = 4

DATA_DIR = Path("/data1/aminur/dataset/geobench")
BIOME_JSON = ROOT_DIR / "configs/data_mappings/eurosat_biome_mapping.json"
DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints/terrafm/TerraFM-B.pth"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Final_eurosat/terrafm"

# 13 bands in m-eurosat: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
# TerraFM uses 12 optical bands (dropping index 10: Cirrus)
TERRAFM_EUROSAT_BAND_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]


def load_biome_map(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"[BiomeMap] WARNING: {json_path} not found.")
        return {}
    with open(json_path) as f:
        records = json.load(f)
    biome_map = {}
    for r in records:
        sid = str(r.get("geobench_id", ""))
        bname = r.get("biome_name", "Unknown")
        if sid:
            biome_map[sid] = bname
    unique_groups = sorted(set(biome_map.values()) - {"Unknown"})
    print(f"[BiomeMap] Loaded {len(biome_map):,} entries | {len(unique_groups)} biomes: {unique_groups}")
    return biome_map


class GeoBenchEuroSATDataset(Dataset):
    def __init__(self, task, split: str, augment: bool = False):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = TERRAFM_EUROSAT_BAND_INDICES
        self.augment      = augment

        self._spatial_aug = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Resize(IMG_SIZE, IMG_SIZE),
        ])
        self._spatial_val = albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        array, _ = sample.pack_to_3d(
            band_names=self.band_names,
            resample=False,
            fill_value=None,
            resample_order=3,
        )
        image = array.astype(np.float32)[:, :, self.band_indices]
        tfm = self._spatial_aug if self.augment else self._spatial_val
        out = tfm(image=image)
        image = out["image"]

        x = torch.from_numpy(image).permute(2, 0, 1).float()
        y = int(sample.label)
        sid = str(sample.sample_name)
        return x, y, sid


def compute_gini(values: list) -> float:
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


class EuroSATBiomeLogger(Callback):
    def __init__(self, biome_map: dict, num_classes: int, class_names: list, output_dir: Path, test_loader=None):
        super().__init__()
        self.biome_map    = biome_map
        self.num_classes  = num_classes
        self.class_names  = class_names
        self.output_dir   = Path(output_dir)
        self.test_loader  = test_loader
        self.val_history  = []
        self.test_history = []
        self._val_data    = None

    def _compute_and_log(self, data_dict: dict, epoch: int, label: str, pl_module=None):
        known_biome_data = {b: data_dict[b] for b in data_dict if b != "Unknown" and len(data_dict[b]["true"]) > 0}
        all_biomes = sorted(data_dict.keys())
        all_true = np.concatenate([data_dict[b]["true"] for b in all_biomes if len(data_dict[b]["true"]) > 0])
        all_pred = np.concatenate([data_dict[b]["pred"] for b in all_biomes if len(data_dict[b]["pred"]) > 0])
        all_probs = np.concatenate([data_dict[b]["probs"] for b in all_biomes if len(data_dict[b]["probs"]) > 0])

        overall_f1  = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
        overall_acc = float(accuracy_score(all_true, all_pred))

        print(f"\n{'='*82}")
        print(f"  [{label.upper()} EVALUATION — Epoch {epoch:02d}] Overall Macro F1: {overall_f1*100:.2f}% | Top-1 Acc: {overall_acc*100:.2f}%")
        print(f"{'-'*82}")

        group_f1s, group_accs, group_counts = {}, {}, {}
        for b in sorted(known_biome_data.keys()):
            yt = np.array(known_biome_data[b]["true"])
            yp = np.array(known_biome_data[b]["pred"])
            group_counts[b] = len(known_biome_data[b]["sample_ids"])
            group_f1s[b]    = float(f1_score(yt, yp, average="macro", zero_division=0))
            group_accs[b]   = float(accuracy_score(yt, yp)) if len(yt) > 0 else float("nan")
            print(f"    • {b:<45} : F1={group_f1s[b]*100:.2f}% | Acc={group_accs[b]*100:.2f}% ({group_counts[b]} samples)")

        valid_f1s = [v for v in group_f1s.values() if v is not None and not np.isnan(v)]
        if len(valid_f1s) >= 2:
            worst_f1  = float(np.min(valid_f1s))
            best_f1   = float(np.max(valid_f1s))
            mean_f1   = float(np.mean(valid_f1s))
            std_f1    = float(np.std(valid_f1s, ddof=1))
            per_ratio = float(worst_f1 / best_f1) if best_f1 > 0 else float("nan")
            gini_coef = compute_gini(valid_f1s)
            nfr       = float((best_f1 - worst_f1) / mean_f1) if mean_f1 > 0 else float("nan")
        else:
            worst_f1 = best_f1 = mean_f1 = std_f1 = per_ratio = gini_coef = nfr = float("nan")

        print(f"  ── Fairness Summary: Worst={worst_f1*100:.2f}% | Best={best_f1*100:.2f}% | PER={per_ratio:.4f} | Std={std_f1*100:.2f}% | Gini={gini_coef:.4f}")
        print(f"{'='*82}\n")

        if pl_module is not None and label == "val":
            if not np.isnan(worst_f1):
                pl_module.log("val/worst_case_macro_f1", worst_f1, prog_bar=True)
            if not np.isnan(overall_f1):
                pl_module.log("val/macro_f1", overall_f1, prog_bar=True)

        record = {
            "epoch": epoch,
            "overall_macro_f1": overall_f1,
            "overall_accuracy": overall_acc,
            "worst_case_macro_f1": worst_f1 if not np.isnan(worst_f1) else None,
            "best_case_macro_f1": best_f1 if not np.isnan(best_f1) else None,
            "group_mean_macro_f1": mean_f1 if not np.isnan(mean_f1) else None,
            "group_std_macro_f1": std_f1 if not np.isnan(std_f1) else None,
            "per_ratio": per_ratio if not np.isnan(per_ratio) else None,
            "normalized_fairness_range": nfr if not np.isnan(nfr) else None,
            "gini_coefficient": gini_coef if not np.isnan(gini_coef) else None,
            "per_group_metrics": {
                g: {
                    "macro_f1": group_f1s[g],
                    "accuracy": group_accs[g],
                    "sample_count": group_counts[g],
                }
                for g in group_f1s
            }
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if label == "val":
            self.val_history.append(record)
            with open(self.output_dir / "val_results.json", "w") as f:
                json.dump(self.val_history, f, indent=2)
        elif label == "test":
            self.test_history.append(record)
            with open(self.output_dir / "test_results.json", "w") as f:
                json.dump(self.test_history, f, indent=2)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_data = defaultdict(lambda: {"true": [], "pred": [], "probs": [], "sample_ids": []})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0 or outputs is None:
            return
        logits = outputs["logits"]
        y      = outputs["y"]
        sids   = outputs["sample_ids"]

        probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)
        labels = y.cpu().numpy()

        for sid, label_val, pred_val, prob_arr in zip(sids, labels, preds, probs):
            biome = self.biome_map.get(str(sid), "Unknown")
            self._val_data[biome]["true"].append(label_val)
            self._val_data[biome]["pred"].append(pred_val)
            self._val_data[biome]["probs"].append(prob_arr)
            self._val_data[biome]["sample_ids"].append(sid)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._val_data:
            return
        epoch = trainer.current_epoch
        self._compute_and_log(self._val_data, epoch, "val", pl_module=pl_module)

        if self.test_loader is not None:
            self._run_test_evaluation(trainer, pl_module, epoch)
        self._val_data = None

    def _run_test_evaluation(self, trainer, pl_module, epoch):
        was_training = pl_module.training
        pl_module.eval()
        test_data = defaultdict(lambda: {"true": [], "pred": [], "probs": [], "sample_ids": []})
        device = pl_module.device

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for batch in self.test_loader:
                    x, y, sids = batch
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    logits = pl_module(x)
                    probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                    preds  = probs.argmax(axis=-1)
                    labels = y.cpu().numpy()

                    for sid, label_val, pred_val, prob_arr in zip(sids, labels, preds, probs):
                        biome = self.biome_map.get(str(sid), "Unknown")
                        test_data[biome]["true"].append(label_val)
                        test_data[biome]["pred"].append(pred_val)
                        test_data[biome]["probs"].append(prob_arr)
                        test_data[biome]["sample_ids"].append(sid)

        self._compute_and_log(test_data, epoch, "test", pl_module=pl_module)
        if was_training:
            pl_module.train()


class TerraFMEuroSATModule(MethodTrainingMixin, L.LightningModule):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float = WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}}


def parse_args():
    p = argparse.ArgumentParser(description="TerraFM EuroSAT Benchmark Pipeline")
    p.add_argument("--method", choices=["erm", "groupdro", "dbr", "bolp"], default="erm")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = out_dir / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    biome_map = load_biome_map(BIOME_JSON)
    benchmark_dir = DATA_DIR / "classification_v1.0"
    geobench.GEO_BENCH_DIR = str(DATA_DIR)
    geobench.task.GEO_BENCH_DIR = DATA_DIR

    tasks = list(geobench.task_iterator(
        benchmark_name="classification_v1.0",
        benchmark_dir=str(benchmark_dir),
    ))
    task = next((t for t in tasks if TASK_KEYWORD in t.dataset_name.lower()), None)
    assert task is not None, f"Task '{TASK_KEYWORD}' not found in classification_v1.0."

    train_ds = GeoBenchEuroSATDataset(task, split="train", augment=True)
    train_ds, protocol_groups = known_training_dataset(train_ds, lambda sid: lookup_biome(biome_map, sid))
    val_ds   = GeoBenchEuroSATDataset(task, split="valid", augment=False)
    test_ds  = GeoBenchEuroSATDataset(task, split="test",  augment=False)

    train_loader = training_loader(train_ds, method=args.method, groups=protocol_groups, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = TerraFMClassificationClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=args.checkpoint,
        freeze_backbone=True,
    )

    module = TerraFMEuroSATModule(model=base_model, lr=args.lr, weight_decay=args.weight_decay)
    configure_training_protocol(module, args.method, 'eurosat', protocol_groups, args, lookup=lambda sid: lookup_biome(biome_map, sid))
    args.method_protocol = module.method_protocol.config

    ckpt_macro_cb = ModelCheckpoint(
        monitor="val/macro_f1",
        mode="max",
        dirpath=CKPT_DIR,
        filename="best_val_macro_f1",
        save_top_k=1,
        save_last=True,
    )
    ckpt_worst_cb = ModelCheckpoint(
        monitor="val/worst_case_macro_f1",
        mode="max",
        dirpath=CKPT_DIR,
        filename="best_val_worst_f1",
        save_top_k=1,
    )
    logger_cb = EuroSATBiomeLogger(
        biome_map=biome_map,
        num_classes=NUM_CLASSES,
        class_names=[f"Class_{i}" for i in range(NUM_CLASSES)],
        output_dir=out_dir,
        test_loader=test_loader,
    )
    prog_cb = TQDMProgressBar(refresh_rate=20)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=args.epochs,
        callbacks=[ckpt_macro_cb, ckpt_worst_cb, logger_cb, prog_cb, lr_cb],
        logger=tb_logger,
        log_every_n_steps=10,
    )

    if args.mode == "train":
        print(f"\n[Training] Starting {args.epochs} Epochs of TerraFM ({args.method.upper()}) on EuroSAT...")
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Post-training evaluation on best worst checkpoint
    best_candidate = CKPT_DIR / "best_val_worst_f1.ckpt"
    eval_ckpt = str(best_candidate) if best_candidate.exists() else (ckpt_macro_cb.best_model_path or str(CKPT_DIR / "last.ckpt"))
    print(f"\n[Evaluation] Loading best checkpoint: {eval_ckpt}")
    ckpt_dict = torch.load(eval_ckpt, map_location="cpu")
    state_dict = ckpt_dict.get("state_dict", ckpt_dict)
    module.load_state_dict(state_dict, strict=False)
    module.to(f"cuda:{args.gpu}")
    module.eval()

    # Final Canonical Test selected metrics
    if (out_dir / "test_results.json").exists():
        with open(out_dir / "test_results.json") as f:
            t_hist = json.load(f)
            if t_hist:
                best_entry = t_hist[-1]
                selected = {
                    "model": "terrafm",
                    "method": args.method,
                    "dataset": "m-eurosat",
                    "test_score": best_entry.get("overall_macro_f1", 0.0),
                    "macro_f1": best_entry.get("overall_macro_f1", 0.0),
                    "micro_score": best_entry.get("overall_accuracy", 0.0),
                    "m_worst": best_entry.get("worst_case_macro_f1", 0.0),
                    "m_best": best_entry.get("best_case_macro_f1", 0.0),
                    "per": best_entry.get("per_ratio", 0.0),
                    "nfr": best_entry.get("normalized_fairness_range", 0.0),
                    "cross_group_std": best_entry.get("group_std_macro_f1", 0.0),
                    "gini": best_entry.get("gini_coefficient", 0.0),
                }
                with open(out_dir / "test_selected.json", "w") as f:
                    json.dump(selected, f, indent=2)
                print(f"[Artifacts] Canonical test_selected.json written to {out_dir / 'test_selected.json'}")


if __name__ == "__main__":
    main()
