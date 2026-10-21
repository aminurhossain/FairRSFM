#!/usr/bin/env python3
"""
SatMAE EuroSAT Test Split Evaluator
====================================
Evaluates trained SatMAE EuroSAT checkpoints (ERM, DBR, GroupDRO, BOLP)
on the official Geo-Bench test split and computes the full fairness suite.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

# Add repo root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import geobench
import geobench.task

from evaluation.metrics import (
    compute_subgroup_macro_f1,
    compute_per_class_f1,
    compute_ece_single_label,
    compute_equalized_odds_single_label,
    compute_demographic_parity_single_label,
    compute_map_score_single_label,
    compute_fairness_summary,
)

# Backbone import
UPSTREAM_SATMAE = ROOT_DIR / "models/upstream/SatMAE"
if str(UPSTREAM_SATMAE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SATMAE))

import models_vit_group_channels


# Constants
SATMAE_EUROSAT_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
SATMAE_10BAND_MEAN = [1353.73, 1117.25, 1041.49, 946.54, 1199.19, 2003.01, 2374.01, 2301.22, 732.18, 1820.69]
SATMAE_10BAND_STD  = [65.47, 154.01, 187.99, 278.18, 228.16, 356.59, 456.03, 531.55, 98.95, 378.99]
NUM_CLASSES = 10
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]


class SatMAENormalize:
    def __init__(self, mean, std):
        mean_arr = np.array(mean, dtype=np.float32)[:, None, None]
        std_arr  = np.array(std,  dtype=np.float32)[:, None, None]
        self.min_val = torch.from_numpy(mean_arr - 2.0 * std_arr)
        self.max_val = torch.from_numpy(mean_arr + 2.0 * std_arr)
        self.range   = self.max_val - self.min_val

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = (tensor - self.min_val) / self.range
        return torch.clamp(norm, 0.0, 1.0)


class GeoBenchSatMAEDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, img_size: int = 96):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = SATMAE_EUROSAT_BAND_INDICES
        self.img_size     = img_size
        self.normalizer   = SatMAENormalize(SATMAE_10BAND_MEAN, SATMAE_10BAND_STD)
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
        x = self.normalizer(x)
        x = self.spatial_transform(x)
        y = int(sample.label)
        sid = str(sample.sample_name)
        return x, y, sid


def load_biome_map(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"Warning: Biome map file not found: {json_path}")
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
        group_name = macro_groups.get(int(code), "Unknown") if code is not None else "Unknown"
        biome_map[str(r["geobench_id"])] = group_name
    return biome_map


class SatMAEClassifier(nn.Module):
    def __init__(self, model_type="vit_base_patch16", img_size=96, patch_size=16, in_chans=10, num_classes=10):
        super().__init__()
        channel_groups = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
        fn = getattr(models_vit_group_channels, model_type)
        self.backbone = fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            channel_groups=channel_groups,
            global_pool=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        return self.head(feats)


class BOLPProjector(nn.Module):
    def __init__(self, embed_dim: int, k: int):
        super().__init__()
        self.k = k
        self.register_buffer("mu0", torch.zeros(embed_dim))
        self.register_buffer("Pk", torch.zeros(k, embed_dim))
        self.register_buffer("mu_table", torch.zeros(10, embed_dim))

    def forward(self, z, biome_ids=None):
        return z @ self.Pk.T


class SatMAEBOLPClassifier(nn.Module):
    def __init__(self, model_type="vit_base_patch16", img_size=96, patch_size=16, in_chans=10, num_classes=10, k=1):
        super().__init__()
        channel_groups = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
        fn = getattr(models_vit_group_channels, model_type)
        self.backbone = fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            channel_groups=channel_groups,
            global_pool=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.projector = BOLPProjector(self.embed_dim, k=k)
        self.head = nn.Linear(k, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        z_proj = self.projector(feats)
        return self.head(z_proj)


def build_and_load_model(ckpt_path: Path, backbone_weights: Path, device: torch.device):
    print(f"[Loading] Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)

    # Clean keys
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("model.", "")
        cleaned[clean_k] = v

    # Check if BOLP
    is_bolp = any("projector" in k for k in cleaned.keys())
    if is_bolp:
        k_val = cleaned["projector.Pk"].shape[0] if "projector.Pk" in cleaned else 1
        model = SatMAEBOLPClassifier(k=k_val)
    else:
        model = SatMAEClassifier()

    # Load backbone pretrained weights first if available
    if backbone_weights and backbone_weights.exists():
        bb_ckpt = torch.load(backbone_weights, map_location="cpu", weights_only=False)
        bb_state = bb_ckpt.get("model", bb_ckpt.get("state_dict", bb_ckpt))
        model.backbone.load_state_dict(bb_state, strict=False)

    # Load trained probe weights
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate(model, test_loader, biome_map, device, num_classes=10):
    biome_data = defaultdict(lambda: {"true": [], "pred": [], "probs": [], "sample_ids": []})
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            labels = y.numpy()

            for i, sid in enumerate(sample_ids):
                group = biome_map.get(str(sid), "Unknown")
                if group != "Unknown":
                    biome_data[group]["true"].append(int(labels[i]))
                    biome_data[group]["pred"].append(int(preds[i]))
                    biome_data[group]["probs"].append(probs[i])
                    biome_data[group]["sample_ids"].append(str(sid))

                all_true.append(int(labels[i]))
                all_pred.append(int(preds[i]))
                all_probs.append(probs[i])

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)
    all_probs = np.array(all_probs)

    global_macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    global_micro_f1 = float(f1_score(all_true, all_pred, average="micro", zero_division=0))
    global_acc      = float(accuracy_score(all_true, all_pred))
    global_map      = compute_map_score_single_label(all_true, all_probs, num_classes)

    biome_macro_f1, biome_eces, biome_counts = {}, {}, {}
    per_group_metrics = {}

    for b in sorted(biome_data.keys()):
        yt = np.array(biome_data[b]["true"])
        yp = np.array(biome_data[b]["pred"])
        pr = np.array(biome_data[b]["probs"])
        if len(yt) == 0:
            continue
        b_macro = compute_subgroup_macro_f1(yt, yp)
        b_micro = float(f1_score(yt, yp, average="micro", zero_division=0))
        b_acc   = float(accuracy_score(yt, yp))
        b_map   = compute_map_score_single_label(yt, pr, num_classes)
        b_ece   = compute_ece_single_label(yt, pr, yp)
        b_count = len(yt)

        biome_macro_f1[b] = b_macro
        biome_eces[b]     = b_ece
        biome_counts[b]   = b_count

        per_group_metrics[b] = {
            "macro_f1": b_macro,
            "micro_f1": b_micro,
            "accuracy": b_acc,
            "mAP": b_map,
            "ece": b_ece,
            "sample_count": b_count,
        }

    fairness = compute_fairness_summary(biome_macro_f1, global_macro_f1)
    eq_odds  = compute_equalized_odds_single_label(biome_data, num_classes)
    dp_disp  = compute_demographic_parity_single_label(biome_data, num_classes)

    ece_vals = [v for v in biome_eces.values() if not np.isnan(v)]
    ece_disp = (max(ece_vals) - min(ece_vals)) if ece_vals else float("nan")

    results = {
        "global_macro_f1": global_macro_f1,
        "global_micro_f1": global_micro_f1,
        "global_accuracy": global_acc,
        "global_map": global_map,
        "worst_case_macro_f1": fairness["worst_case"],
        "best_case_macro_f1": fairness["best_case"],
        "group_mean_macro_f1": fairness["mean"],
        "group_macro_f1_std": fairness["std"],
        "performance_equity_ratio": fairness["per"],
        "normalized_fairness_range": fairness["nfr"],
        "gini_coefficient": fairness["gini"],
        "ece_disparity": ece_disp,
        "equalized_odds_disparity": eq_odds,
        "demographic_parity_disparity": dp_disp,
        "per_biome_macro_f1": biome_macro_f1,
        "per_biome_ece": biome_eces,
        "per_biome_counts": biome_counts,
        "per_group_metrics": per_group_metrics,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SatMAE EuroSAT Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"))
    parser.add_argument("--data_dir", type=str, default=None, help="Path to GeoBench data root")
    parser.add_argument("--biome_json", type=str, default=str(ROOT_DIR / "configs/eurosat_biome_mapping.json"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Locate dataset
    data_dir = Path(args.data_dir) if args.data_dir else (ROOT_DIR / "configs/geobench")
    if not data_dir.exists():
        data_dir = ROOT_DIR / "Geobench"
    benchmark_dir = data_dir / "classification_v1.0"

    tasks = list(geobench.task_iterator(benchmark_name="classification_v1.0", benchmark_dir=str(benchmark_dir)))
    task = next((t for t in tasks if "eurosat" in t.dataset_name.lower()), None)
    if task is None:
        raise RuntimeError("EuroSAT task not found in GeoBench benchmark directory.")

    test_ds = GeoBenchSatMAEDataset(task, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    biome_map = load_biome_map(Path(args.biome_json))
    model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on test split...")
    results = evaluate(model, test_loader, biome_map, device, num_classes=NUM_CLASSES)

    print("\n" + "=" * 60)
    print("Test Evaluation Results")
    print("=" * 60)
    print(f"Overall Macro F1 : {results['global_macro_f1']:.4f}")
    print(f"Overall Accuracy : {results['global_accuracy']:.4f}")
    print(f"Worst-Group F1   : {results['worst_case_macro_f1']:.4f}")
    print(f"Best-Group F1    : {results['best_case_macro_f1']:.4f}")
    print(f"Equity Ratio PER : {results['performance_equity_ratio']:.4f}")
    print(f"Disparity (NFR)  : {results['normalized_fairness_range']:.4f}")
    print("=" * 60)

    out_path = Path(args.out_json) if args.out_json else Path(args.ckpt).parent / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
