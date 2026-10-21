#!/usr/bin/env python3
"""
DOFA BigEarthNet Test Split Evaluator
=====================================
Evaluates trained DOFA BigEarthNet checkpoints (ERM, DBR, GroupDRO, BOLP)
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
from sklearn.metrics import accuracy_score, f1_score

# Add repo root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import geobench
import geobench.task

from models.dofa_wrapper import DOFAClassifier
from evaluation.metrics import (
    compute_subgroup_macro_f1_multilabel,
    compute_ece_multilabel,
    compute_map_multilabel,
    compute_fairness_summary,
)

DOFA_BAND_NAMES = [
    "01 - Coastal aerosol", "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "09 - Water vapour",
    "11 - SWIR", "12 - SWIR"
]
DOFA_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
BIGEARTHNET_DOFA_MEAN = [386.6516, 488.9927, 714.6071, 738.2603, 1114.4219, 1910.1897, 2191.4824, 2334.2920, 2392.9136, 2367.2874, 1902.6918, 1261.0698]
BIGEARTHNET_DOFA_STD  = [467.3123, 510.7866, 551.7964, 691.7815,  700.4516,  976.7469, 1134.8865, 1238.0713, 1215.9766, 1153.8583, 1117.0094,  894.7390]
NUM_CLASSES = 43

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


class GeoBenchClassifDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str):
        super().__init__()
        self.dataset = task.get_dataset(split=split)
        self.mean = torch.tensor(BIGEARTHNET_DOFA_MEAN, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(BIGEARTHNET_DOFA_STD,  dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x_data, _ = sample.pack_to_3d(band_names=DOFA_BAND_NAMES)
        x = torch.from_numpy(x_data.astype(np.float32)).permute(2, 0, 1)
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - self.mean) / self.std

        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for label_idx in sample.label:
            if label_idx < NUM_CLASSES:
                label_vec[label_idx] = 1.0

        sid = str(sample.sample_name)
        return x, torch.from_numpy(label_vec), sid


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


class DOFABOLPClassifier(nn.Module):
    def __init__(self, num_classes=43, k=2, checkpoint_path=None):
        super().__init__()
        self.encoder = DOFAClassifier(
            num_classes=0,
            model_type="vit_base_patch16_224",
            checkpoint_path=checkpoint_path,
            freeze_backbone=True
        )
        embed_dim = self.encoder.embed_dim
        self.register_buffer("mu0", torch.zeros(embed_dim))
        self.register_buffer("Pk", torch.zeros(k, embed_dim))
        self.register_buffer("mu_table", torch.zeros(10, embed_dim))
        self.head = nn.Linear(k, num_classes)

    def forward(self, x, wavelengths):
        feats = self.encoder.forward_features(x, wavelengths)
        z_proj = feats @ self.Pk.T
        return self.head(z_proj)


def build_and_load_model(ckpt_path: Path, backbone_weights: Path, device: torch.device):
    print(f"[Loading] Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)

    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("model.", "")
        cleaned[clean_k] = v

    is_bolp = any("Pk" in k for k in cleaned.keys())
    if is_bolp:
        k_val = cleaned["Pk"].shape[0] if "Pk" in cleaned else 2
        model = DOFABOLPClassifier(num_classes=NUM_CLASSES, k=k_val, checkpoint_path=str(backbone_weights))
    else:
        model = DOFAClassifier(
            num_classes=NUM_CLASSES,
            model_type="vit_base_patch16_224",
            checkpoint_path=str(backbone_weights) if backbone_weights.exists() else None,
            freeze_backbone=True
        )

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate(model, test_loader, biome_map, device, num_classes=43, threshold=0.5):
    biome_data = defaultdict(lambda: {"true": [], "probs": [], "sample_ids": []})
    all_true, all_probs = [], []

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)
            try:
                logits = model(x, DOFA_WAVELENGTHS)
            except TypeError:
                logits = model(x)

            probs = torch.sigmoid(logits.float()).cpu().numpy()
            labels = y.numpy()

            for i, sid in enumerate(sample_ids):
                group = biome_map.get(str(sid), "Unknown")
                if group != "Unknown":
                    biome_data[group]["true"].append(labels[i])
                    biome_data[group]["probs"].append(probs[i])
                    biome_data[group]["sample_ids"].append(str(sid))

                all_true.append(labels[i])
                all_probs.append(probs[i])

    all_true  = np.array(all_true)
    all_probs = np.array(all_probs)
    all_pred  = (all_probs >= threshold).astype(int)

    global_macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    global_micro_f1 = float(f1_score(all_true, all_pred, average="micro", zero_division=0))
    global_acc      = float(accuracy_score(all_true, all_pred))
    global_map      = compute_map_multilabel(all_true, all_probs, num_classes)

    biome_macro_f1, biome_eces, biome_counts = {}, {}, {}
    per_group_metrics = {}

    for b in sorted(biome_data.keys()):
        yt = np.array(biome_data[b]["true"])
        pr = np.array(biome_data[b]["probs"])
        if len(yt) == 0:
            continue
        yp = (pr >= threshold).astype(int)
        b_macro = compute_subgroup_macro_f1_multilabel(yt, yp, num_classes)
        b_micro = float(f1_score(yt, yp, average="micro", zero_division=0))
        b_acc   = float(accuracy_score(yt, yp))
        b_map   = compute_map_multilabel(yt, pr, num_classes)
        b_ece   = compute_ece_multilabel(yt, pr, yp)
        b_count = len(yt)

        biome_macro_f1[b] = b_macro
        biome_eces[b]     = b_ece
        biome_counts[b]   = b_count

        per_group_metrics[b] = {
            "macro_f1": b_macro,
            "micro_f1": b_micro,
            "exact_accuracy": b_acc,
            "mAP": b_map,
            "ece": b_ece,
            "sample_count": b_count,
        }

    fairness = compute_fairness_summary(biome_macro_f1, global_macro_f1)
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
        "per_biome_macro_f1": biome_macro_f1,
        "per_biome_ece": biome_eces,
        "per_biome_counts": biome_counts,
        "per_group_metrics": per_group_metrics,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DOFA BigEarthNet Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "checkpoints/dofa/dofa_base_patch16_224.pth"))
    parser.add_argument("--data_dir", type=str, default=None, help="Path to GeoBench data root")
    parser.add_argument("--biome_json", type=str, default=str(ROOT_DIR / "configs/bigearthnet_biome_mapping.json"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir) if args.data_dir else (ROOT_DIR / "configs/geobench")
    if not data_dir.exists():
        data_dir = ROOT_DIR / "Geobench"
    benchmark_dir = data_dir / "classification_v1.0"

    tasks = list(geobench.task_iterator(benchmark_name="classification_v1.0", benchmark_dir=str(benchmark_dir)))
    task = next((t for t in tasks if "bigearthnet" in t.dataset_name.lower()), None)
    if task is None:
        raise RuntimeError("BigEarthNet task not found in GeoBench benchmark directory.")

    test_ds = GeoBenchClassifDataset(task, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    biome_map = load_biome_map(Path(args.biome_json))
    model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on DOFA BigEarthNet test split...")
    results = evaluate(model, test_loader, biome_map, device, num_classes=NUM_CLASSES, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("DOFA BigEarthNet Test Evaluation Results")
    print("=" * 60)
    print(f"Overall Macro F1 : {results['global_macro_f1']:.4f}")
    print(f"Overall Micro F1 : {results['global_micro_f1']:.4f}")
    print(f"Overall mAP      : {results['global_map']:.4f}")
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
