#!/usr/bin/env python3
"""
DOFA ViT-Base 16 SA-Crop Test Split Evaluator
============================================
Evaluates trained DOFA SA-Crop checkpoints across all methods (ERM, DBR, GroupDRO, BOLP)
and head architectures (Linear Probe, UPerNet) on the official Geo-Bench test split.
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

# Add repo root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import geobench
import geobench.task

from models.dofa_wrapper import DOFAClassifier, DOFASegmentationProbe, DOFAUperNet
from evaluation.metrics import (
    compute_miou,
    compute_macro_f1_seg,
    compute_ece_seg,
    compute_eqodds_seg,
    compute_demographic_parity_seg,
    compute_fairness_summary,
)

DOFA_BAND_NAMES = [
    "04 - Red", "03 - Green", "02 - Blue",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "11 - SWIR", "12 - SWIR"
]
DOFA_WAVELENGTHS = [0.66, 0.56, 0.49, 0.70, 0.74, 0.78, 0.84, 1.61, 2.20]
NUM_CLASSES  = 10
IGNORE_INDEX = 0


class GeoBenchSegDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str):
        super().__init__()
        self.dataset = task.get_dataset(split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x_data, _ = sample.pack_to_3d(band_names=DOFA_BAND_NAMES)
        x = torch.from_numpy(x_data.astype(np.float32)).permute(2, 0, 1)
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        y_data = sample.label.data if hasattr(sample.label, "data") else sample.label
        if hasattr(y_data, "squeeze"):
            y_data = y_data.squeeze()
        y = torch.tensor(y_data, dtype=torch.long)
        y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=(224, 224), mode="nearest").squeeze().long()

        sid = str(sample.sample_name)
        return x, y, sid


def load_biome_map(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"Warning: Biome map file not found: {json_path}")
        return {}
    with open(json_path) as f:
        records = json.load(f)

    macro_groups = {
        12: "Mediterranean Forests, Woodlands & Scrub",
        13: "Deserts & Xeric Shrublands",
    }

    biome_map = {}
    for r in records:
        code = r.get("biome_code", None)
        if code is not None:
            try:
                group_name = macro_groups.get(int(code), r.get("biome_name", "Unknown"))
            except ValueError:
                group_name = r.get("biome_name", "Unknown")
        else:
            group_name = "Unknown"
        biome_map[str(r["geobench_id"])] = group_name
    return biome_map


class DOFABOLPSegmentationProbe(nn.Module):
    def __init__(self, num_classes=10, img_size=224, checkpoint_path=None):
        super().__init__()
        self.encoder = DOFAClassifier(
            num_classes=0,
            model_type="vit_base_patch16_224",
            checkpoint_path=checkpoint_path,
            freeze_backbone=True
        )
        embed_dim = self.encoder.embed_dim
        self.register_buffer("mu0", torch.zeros(embed_dim))
        self.register_buffer("Pk", torch.zeros(embed_dim, embed_dim))
        self.register_buffer("mu_table", torch.zeros(10, embed_dim))
        self.grid_size = img_size // 16
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x, wavelengths):
        b, c, h, w = x.shape
        patches = self.encoder.forward_patches(x, wavelengths)
        proj_patches = patches @ self.Pk.T
        spatial = proj_patches.permute(0, 2, 1).view(b, -1, self.grid_size, self.grid_size)
        logits = self.head(spatial)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


class DOFAUperNetBOLP(nn.Module):
    def __init__(self, num_classes=10, img_size=224, checkpoint_path=None):
        super().__init__()
        self.base_model = DOFAUperNet(
            num_classes=num_classes,
            model_type="vit_base_patch16_224",
            img_size=img_size,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True
        )
        embed_dim = self.base_model.encoder.embed_dim
        self.register_buffer("proj_matrix", torch.eye(embed_dim))

    def forward(self, x, wavelengths):
        b, c, h, w = x.shape
        multi_feats = self.base_model.encoder.forward_intermediates(
            x, wavelengths, out_indices=self.base_model.out_indices
        )
        proj_feats = []
        for feat in multi_feats:
            B, D, H_i, W_i = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, D)
            feat_proj = feat_flat @ self.proj_matrix.T
            feat_out = feat_proj.view(B, H_i, W_i, D).permute(0, 3, 1, 2)
            proj_feats.append(feat_out)
        logits = self.base_model.head(proj_feats)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def build_and_load_model(ckpt_path: Path, backbone_weights: Path, device: torch.device):
    print(f"[Loading] Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)

    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("model.", "")
        cleaned[clean_k] = v

    is_upernet = any("decode_head" in k or "fpn" in k for k in cleaned.keys())
    is_bolp    = any("Pk" in k or "proj_matrix" in k or "projector" in k for k in cleaned.keys())

    if is_upernet and is_bolp:
        print("[Model Type] Detected DOFA UPerNet BOLP architecture")
        model = DOFAUperNetBOLP(
            num_classes=NUM_CLASSES,
            img_size=224,
            checkpoint_path=str(backbone_weights) if backbone_weights.exists() else None,
        )
    elif is_upernet:
        print("[Model Type] Detected DOFA UPerNet architecture (ERM / GroupDRO / DBR)")
        model = DOFAUperNet(
            num_classes=NUM_CLASSES,
            model_type="vit_base_patch16_224",
            img_size=224,
            checkpoint_path=str(backbone_weights) if backbone_weights.exists() else None,
            freeze_backbone=True
        )
    elif is_bolp:
        print("[Model Type] Detected DOFA Linear Probe BOLP architecture")
        model = DOFABOLPSegmentationProbe(
            num_classes=NUM_CLASSES,
            img_size=224,
            checkpoint_path=str(backbone_weights) if backbone_weights.exists() else None,
        )
    else:
        print("[Model Type] Detected DOFA Linear Probe architecture (ERM / GroupDRO / DBR)")
        model = DOFASegmentationProbe(
            num_classes=NUM_CLASSES,
            model_type="vit_base_patch16_224",
            img_size=224,
            checkpoint_path=str(backbone_weights) if backbone_weights.exists() else None,
            freeze_backbone=True
        )

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate(model, test_loader, biome_map, device, num_classes=10):
    biome_data = defaultdict(lambda: {"true": [], "pred": [], "probs_max": [], "sample_ids": []})
    all_true, all_pred, all_pmax = [], [], []

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)
            try:
                logits = model(x, DOFA_WAVELENGTHS)
            except TypeError:
                logits = model(x)

            labels = y.numpy()
            probs = torch.softmax(logits, dim=1)
            probs_max = probs.max(dim=1).values.cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            for i, sid in enumerate(sample_ids):
                group = biome_map.get(str(sid), "Unknown")
                if group != "Unknown":
                    biome_data[group]["true"].append(labels[i].reshape(-1))
                    biome_data[group]["pred"].append(preds[i].reshape(-1))
                    biome_data[group]["probs_max"].append(probs_max[i].reshape(-1))
                    biome_data[group]["sample_ids"].append(str(sid))

                all_true.append(labels[i].reshape(-1))
                all_pred.append(preds[i].reshape(-1))
                all_pmax.append(probs_max[i].reshape(-1))

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_pmax = np.concatenate(all_pmax, axis=0)

    global_miou = compute_miou(all_true, all_pred, num_classes, ignore_index=IGNORE_INDEX)
    global_f1   = compute_macro_f1_seg(all_true, all_pred, num_classes, ignore_index=IGNORE_INDEX)
    global_ece  = compute_ece_seg(all_true, all_pmax, all_pred, ignore_index=IGNORE_INDEX)

    biome_mious, biome_f1s, biome_eces, biome_counts = {}, {}, {}, {}
    per_group_metrics = {}

    for b in sorted(biome_data.keys()):
        yt = np.concatenate(biome_data[b]["true"])
        yp = np.concatenate(biome_data[b]["pred"])
        pm = np.concatenate(biome_data[b]["probs_max"])
        if len(yt) == 0:
            continue
        b_miou = compute_miou(yt, yp, num_classes, ignore_index=IGNORE_INDEX)
        b_f1   = compute_macro_f1_seg(yt, yp, num_classes, ignore_index=IGNORE_INDEX)
        b_ece  = compute_ece_seg(yt, pm, yp, ignore_index=IGNORE_INDEX)

        biome_mious[b]  = b_miou
        biome_f1s[b]    = b_f1
        biome_eces[b]   = b_ece
        biome_counts[b] = len(biome_data[b]["sample_ids"])
        per_group_metrics[b] = {
            "miou": b_miou,
            "macro_f1": b_f1,
            "ece": b_ece,
            "tile_count": len(biome_data[b]["sample_ids"]),
        }

    fairness = compute_fairness_summary(biome_mious)
    eq_odds = compute_eqodds_seg(biome_data, num_classes, ignore_index=IGNORE_INDEX)
    dp_diff = compute_demographic_parity_seg(biome_data, num_classes, ignore_index=IGNORE_INDEX)

    return {
        "global_miou": global_miou,
        "global_macro_f1": global_f1,
        "global_ece": global_ece,
        "worst_case_miou": fairness["worst_case"],
        "best_case_miou": fairness["best_case"],
        "miou_disparity": fairness["disparity"],
        "performance_equity_ratio": fairness["performance_equity_ratio"],
        "gini_coefficient": fairness["gini_coefficient"],
        "normalized_fairness_range": fairness["normalized_fairness_range"],
        "equalized_odds_disparity": eq_odds,
        "demographic_parity_difference": dp_diff,
        "per_biome_miou": biome_mious,
        "per_biome_f1": biome_f1s,
        "per_biome_ece": biome_eces,
        "per_group_metrics": per_group_metrics,
        "biome_tile_counts": biome_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DOFA SA-Crop Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "models/dofa/DOFA_ViT_base_e100.pth"))
    parser.add_argument("--data_dir", type=str, default=None, help="Path to GeoBench data root")
    parser.add_argument("--biome_json", type=str, default=str(ROOT_DIR / "configs/sa_crop_biome_mapping.json"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve data dir
    data_dir = Path(args.data_dir) if args.data_dir else Path("/data1/aminur/dataset/geobench")
    if not data_dir.exists():
        data_dir = Path("/mnt/DATA1/geobench")
    os.environ["GEO_BENCH_DIR"] = str(data_dir)
    geobench.GEO_BENCH_DIR = str(data_dir)
    geobench.task.GEO_BENCH_DIR = data_dir

    task = None
    for t in geobench.task_iterator(benchmark_name="segmentation_v1.0"):
        if "sa-crop" in t.dataset_name.lower():
            task = t
            break
    assert task is not None, "SA-Crop task not found in GeoBench"

    test_ds = GeoBenchSegDataset(task, "test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    biome_map = load_biome_map(Path(args.biome_json))

    model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on DOFA SA-Crop test split...")
    results = evaluate(model, test_loader, biome_map, device, num_classes=NUM_CLASSES)

    print("\n" + "=" * 60)
    print("DOFA SA-Crop Test Evaluation Results")
    print("=" * 60)
    print(f"Overall mIoU     : {results['global_miou']:.4f}")
    print(f"Overall Macro F1 : {results['global_macro_f1']:.4f}")
    print(f"Worst-Group mIoU : {results['worst_case_miou']:.4f}")
    print(f"Equity Ratio     : {results['performance_equity_ratio']:.4f}")
    print(f"Gini Index       : {results['gini_coefficient']:.4f}")
    print("=" * 60)

    out_json = Path(args.out_json) if args.out_json else Path(args.ckpt).parent / "test_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Test results saved to: {out_json}")


if __name__ == "__main__":
    main()
