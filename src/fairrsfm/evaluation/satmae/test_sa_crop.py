#!/usr/bin/env python3
"""
SatMAE South Africa Crop Type Test Split Evaluator
===================================================
Evaluates trained SatMAE SA-Crop segmentation checkpoints (ERM, DBR, GroupDRO, BOLP)
on the official Geo-Bench test split and computes mIoU, per-class IoU, and fairness metrics.
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

from evaluation.metrics import (
    compute_miou,
    compute_macro_f1_seg,
    compute_per_class_iou,
    compute_ece_seg,
    compute_fairness_summary,
    compute_gini,
)

# Backbone import
UPSTREAM_SATMAE = ROOT_DIR / "models/upstream/SatMAE"
if str(UPSTREAM_SATMAE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SATMAE))

import models_vit_group_channels

SATMAE_SA_CROP_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
SA_CROP_10BAND_MEAN = [16.5267, 26.6364, 36.6966, 46.3887, 58.2815, 63.5758, 68.1836, 69.1426, 83.6268, 65.7677]
SA_CROP_10BAND_STD  = [ 9.3295, 12.6745, 19.4219, 19.4874, 19.9592, 21.5381, 23.0508, 22.3297, 28.1442, 27.2346]
NUM_CLASSES  = 10
IGNORE_INDEX = -1
CLASS_NAMES  = [f"Crop_{i}" for i in range(NUM_CLASSES)]


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


class GeoBenchSACropDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str, img_size: int = 96):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = SATMAE_SA_CROP_BAND_INDICES
        self.img_size     = img_size
        self.normalizer   = SatMAENormalize(SA_CROP_10BAND_MEAN, SA_CROP_10BAND_STD)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        array, _ = sample.pack_to_3d(band_names=self.band_names, resample=False, fill_value=None, resample_order=3)
        image = array.astype(np.float32)[:, :, self.band_indices]
        mask  = sample.label.data.astype(np.int64)

        x = torch.from_numpy(image).permute(2, 0, 1).float()
        y = torch.from_numpy(mask).long().unsqueeze(0)

        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0).float(), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).long().squeeze(0)

        x = self.normalizer(x)
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


class SatMAESegmentationLinearProbe(nn.Module):
    def __init__(self, model_type="vit_base_patch16", img_size=96, patch_size=16, in_chans=10, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.G = 3
        channel_groups = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))

        fn = getattr(models_vit_group_channels, model_type)
        self.backbone = fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            channel_groups=channel_groups,
            global_pool=False
        )
        self.embed_dim = self.backbone.embed_dim
        self.probe_head = nn.Sequential(
            nn.BatchNorm2d(self.G * self.embed_dim, affine=False),
            nn.Conv2d(self.G * self.embed_dim, num_classes, kernel_size=1)
        )

    def forward_features_spatial(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        bb = self.backbone
        x_c_embed = []
        for i, group in enumerate(bb.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(bb.patch_embed[i](x_c))

        x_tokens = torch.stack(x_c_embed, dim=1)
        _, G_bb, L, D_bb = x_tokens.shape

        channel_embed = bb.channel_embed.unsqueeze(2)
        pos_embed = bb.pos_embed[:, 1:, :].unsqueeze(1)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)

        x_tokens = x_tokens + pos_channel
        x_tokens = x_tokens.view(b, -1, D_bb)
        cls_pos_channel = torch.cat((bb.pos_embed[:, :1, :], bb.channel_cls_embed), dim=-1)
        cls_tokens = cls_pos_channel + bb.cls_token.expand(b, -1, -1)
        x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
        x_tokens = bb.pos_drop(x_tokens)

        for blk in bb.blocks:
            x_tokens = blk(x_tokens)

        patches = x_tokens[:, 1:, :]
        patches = patches.view(b, self.G, L, self.embed_dim)
        patches = patches.permute(0, 2, 1, 3).reshape(b, L, self.G * self.embed_dim)
        spatial_features = patches.permute(0, 2, 1).view(b, self.G * self.embed_dim, self.grid_size, self.grid_size)
        return spatial_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feats = self.forward_features_spatial(x)
        logits = self.probe_head(feats)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


class SatMAEBOLPSegmentationProbe(SatMAESegmentationLinearProbe):
    def __init__(self, model_type="vit_base_patch16", img_size=96, patch_size=16, in_chans=10, num_classes=10, k=1):
        super().__init__(model_type, img_size, patch_size, in_chans, num_classes)
        in_dim = self.G * self.embed_dim
        self.register_buffer("mu0", torch.zeros(in_dim))
        self.register_buffer("Pk", torch.zeros(in_dim, in_dim))
        self.register_buffer("mu_table", torch.zeros(10, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feats = self.forward_features_spatial(x)
        # Apply projection
        b_f, c_f, h_f, w_f = feats.shape
        z_perm = feats.permute(0, 2, 3, 1).reshape(-1, c_f)
        z_proj = z_perm @ self.Pk.T
        feats_proj = z_proj.view(b_f, h_f, w_f, c_f).permute(0, 3, 1, 2)
        logits = self.probe_head(feats_proj)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


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
        model = SatMAEBOLPSegmentationProbe()
    else:
        model = SatMAESegmentationLinearProbe()

    if backbone_weights and backbone_weights.exists():
        bb_ckpt = torch.load(backbone_weights, map_location="cpu", weights_only=False)
        bb_state = bb_ckpt.get("model", bb_ckpt.get("state_dict", bb_ckpt))
        model.backbone.load_state_dict(bb_state, strict=False)

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
    per_class_iou_by_biome = {}

    for b in sorted(biome_data.keys()):
        yt = np.concatenate(biome_data[b]["true"])
        yp = np.concatenate(biome_data[b]["pred"])
        pm = np.concatenate(biome_data[b]["probs_max"])
        if len(yt) == 0:
            continue
        b_miou  = compute_miou(yt, yp, num_classes, ignore_index=IGNORE_INDEX)
        b_f1    = compute_macro_f1_seg(yt, yp, num_classes, ignore_index=IGNORE_INDEX)
        b_ece   = compute_ece_seg(yt, pm, yp, ignore_index=IGNORE_INDEX)
        b_count = len(biome_data[b]["sample_ids"])

        biome_mious[b]  = b_miou
        biome_f1s[b]    = b_f1
        biome_eces[b]   = b_ece
        biome_counts[b] = b_count

        per_group_metrics[b] = {
            "miou": b_miou,
            "macro_f1": b_f1,
            "ece": b_ece,
            "sample_count": b_count,
        }
        per_class_iou_by_biome[b] = compute_per_class_iou(yt, yp, num_classes, ignore_index=IGNORE_INDEX)

    per_class_iou_table = {}
    class_disparities = {}
    for c in range(num_classes):
        c_name = CLASS_NAMES[c]
        c_scores, c_vals = {}, []
        for b in sorted(biome_data.keys()):
            val = per_class_iou_by_biome[b][c]
            if not np.isnan(val):
                c_scores[b] = float(val)
                c_vals.append(float(val))
        per_class_iou_table[c_name] = c_scores
        class_disparities[c_name] = float(max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0

    fairness = compute_fairness_summary(biome_mious, global_miou)
    ece_vals = [v for v in biome_eces.values() if not np.isnan(v)]
    ece_disp = (max(ece_vals) - min(ece_vals)) if ece_vals else float("nan")

    results = {
        "global_miou": global_miou,
        "global_macro_f1": global_f1,
        "global_ece": global_ece,
        "worst_case_miou": fairness["worst_case"],
        "best_case_miou": fairness["best_case"],
        "group_mean_miou": fairness["mean"],
        "group_miou_std": fairness["std"],
        "performance_equity_ratio": fairness["per"],
        "normalized_fairness_range": fairness["nfr"],
        "gini_coefficient": fairness["gini"],
        "ece_disparity": ece_disp,
        "per_biome_miou": biome_mious,
        "per_biome_macro_f1": biome_f1s,
        "per_biome_ece": biome_eces,
        "per_biome_counts": biome_counts,
        "per_group_metrics": per_group_metrics,
        "class_disparities": class_disparities,
        "per_class_iou_table": per_class_iou_table,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SatMAE SA-Crop Segmentation Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"))
    parser.add_argument("--data_dir", type=str, default=None, help="Path to GeoBench data root")
    parser.add_argument("--biome_json", type=str, default=str(ROOT_DIR / "configs/sa_crop_biome_mapping.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir) if args.data_dir else (ROOT_DIR / "configs/geobench")
    if not data_dir.exists():
        data_dir = ROOT_DIR / "Geobench"
    benchmark_dir = data_dir / "segmentation_v1.0"

    tasks = list(geobench.task_iterator(benchmark_name="segmentation_v1.0", benchmark_dir=str(benchmark_dir)))
    task = next((t for t in tasks if "crop" in t.dataset_name.lower()), None)
    if task is None:
        raise RuntimeError("SA-Crop task not found in GeoBench benchmark directory.")

    test_ds = GeoBenchSACropDataset(task, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    biome_map = load_biome_map(Path(args.biome_json))
    model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on SA-Crop test split...")
    results = evaluate(model, test_loader, biome_map, device, num_classes=NUM_CLASSES)

    print("\n" + "=" * 60)
    print("SA-Crop Test Evaluation Results")
    print("=" * 60)
    print(f"Overall mIoU     : {results['global_miou']:.4f}")
    print(f"Overall Macro F1 : {results['global_macro_f1']:.4f}")
    print(f"Worst-Group mIoU : {results['worst_case_miou']:.4f}")
    print(f"Best-Group mIoU  : {results['best_case_miou']:.4f}")
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
