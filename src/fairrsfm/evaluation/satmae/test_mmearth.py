#!/usr/bin/env python3
"""
SatMAE MMEarth-20k Test Split Evaluator
=======================================
Evaluates trained SatMAE MMEarth segmentation checkpoints (ERM, DBR, GroupDRO, BOLP)
on the test split and computes mIoU, per-class IoU, and biome fairness metrics.
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

# MMEarth dataset directory
MMEARTH_DIR = ROOT_DIR / "methods/prithvi/segmentation/mmearth_20k"
if str(MMEARTH_DIR) not in sys.path:
    sys.path.insert(0, str(MMEARTH_DIR))

from mmearth_dynamic_world_datamodule import (
    DYNAMIC_WORLD_CLASSES,
    MMEarthDynamicWorldSegmentationDataModule,
    MMEarthDynamicWorldSegmentationDataset,
    find_one
)

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

SATMAE_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
NUM_CLASSES  = 9
IGNORE_INDEX = 0

BIOME_MACRO_GROUPS = {
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


def load_tile_info_dict(data_root: Path) -> dict:
    tile_info_path = find_one(data_root, "*tile_info.json")
    with open(tile_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_macro_biome(tile_info_dict: dict, tile_id: str) -> str:
    if tile_id not in tile_info_dict:
        return "Unknown"
    info = tile_info_dict[tile_id]
    b_idx = info.get("biome", None)
    if b_idx is None:
        return "Unknown"
    key = int(b_idx) + 1
    return BIOME_MACRO_GROUPS.get(key, "Unknown")


class MMEarthDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: MMEarthDynamicWorldSegmentationDataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = item["image"]
        y = item["mask"].squeeze(0).long()
        sid = str(item["filename"])
        return x, y, sid


class SatMAESegmentationLinearProbe(nn.Module):
    def __init__(
        self,
        model_type: str = "vit_large_patch16",
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 10,
        num_classes: int = 9,
    ):
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


def build_and_load_model(ckpt_path: Path, backbone_weights: Path, device: torch.device):
    print(f"[Loading] Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)

    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("model.", "")
        cleaned[clean_k] = v

    model = SatMAESegmentationLinearProbe(num_classes=NUM_CLASSES)

    if backbone_weights and backbone_weights.exists():
        bb_ckpt = torch.load(backbone_weights, map_location="cpu", weights_only=False)
        bb_state = bb_ckpt.get("model", bb_ckpt.get("state_dict", bb_ckpt))
        model.backbone.load_state_dict(bb_state, strict=False)

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate(model, test_loader, tile_info_dict, test_dataset, device, num_classes=9):
    biome_data = defaultdict(lambda: {"true": [], "pred": [], "probs_max": [], "sample_ids": []})
    all_true, all_pred, all_pmax = [], [], []

    # Map sample index to tile_id
    raw_ds = getattr(test_dataset, "dataset", test_dataset)
    h5 = raw_ds._file()
    metadata = h5["metadata"][:]
    indices = raw_ds.indices

    sample_id_to_biome = {}
    for idx in indices:
        tile_id = metadata[idx]["tile_id"].decode("utf-8")
        filename = metadata[idx]["filename"].decode("utf-8")
        sample_id_to_biome[filename] = get_macro_biome(tile_info_dict, tile_id)

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)
            logits = model(x)
            labels = y.numpy()
            probs = torch.softmax(logits, dim=1)
            probs_max = probs.max(dim=1).values.cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            for i, sid in enumerate(sample_ids):
                group = sample_id_to_biome.get(str(sid), "Unknown")
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
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SatMAE MMEarth Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth"))
    parser.add_argument("--data_root", type=str, default=str(ROOT_DIR / "methods/prithvi/segmentation/mmearth_20k/hardcoded"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

    tile_info_dict = load_tile_info_dict(data_root)

    dm = MMEarthDynamicWorldSegmentationDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=4,
        bands=SATMAE_S2_BANDS,
        image_size=96,
    )
    dm.setup("test")
    test_ds = MMEarthDatasetWrapper(dm.test_dataset)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on MMEarth test split...")
    results = evaluate(model, test_loader, tile_info_dict, test_ds, device, num_classes=NUM_CLASSES)

    print("\n" + "=" * 60)
    print("MMEarth Test Evaluation Results")
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
