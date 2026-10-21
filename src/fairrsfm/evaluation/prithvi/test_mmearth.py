#!/usr/bin/env python3
"""
Prithvi EO V2 300M MMEarth-20k Test Split Evaluator
==================================================
Evaluates trained Prithvi MMEarth-20k checkpoints (ERM, DBR, GroupDRO, BOLP)
and computes the full segmentation fairness suite.
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

try:
    import pyproj
    os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())
    os.environ.setdefault("PROJ_LIB", pyproj.datadir.get_data_dir())
except Exception:
    pass

from terratorch.models import PrithviModelFactory
from terratorch.models.backbones.prithvi_vit import HLSBands
from terratorch.models.backbones import prithvi_vit

from methods.prithvi.segmentation.mmearth_20k.mmearth_dynamic_world_datamodule import (
    MMEarthDynamicWorldDataModule,
    DW_CLASS_NAMES,
    PRITHVI_DW_BAND_INDICES,
)
from evaluation.metrics import (
    compute_miou,
    compute_macro_f1_seg,
    compute_fairness_summary,
)

BANDS = [
    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2,
]
NUM_CLASSES  = len(DW_CLASS_NAMES)
IGNORE_INDEX = 255


class MMEarthDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, inner_dataset):
        self.dataset = inner_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = item["image"]
        y = item["mask"].squeeze(0).long()
        sid = str(item["filename"])
        return x, y, sid


class BOLPProjector(nn.Module):
    def __init__(self, global_centroid: torch.Tensor, Pk: torch.Tensor,
                 mu_table: torch.Tensor = None, biome_index: dict = None,
                 k: int = 1, skip_residualization: bool = True):
        super().__init__()
        self.skip_residualization = skip_residualization
        self.k = k
        self.biome_index = biome_index or {}
        self.register_buffer("mu0", global_centroid)
        self.register_buffer("Pk", Pk)
        if mu_table is not None:
            self.register_buffer("mu_table", mu_table)
        else:
            self.mu_table = None

    def forward(self, z: torch.Tensor, biome_ids: list = None) -> torch.Tensor:
        z_tilde = z.clone()
        if not self.skip_residualization and biome_ids is not None and self.mu_table is not None:
            for i, biome in enumerate(biome_ids):
                idx = self.biome_index.get(biome, None)
                if idx is not None:
                    z_tilde[i] = z[i] - self.mu_table[idx] + self.mu0
        return z_tilde @ self.Pk.T


def build_and_load_model(ckpt_path: Path, backbone_weights: Path, device: torch.device):
    print(f"[Loading] Checkpoint: {ckpt_path}")
    factory = PrithviModelFactory()
    base_model = factory.build_model(
        task="segmentation",
        backbone="prithvi_eo_v2_300_tl",
        decoder="FCNDecoder",
        bands=BANDS,
        num_classes=NUM_CLASSES,
        pretrained=True if backbone_weights.exists() else False,
        backbone_ckpt_path=str(backbone_weights) if backbone_weights.exists() else None,
        head_dropout=0.1,
        neck_indices=[5, 11, 17, 23],
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    is_bolp = any("projector" in k or "Pk" in k for k in sd.keys())
    if is_bolp:
        print("[Model Type] Detected BOLP Checkpoint — attaching MMEarth segmentation hook")
        mu0 = sd.get("projector.mu0", torch.zeros(1024))
        Pk = sd.get("projector.Pk", None)
        if Pk is None:
            for k in sd.keys():
                if "Pk" in k:
                    Pk = sd[k]
                    break
        mu_table = sd.get("projector.mu_table", None)
        biome_index = ckpt.get("bolp_biome_index", {})
        skip_resid = ckpt.get("bolp_skip_residualization", True)
        bolp_k = ckpt.get("bolp_k", Pk.shape[0] if Pk is not None else 1)

        projector = BOLPProjector(
            global_centroid=mu0, Pk=Pk, mu_table=mu_table,
            biome_index=biome_index, k=bolp_k, skip_residualization=skip_resid
        )
        projector.to(device)

        def hook_fn(module, input, output):
            biome_ids = getattr(base_model, "_current_biome_ids", None)
            if isinstance(output, (list, tuple)):
                last = output[-1]
            else:
                last = output
            if last.dim() == 3:
                z_avg = last.mean(dim=1)
                z_hat = projector(z_avg, biome_ids)
                last = last + (z_hat - z_avg).unsqueeze(1)
            else:
                last = projector(last, biome_ids)
            if isinstance(output, (list, tuple)):
                out_list = list(output)
                out_list[-1] = last
                return type(output)(out_list)
            return last

        base_model.encoder.register_forward_hook(hook_fn)
    else:
        print("[Model Type] Detected Standard Head (ERM / GroupDRO / DBR)")

    model_sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    base_model.load_state_dict(model_sd, strict=False)
    base_model.to(device)
    base_model.eval()

    return base_model


def evaluate(base_model, test_loader, tile_info_dict, test_ds, device, num_classes=NUM_CLASSES):
    biome_data = defaultdict(lambda: {"true": [], "pred": []})
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)

            biome_ids = []
            for sid in sample_ids:
                info = tile_info_dict.get(sid, {})
                biome_ids.append(info.get("biome_desc", info.get("biome", "Unknown")))
            base_model._current_biome_ids = biome_ids

            out = base_model(x)
            logits = out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y.numpy()

            for i, sid in enumerate(sample_ids):
                biome = biome_ids[i]
                if biome != "Unknown":
                    biome_data[biome]["true"].append(labels[i].reshape(-1))
                    biome_data[biome]["pred"].append(preds[i].reshape(-1))

                all_true.append(labels[i].reshape(-1))
                all_pred.append(preds[i].reshape(-1))

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    global_miou = compute_miou(all_true, all_pred, num_classes, ignore_index=IGNORE_INDEX)
    global_f1   = compute_macro_f1_seg(all_true, all_pred, num_classes, ignore_index=IGNORE_INDEX)

    biome_mious, biome_f1s, biome_counts = {}, {}, {}
    per_group_metrics = {}

    for b in sorted(biome_data.keys()):
        yt = np.concatenate(biome_data[b]["true"])
        yp = np.concatenate(biome_data[b]["pred"])
        if len(yt) == 0:
            continue
        b_miou = compute_miou(yt, yp, num_classes, ignore_index=IGNORE_INDEX)
        b_f1   = compute_macro_f1_seg(yt, yp, num_classes, ignore_index=IGNORE_INDEX)

        biome_mious[b]  = b_miou
        biome_f1s[b]    = b_f1
        biome_counts[b] = len(biome_data[b]["true"])
        per_group_metrics[b] = {
            "miou": b_miou,
            "macro_f1": b_f1,
            "tile_count": len(biome_data[b]["true"]),
        }

    fairness = compute_fairness_summary(biome_mious)

    return {
        "global_miou": global_miou,
        "global_macro_f1": global_f1,
        "worst_case_miou": fairness["worst_case"],
        "best_case_miou": fairness["best_case"],
        "miou_disparity": fairness["disparity"],
        "performance_equity_ratio": fairness["performance_equity_ratio"],
        "gini_coefficient": fairness["gini_coefficient"],
        "normalized_fairness_range": fairness["normalized_fairness_range"],
        "per_biome_miou": biome_mious,
        "per_biome_f1": biome_f1s,
        "per_group_metrics": per_group_metrics,
        "biome_tile_counts": biome_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prithvi MMEarth Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "models/Prithvi_EO_V2_300M_TL.pt"))
    parser.add_argument("--data_dir", type=str, default="/data1/aminur/dataset/mmearth", help="Path to MMEarth data root")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_dir)
    tile_info_path = data_root / "tile_info.json"
    tile_info_dict = {}
    if tile_info_path.exists():
        with open(tile_info_path) as f:
            tile_info_dict = json.load(f)

    dm = MMEarthDynamicWorldDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=4,
        bands=PRITHVI_DW_BAND_INDICES,
        image_size=112,
    )
    dm.setup("test")
    test_ds = MMEarthDatasetWrapper(dm.test_dataset)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    base_model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on Prithvi MMEarth test split...")
    results = evaluate(base_model, test_loader, tile_info_dict, test_ds, device, num_classes=NUM_CLASSES)

    print("\n" + "=" * 60)
    print("Prithvi MMEarth Test Evaluation Results")
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
