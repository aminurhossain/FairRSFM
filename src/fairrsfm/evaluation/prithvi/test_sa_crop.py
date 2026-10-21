#!/usr/bin/env python3
"""
Prithvi EO V2 300M SA-Crop Test Split Evaluator
==============================================
Evaluates trained Prithvi SA-Crop checkpoints (ERM, DBR, GroupDRO, BOLP)
on the official Geo-Bench test split and computes the full segmentation fairness suite.
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

import geobench
import geobench.task
from terratorch.models import PrithviModelFactory
from terratorch.models.backbones.prithvi_vit import HLSBands
from terratorch.models.backbones import prithvi_vit

from evaluation.metrics import (
    compute_miou,
    compute_macro_f1_seg,
    compute_ece_seg,
    compute_eqodds_seg,
    compute_demographic_parity_seg,
    compute_fairness_summary,
)

BANDS = [
    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2,
]
PRITHVI_BAND_INDICES = [1, 2, 3, 7, 8, 9]
NUM_CLASSES  = 10
IGNORE_INDEX = 0


class GeoBenchSegDataset(torch.utils.data.Dataset):
    def __init__(self, task, split: str):
        super().__init__()
        self.dataset      = task.get_dataset(split=split)
        self.band_names   = self.dataset[0].band_names
        self.band_indices = PRITHVI_BAND_INDICES
        self.mean = torch.tensor(prithvi_vit.PRITHVI_V2_MEAN, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(prithvi_vit.PRITHVI_V2_STD,  dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x_data = sample.pack_to_3d(band_names=self.band_names)[0]
        x_data = x_data[:, :, self.band_indices]
        x      = torch.tensor(x_data, dtype=torch.float32).permute(2, 0, 1)
        x      = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        x      = (x - self.mean) / self.std

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
        print("[Model Type] Detected BOLP Checkpoint — attaching segmentation hook")
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


def evaluate(base_model, test_loader, biome_map, device, num_classes=10):
    biome_data = defaultdict(lambda: {"true": [], "pred": [], "probs_max": [], "sample_ids": []})
    all_true, all_pred, all_pmax = [], [], []

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)
            base_model._current_biome_ids = [biome_map.get(str(sid), "Unknown") for sid in sample_ids]
            out = base_model(x)
            logits = out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)

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
    parser = argparse.ArgumentParser(description="Evaluate Prithvi SA-Crop Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "models/Prithvi_EO_V2_300M_TL.pt"))
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

    base_model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on SA-Crop test split...")
    results = evaluate(base_model, test_loader, biome_map, device, num_classes=NUM_CLASSES)

    print("\n" + "=" * 60)
    print("Prithvi SA-Crop Test Evaluation Results")
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
