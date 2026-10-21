#!/usr/bin/env python3
"""
Prithvi EO V2 300M BigEarthNet Test Split Evaluator
==================================================
Evaluates trained Prithvi BigEarthNet checkpoints (ERM, DBR, GroupDRO, BOLP)
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
from sklearn.metrics import f1_score, average_precision_score

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
    compute_subgroup_macro_f1_multilabel,
    compute_per_class_f1_multilabel,
    compute_fairness_summary,
)

BANDS = [
    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2,
]
PRITHVI_BAND_INDICES = [1, 2, 3, 7, 10, 11]
NUM_CLASSES = 43
CLASS_NAMES = [
    "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops", "Pastures",
    "Complex cultivation patterns", "Land principally occupied by agriculture", "Agro-forestry areas",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grasslands", "Moors and heathland",
    "Sclerophyllous vegetation", "Transitional woodland-shrub", "Beaches, dunes, sands", "Bare rocks",
    "Sparsely vegetated areas", "Burnt areas", "Inland marshes", "Peat bogs", "Salt marshes",
    "Salines", "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons", "Estuaries",
    "Sea and ocean", "Non-irrigated arable land", "Permanently irrigated land", "Rice fields",
    "Vineyards", "Fruit trees and berry plantations", "Olive groves", "Pastures", "Annual crops with permanent crops",
    "Complex cultivation patterns", "Land principally occupied by agriculture", "Agro-forestry areas",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
][:43]


class GeoBenchClassifDataset(torch.utils.data.Dataset):
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

        raw_label = sample.label
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        if isinstance(raw_label, (list, np.ndarray)):
            for l in raw_label:
                if int(l) < NUM_CLASSES:
                    label[int(l)] = 1.0
        elif isinstance(raw_label, (int, np.integer)):
            if int(raw_label) < NUM_CLASSES:
                label[int(raw_label)] = 1.0

        sid = str(sample.sample_name)
        return x, torch.tensor(label, dtype=torch.float32), sid


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
        task="classification",
        backbone="prithvi_eo_v2_300_tl",
        decoder="MLPDecoder",
        bands=BANDS,
        num_classes=NUM_CLASSES,
        pretrained=True if backbone_weights.exists() else False,
        backbone_ckpt_path=str(backbone_weights) if backbone_weights.exists() else None
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    is_bolp = any("projector" in k or "Pk" in k for k in sd.keys())
    if is_bolp:
        print("[Model Type] Detected BOLP Checkpoint — attaching projection hook")
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
            if isinstance(output, (list, tuple)):
                last = output[-1]
            else:
                last = output
            if last.dim() > 2:
                last = last.mean(dim=list(range(1, last.dim() - 1))) if last.dim() == 3 else last.mean(dim=[-2, -1])
            last = projector(last)
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


def evaluate(base_model, test_loader, biome_map, device, num_classes=43, threshold=0.5):
    biome_data = defaultdict(lambda: {"true": [], "probs": [], "sample_ids": []})
    all_true, all_probs = [], []

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x = x.to(device)
            out = base_model(x)
            logits = out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)
            probs  = torch.sigmoid(logits.float()).cpu().numpy()
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

    try:
        global_map = float(average_precision_score(all_true, all_probs, average="macro"))
    except Exception:
        global_map = float("nan")

    biome_macro_f1, biome_counts = {}, {}
    per_group_metrics = {}

    for b in sorted(biome_data.keys()):
        yt = np.array(biome_data[b]["true"])
        pr = np.array(biome_data[b]["probs"])
        if len(yt) == 0:
            continue
        yp = (pr >= threshold).astype(int)
        b_macro = compute_subgroup_macro_f1_multilabel(yt, yp)
        b_micro = float(f1_score(yt, yp, average="micro", zero_division=0))
        try:
            b_map = float(average_precision_score(yt, pr, average="macro"))
        except Exception:
            b_map = float("nan")

        biome_macro_f1[b] = b_macro
        biome_counts[b]   = len(yt)
        per_group_metrics[b] = {
            "macro_f1": b_macro,
            "micro_f1": b_micro,
            "map": b_map,
            "count": len(yt),
        }

    fairness = compute_fairness_summary(biome_macro_f1)
    per_class_f1_overall, per_group_class_f1 = compute_per_class_f1_multilabel(all_true, all_pred, biome_data, num_classes)

    return {
        "global_macro_f1": global_macro_f1,
        "global_micro_f1": global_micro_f1,
        "global_map": global_map,
        "worst_case_macro_f1": fairness["worst_case"],
        "best_case_macro_f1": fairness["best_case"],
        "macro_f1_disparity": fairness["disparity"],
        "performance_equity_ratio": fairness["performance_equity_ratio"],
        "gini_coefficient": fairness["gini_coefficient"],
        "normalized_fairness_range": fairness["normalized_fairness_range"],
        "per_biome_macro_f1": biome_macro_f1,
        "per_group_metrics": per_group_metrics,
        "per_class_f1": per_class_f1_overall,
        "per_group_class_f1": per_group_class_f1,
        "biome_sample_counts": biome_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prithvi BigEarthNet Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--backbone_ckpt", type=str, default=str(ROOT_DIR / "models/Prithvi_EO_V2_300M_TL.pt"))
    parser.add_argument("--data_dir", type=str, default=None, help="Path to GeoBench data root")
    parser.add_argument("--biome_json", type=str, default=str(ROOT_DIR / "configs/bigearthnet_biome_mapping.json"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
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
    for t in geobench.task_iterator(benchmark_name="classification_v1.0"):
        if "bigearthnet" in t.dataset_name.lower():
            task = t
            break
    assert task is not None, "BigEarthNet task not found in GeoBench"

    test_ds = GeoBenchClassifDataset(task, "test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    biome_map = load_biome_map(Path(args.biome_json))

    base_model = build_and_load_model(Path(args.ckpt), Path(args.backbone_ckpt), device)

    print("\nRunning evaluation on BigEarthNet test split...")
    results = evaluate(base_model, test_loader, biome_map, device, num_classes=NUM_CLASSES, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("Prithvi BigEarthNet Test Evaluation Results")
    print("=" * 60)
    print(f"Overall Macro F1 : {results['global_macro_f1']:.4f}")
    print(f"Overall Micro F1 : {results['global_micro_f1']:.4f}")
    print(f"Overall mAP      : {results['global_map']:.4f}")
    print(f"Worst-Group F1   : {results['worst_case_macro_f1']:.4f}")
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
