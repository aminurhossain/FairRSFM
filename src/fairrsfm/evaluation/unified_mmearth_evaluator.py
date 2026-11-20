"""
Unified MMEarth-20K Evaluation & Diagnostic Suite
=================================================
Comprehensive, multi-granularity evaluation suite from a 14x9x9 confusion matrix:
1. Canonical Paper Metrics Suite (matching paper.tex / main_tables_new.tex):
   - Test Score (mIoU), Micro Score (Micro F1 / Pixel Acc), Macro F1
   - Worst-case Biome (M_worst), Best-case Biome (M_best)
   - Performance Equity Ratio (PER), Normalized Fairness Range (NFR)
   - Cross-group Standard Deviation (Std), Gini Coefficient
   - Equalized Odds Disparity (EOdd), Demographic Parity Disparity (DPM)
   - Expected Calibration Error (ECE)
2. Multi-Threshold Per-Biome Metrics:
   - Present mIoU (GT>0), Support mIoU (>=1k, >=10k, >=50k), Macro F1, FWIoU, Pixel Acc
3. Class-Controlled Shared-5-Class Metric (Water, Trees, Grass, Shrub, Bare)
4. Multi-Granularity Rollups:
   - 14 Granular Biomes (True regional audit)
   - 6 WWF Macro-Groups (Phenological/climate regimes)
   - 4 Balanced Land-Cover Clusters (Equalized class-support view)
   - Global Dataset-Wide Pooled Metrics
5. Single-Class Geographic Disparity Spread across all 9 Dynamic World classes
6. Full Train & Test Ground Truth Pixel Count Matrices (14 biomes x 9 classes)
7. Standardized Output Artifacts:
   - test_selected.json (Canonical schema matching reporting.py)
   - test_results.json (Full multi-granularity metrics dictionary)
   - test_summary.md (Comprehensive markdown report for paper tables)
   - test_biome_metrics.csv (Tabular per-biome metrics)
   - conf_mat_14x9x9.npy (Raw integer confusion matrix)
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import h5py

BIOME_NAMES = [
    "Tropical and Subtropical Moist Broadleaf Forests",
    "Tropical and Subtropical Dry Broadleaf Forests",
    "Tropical and Subtropical Coniferous Forests",
    "Temperate Broadleaf and Mixed Forests",
    "Temperate Coniferous Forests",
    "Boreal forests/Taiga",
    "Tropical and Subtropical Grasslands, Savannas",
    "Temperate Grasslands, Savannas, and Shrublands",
    "Flooded Grasslands and Savannas",
    "Montane Grasslands and Shrublands",
    "Tundra",
    "Mediterranean Forests, Woodlands, and Scrub",
    "Deserts and Xeric Shrublands",
    "Mangroves",
]

DYNAMIC_WORLD_CLASSES = [
    "Water",
    "Trees",
    "Grass",
    "Flooded Vegetation",
    "Crops",
    "Shrub and Scrub",
    "Built",
    "Bare",
    "Snow and Ice",
]

# 6 Macro Groups mapping
MACRO_GROUPS = {
    "High Biomass": [0, 1, 2],
    "High Amplitude Phenological": [3, 4, 5],
    "Transitional Herbaceous & Scrub": [6, 7, 11],
    "Hydrologically Modulated": [8, 13],
    "Cryospheric & Short-Cycle": [9, 10],
    "Xeric & Mineralogical": [12],
}

# 4 Balanced Land-Cover Clusters mapping
CLUSTERS_4 = {
    "Cluster 1 (Alpine & Savannas)": [9, 6],
    "Cluster 2 (Cold & Hardwood)": [10, 3],
    "Cluster 3 (Tropical & Wet-Dry)": [0, 1, 8, 11],
    "Cluster 4 (Coniferous, Arid & Coastal)": [2, 4, 5, 7, 12, 13],
}

SHARED_5_CLASSES = [0, 1, 2, 5, 7]  # Water, Trees, Grass, Shrub, Bare


def compute_metrics_from_confmat(cm: np.ndarray, num_classes: int = 9) -> Dict[str, Any]:
    """
    Computes all standard segmentation and fairness metrics from a (C, C) confusion matrix.
    Rows = True, Cols = Pred.
    """
    tp = np.diag(cm).astype(np.float64)
    gt_pixels = cm.sum(axis=1).astype(np.float64)
    pred_pixels = cm.sum(axis=0).astype(np.float64)
    union = gt_pixels + pred_pixels - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union > 0, tp / union, np.nan)
        precisions = np.where(pred_pixels > 0, tp / pred_pixels, np.nan)
        recalls = np.where(gt_pixels > 0, tp / gt_pixels, np.nan)
        denom_f1 = gt_pixels + pred_pixels
        f1s = np.where(denom_f1 > 0, (2.0 * tp) / denom_f1, np.nan)

    valid_pixels = float(gt_pixels.sum())
    correct_pixels = float(tp.sum())
    pixel_acc = (correct_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0

    # FWIoU
    valid_gt = gt_pixels.sum()
    if valid_gt > 0:
        fwiou = float(np.nansum((gt_pixels / valid_gt) * np.nan_to_num(ious, nan=0.0)) * 100.0)
    else:
        fwiou = 0.0

    # mIoUs at different support thresholds
    def get_miou(mask):
        selected = ious[mask]
        valid_sel = selected[~np.isnan(selected)]
        return float(np.mean(valid_sel) * 100.0) if len(valid_sel) > 0 else 0.0

    miou_std = get_miou(union > 0)
    miou_gt0 = get_miou(gt_pixels > 0)
    miou_gt1k = get_miou(gt_pixels >= 1000)
    miou_gt10k = get_miou(gt_pixels >= 10000)
    miou_gt50k = get_miou(gt_pixels >= 50000)

    # All 9 classes mIoU (strictly dividing by 9; zero-filling false positive predictions where union > 0)
    ious_all9 = np.nan_to_num(ious, nan=0.0)
    miou_all9 = float(np.mean(ious_all9) * 100.0)

    # Shared 5 classes mIoU
    shared_ious = ious[SHARED_5_CLASSES]
    valid_shared = shared_ious[~np.isnan(shared_ious)]
    miou_shared5 = float(np.mean(valid_shared) * 100.0) if len(valid_shared) > 0 else 0.0

    # Macro F1 (over present classes vs all 9 classes)
    present_mask = gt_pixels > 0
    valid_f1 = f1s[present_mask]
    valid_f1 = valid_f1[~np.isnan(valid_f1)]
    macro_f1 = float(np.mean(valid_f1) * 100.0) if len(valid_f1) > 0 else 0.0
    f1s_all9 = np.nan_to_num(f1s, nan=0.0)
    macro_f1_all9 = float(np.mean(f1s_all9) * 100.0)

    # Class frequency distribution (%)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_freqs = np.where(valid_pixels > 0, (gt_pixels / valid_pixels) * 100.0, 0.0)

    # Class-level breakdown dict
    class_details = {}
    for c_idx, c_name in enumerate(DYNAMIC_WORLD_CLASSES):
        class_details[c_name] = {
            "iou": float(ious[c_idx]) if not np.isnan(ious[c_idx]) else None,
            "f1": float(f1s[c_idx]) if not np.isnan(f1s[c_idx]) else None,
            "precision": float(precisions[c_idx]) if not np.isnan(precisions[c_idx]) else None,
            "recall": float(recalls[c_idx]) if not np.isnan(recalls[c_idx]) else None,
            "tp": int(tp[c_idx]),
            "gt_pixels": int(gt_pixels[c_idx]),
            "pred_pixels": int(pred_pixels[c_idx]),
            "union": int(union[c_idx]),
            "frequency_pct": float(class_freqs[c_idx]),
        }

    return {
        "valid_pixels": int(valid_pixels),
        "correct_pixels": int(correct_pixels),
        "pixel_accuracy": pixel_acc,
        "fwiou": fwiou,
        "miou_standard": miou_std,
        "miou_present": miou_gt0,
        "miou_all9": miou_all9,
        "macro_f1_all9": macro_f1_all9,
        "miou_1k": miou_gt1k,
        "miou_10k": miou_gt10k,
        "miou_50k": miou_gt50k,
        "miou_shared5": miou_shared5,
        "macro_f1": macro_f1,
        "class_details": class_details,
        "ious": [float(x) if not np.isnan(x) else None for x in ious],
        "f1s": [float(x) if not np.isnan(x) else None for x in f1s],
        "class_frequencies": [float(x) for x in class_freqs],
        "class_frequency_pct": {DYNAMIC_WORLD_CLASSES[c]: float(class_freqs[c]) for c in range(num_classes)},
    }


def compute_gini(values: List[float]) -> float:
    """Canonical Gini coefficient calculation from metrics.py / reporting.py."""
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    array = np.sort(np.array(vals, dtype=np.float64))
    n = len(array)
    sum_val = float(np.sum(array))
    if sum_val == 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * array) - (n + 1) * sum_val) / (n * sum_val))


def compute_eod_and_dpm(cm_groups: np.ndarray, num_classes: int = 9) -> Dict[str, float]:
    """
    Computes Equalized Odds Disparity (EOdd) and Demographic Parity Disparity (DPM)
    across groups from a (G, C, C) confusion matrix. Rows = True, Cols = Pred.
    Matches compute_eqodds_seg and compute_demographic_parity_seg from metrics.py.
    """
    G, C, _ = cm_groups.shape
    if G < 2:
        return {"eqodds": 0.0, "dpm": 0.0}

    J = np.zeros((G, C), dtype=np.float64)
    PPR = np.zeros((G, C), dtype=np.float64)

    for g in range(G):
        cm_g = cm_groups[g].astype(np.float64)
        tp = np.diag(cm_g)
        gt = cm_g.sum(axis=1)
        pred = cm_g.sum(axis=0)
        total = float(cm_g.sum())

        fp = pred - tp
        tn = (total - gt) - fp

        with np.errstate(divide="ignore", invalid="ignore"):
            tpr = np.where(gt > 0, tp / gt, np.nan)
            denom_fpr = fp + tn
            fpr = np.where(denom_fpr > 0, fp / denom_fpr, np.nan)
            J[g] = tpr - fpr
            PPR[g] = np.where(total > 0, pred / total, 0.0)

    worst_eod = 0.0
    worst_dpm = 0.0

    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            valid_j = ~np.isnan(J[g1]) & ~np.isnan(J[g2])
            if np.any(valid_j):
                eod_diff = float(np.mean(np.abs(J[g1, valid_j] - J[g2, valid_j])))
                worst_eod = max(worst_eod, eod_diff)
            dpm_diff = float(np.mean(np.abs(PPR[g1] - PPR[g2])))
            worst_dpm = max(worst_dpm, dpm_diff)

    return {
        "eqodds": float(worst_eod),
        "dpm": float(worst_dpm),
    }


def compute_group_fairness(group_scores: Dict[str, float]) -> Dict[str, float]:
    """Computes equity metrics from a mapping of group_name -> score."""
    vals = np.array(list(group_scores.values()), dtype=np.float64)
    if len(vals) == 0:
        return {}
    best = float(np.max(vals))
    worst = float(np.min(vals))
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    spread = best - worst
    per = (worst / best) if best > 0 else 0.0
    nfr = (spread / (mean + 1e-8))
    gini = compute_gini(list(vals))

    return {
        "mean": mean,
        "worst": worst,
        "best": best,
        "spread": spread,
        "per": per,
        "nfr": nfr,
        "std": std,
        "gini": gini,
    }


def get_ground_truth_pixel_counts(data_root: Path) -> Dict[str, np.ndarray]:
    """Reads exact per-biome GT pixel counts from HDF5 file for Train and Test splits."""
    h5_candidates = list(data_root.glob("*.h5"))
    splits_candidates = list(data_root.glob("*splits.json"))
    if not h5_candidates or not splits_candidates:
        return {"train": np.zeros((14, 9), dtype=np.int64), "test": np.zeros((14, 9), dtype=np.int64)}

    h5_path = h5_candidates[0]
    splits_path = splits_candidates[0]

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_idx = np.array(splits["train"])
    test_idx = np.array(splits["test"])

    with h5py.File(h5_path, "r") as hf:
        biomes_onehot = hf["biome"][:]
        biome_ids = np.argmax(biomes_onehot, axis=1)
        dw = hf["dynamic_world"]

        train_counts = np.zeros((14, 9), dtype=np.int64)
        test_counts = np.zeros((14, 9), dtype=np.int64)

        for b in range(14):
            tr_b = train_idx[biome_ids[train_idx] == b]
            te_b = test_idx[biome_ids[test_idx] == b]
            tr_masks = dw[tr_b, 0]
            te_masks = dw[te_b, 0]
            for c in range(9):
                train_counts[b, c] = int(np.count_nonzero(tr_masks == (c + 1)))
                test_counts[b, c] = int(np.count_nonzero(te_masks == (c + 1)))

    return {"train": train_counts, "test": test_counts}


def generate_markdown_summary(results: Dict[str, Any], output_path: Path) -> None:
    """Generates a publication-ready test_summary.md file combining all metrics."""
    b_res = results["biomes"]
    g_res = results["macro_groups"]
    c_res = results["clusters_4"]
    f_res = results["fairness"]
    can = results["canonical"]

    md = []
    md.append("# Unified MMEarth-20K Evaluation & Diagnostic Summary\n")
    md.append(f"**Dataset:** Balanced MMEarth-20K ($N=20,160$ tiles, 14 Biomes, 9 Dynamic World Classes)  ")
    md.append(f"**Evaluated Split:** Test Split ($N=2,016$ tiles, 144 tiles per biome)  \n")
    md.append("---\n")

    # 1. Canonical Paper Metrics Table
    md.append("## 1. Canonical Paper Benchmark Metrics (14 Biomes)\n")
    md.append("*Matches the exact primary result columns from manuscript table (`main_tables_new.tex`).*\n")
    md.append(r"| Metric Dimension | Test Score (mIoU) | Micro Score (Acc) | Macro F1 | $M_{\text{worst}}$ | PER $\uparrow$ | Gini $\downarrow$ | NFR $\downarrow$ | Std ($\sigma$) $\downarrow$ | EOdd $\downarrow$ | DPM $\downarrow$ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(
        f"| **14-Biome Global / Equity** | "
        f"**{can['overall_miou']*100.0:.2f}%** | "
        f"**{can['overall_micro_f1']*100.0:.2f}%** | "
        f"**{can['overall_macro_f1']*100.0:.2f}%** | "
        f"**{can['worst_case_miou']*100.0:.2f}%** | "
        f"**{can['performance_equity_ratio']:.4f}** | "
        f"**{can['gini_coefficient']:.4f}** | "
        f"**{can['normalized_fairness_range']:.4f}** | "
        f"**{can['group_std']*100.0:.2f}%** | "
        f"**{can['equalized_odds_disparity']:.4f}** | "
        f"**{can['demographic_parity_disparity']:.4f}** |"
    )
    md.append("\n---\n")

    # 2. Table 1B: Class-Controlled Standardization & Robustness Audit
    md.append("## 2. Table 1B: Class-Controlled Standardization & Robustness Audit (Reviewer ADVW / AC Analysis)\n")
    md.append("*Directly addressing reviewer concerns on whether biome disparities reflect class difficulty vs. pure geographic domain shift.*\n")
    md.append(r"| Metric Formulation | Core Rationale & Mathematical Definition | Dataset Mean | Worst-Case Group | Best-Case Group | Disparity Spread | PER $\uparrow$ | Std ($\sigma$) $\downarrow$ |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    audit_rows = [
        (
            "Raw Present mIoU",
            r"Standard unweighted mIoU over present classes: $\frac{1}{|C|} \sum \text{IoU}_c$",
            f_res["biomes_present_miou"],
            "%",
            "pp",
        ),
        (
            "Standardized Counterfactual mIoU",
            r"Reweighted by Global Marginal Class Distribution: $\sum w_c \text{IoU}_c / \sum w_c$",
            f_res["biomes_std_miou"],
            "%",
            "pp",
        ),
        (
            "Difficulty-Normalized Relative Index",
            r"Normalized by global class difficulty: $\frac{1}{|C|} \sum \frac{\text{IoU}_{b, c}}{\text{Global-IoU}_c}$",
            f_res["biomes_rel_score"],
            "",
            "",
        ),
        (
            "Shared-5 Ubiquitous Classes mIoU",
            r"Uniform 20% weight on 5 planet-wide shared classes (Water, Trees, Grass, Shrub, Bare)",
            f_res["biomes_shared5_miou"],
            "%",
            "pp",
        ),
        (
            "Robust Support (>=50k px) mIoU",
            r"Restricted to classes with reliable geographic sample support ($\ge 50\text{k}$ px)",
            f_res["biomes_50k_miou"],
            "%",
            "pp",
        ),
        (
            "FWIoU (Frequency-Weighted)",
            r"Locally weighted by biome ground-truth pixel frequencies: $\sum \frac{\text{GT}_c}{N_b} \text{IoU}_c$",
            f_res["biomes_fwiou"],
            "%",
            "pp",
        ),
    ]

    for label, rat, stat, u_val, u_spd in audit_rows:
        if u_val == "%":
            md.append(
                f"| **{label}** | {rat} | **{stat['mean']:.2f}%** | **{stat['worst']:.2f}%** | **{stat['best']:.2f}%** | **{stat['spread']:.2f} {u_spd}** | **{stat['per']:.4f}** | **{stat['std']:.2f}%** |"
            )
        else:
            md.append(
                f"| **{label}** | {rat} | **{stat['mean']:.4f}** | **{stat['worst']:.4f}** | **{stat['best']:.4f}** | **{stat['spread']:.4f}** | **{stat['per']:.4f}** | **{stat['std']:.4f}** |"
            )
    md.append("\n---\n")

    # 3. Multi-Threshold Headline Metrics (14 Biomes)
    md.append("## 3. Multi-Threshold Headline Metrics (14 Biomes)\n")
    f14 = f_res["biomes_present_miou"]
    f14_std = f_res["biomes_std_miou"]
    f14_50k = f_res["biomes_50k_miou"]
    f14_fw = f_res["biomes_fwiou"]
    f14_f1 = f_res["biomes_macro_f1"]

    md.append(r"| Metric Dimension | Present mIoU (GT>0) | Standardized mIoU | Robust Support (>=50k px) | Macro F1 | FWIoU (Pixel-Weighted) | Pixel Accuracy |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **Dataset Mean across Biomes** | **{f14['mean']:.2f}%** | **{f14_std['mean']:.2f}%** | **{f14_50k['mean']:.2f}%** | **{f14_f1['mean']:.2f}%** | **{f14_fw['mean']:.2f}%** | **{results['global']['pixel_accuracy']:.2f}%** |")
    md.append(f"| **Worst-Case Biome** | **{f14['worst']:.2f}%** | **{f14_std['worst']:.2f}%** | **{f14_50k['worst']:.2f}%** | **{f14_f1['worst']:.2f}%** | **{f14_fw['worst']:.2f}%** | — |")
    md.append(f"| **Best-Case Biome** | **{f14['best']:.2f}%** | **{f14_std['best']:.2f}%** | **{f14_50k['best']:.2f}%** | **{f14_f1['best']:.2f}%** | **{f14_fw['best']:.2f}%** | — |")
    md.append(f"| **Disparity Spread (B - W)** | **{f14['spread']:.2f} pp** | **{f14_std['spread']:.2f} pp** | **{f14_50k['spread']:.2f} pp** | **{f14_f1['spread']:.2f} pp** | **{f14_fw['spread']:.2f} pp** | — |")
    md.append(f"| **Performance Equity Ratio (PER)** | **{f14['per']:.4f}** | **{f14_std['per']:.4f}** | **{f14_50k['per']:.4f}** | **{f14_f1['per']:.4f}** | **{f14_fw['per']:.4f}** | — |")
    md.append(f"| **Cross-Biome Standard Deviation (Std)** | **{f14['std']:.2f}%** | **{f14_std['std']:.2f}%** | **{f14_50k['std']:.2f}%** | **{f14_f1['std']:.2f}%** | **{f14_fw['std']:.2f}%** | — |\n")

    md.append("---\n")
    # 4. Table 1: Multi-Threshold 14-Biome Breakdown
    md.append("## 4. Table 1: Multi-Threshold 14-Biome Performance Breakdown\n")
    md.append(r"| # | Terrestrial Biome Name | Present mIoU | All-9 mIoU | Std mIoU | Rel Index | Macro F1 | mIoU (>=50k) | Shared-5 mIoU | FWIoU | Pixel Acc |")
    md.append("| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for b_id in range(14):
        bd = b_res[str(b_id)]
        md.append(
            f"| {b_id} | **{bd['name']}** | {bd['miou_present']:.2f}% | {bd['miou_all9']:.2f}% | {bd['std_miou']:.2f}% | {bd['rel_score']:.4f} | {bd['macro_f1']:.2f}% | {bd['miou_50k']:.2f}% | {bd['miou_shared5']:.2f}% | {bd['fwiou']:.2f}% | {bd['pixel_accuracy']:.2f}% |"
        )
    md.append("\n---\n")

    # 5. Table 2: Multi-Granularity Rollup
    md.append("## 5. Table 2: Multi-Granularity Rollup (Simpson's Paradox Analysis)\n")
    md.append(r"| Granularity Strata | Strata Count | Mean mIoU | Worst Group | Best Group | Disparity Spread | Equity Ratio (PER) | Group Std |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, f_dict, cnt in [
        ("4 Balanced Clusters", f_res["clusters_4_present_miou"], 4),
        ("6 WWF Macro-Groups", f_res["macro_groups_present_miou"], 6),
        ("14 Granular Biomes", f_res["biomes_present_miou"], 14),
    ]:
        md.append(
            f"| **{name}** | {cnt} | **{f_dict['mean']:.2f}%** | **{f_dict['worst']:.2f}%** | **{f_dict['best']:.2f}%** | **{f_dict['spread']:.2f} pp** | **{f_dict['per']:.4f}** | **{f_dict['std']:.2f}%** |"
        )
    md.append("\n---\n")

    # 6. Table 3: 4 Balanced Land-Cover Clusters Breakdown
    md.append("## 6. Table 3: 4 Balanced Land-Cover Clusters Breakdown\n")
    md.append(r"| Cluster Name | Present mIoU | Std mIoU | Rel Index | Macro F1 | mIoU (>=50k) | FWIoU | Pixel Acc | Included Biomes |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for c_name, b_indices in CLUSTERS_4.items():
        cd = c_res[c_name]
        b_str = ", ".join([BIOME_NAMES[b][:20] for b in b_indices])
        md.append(
            f"| **{c_name}** | {cd['miou_present']:.2f}% | {cd['std_miou']:.2f}% | {cd['rel_score']:.4f} | {cd['macro_f1']:.2f}% | {cd['miou_50k']:.2f}% | {cd['fwiou']:.2f}% | {cd['pixel_accuracy']:.2f}% | {b_str} |"
        )
    md.append("\n---\n")

    # 7. Table 4: 6 WWF Macro-Groups Breakdown
    md.append("## 7. Table 4: 6 WWF Macro-Groups Breakdown\n")
    md.append(r"| Macro Group Name | Present mIoU | Std mIoU | Rel Index | Macro F1 | mIoU (>=50k) | FWIoU | Pixel Acc | Included Biomes |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for g_name, b_indices in MACRO_GROUPS.items():
        gd = g_res[g_name]
        b_str = ", ".join([BIOME_NAMES[b][:20] for b in b_indices])
        md.append(
            f"| **{g_name}** | {gd['miou_present']:.2f}% | {gd['std_miou']:.2f}% | {gd['rel_score']:.4f} | {gd['macro_f1']:.2f}% | {gd['miou_50k']:.2f}% | {gd['fwiou']:.2f}% | {gd['pixel_accuracy']:.2f}% | {b_str} |"
        )
    md.append("\n---\n")

    # 8. Table 5: Controlled Single-Class Geographic Disparity Spread
    md.append("## 8. Table 5: Controlled Single-Class Geographic Disparity Spread\n")
    md.append("| Semantic Class | Global IoU | Worst Biome (IoU) | Best Biome (IoU) | Disparity Gap (pp) | Class PER |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for c_idx, c_name in enumerate(DYNAMIC_WORLD_CLASSES):
        c_ious = [b_res[str(b)]["class_details"][c_name]["iou"] for b in range(14)]
        valid_ious = [(b, iou) for b, iou in enumerate(c_ious) if iou is not None]
        if valid_ious:
            w_b, w_val = min(valid_ious, key=lambda x: x[1])
            b_b, b_val = max(valid_ious, key=lambda x: x[1])
            gap = (b_val - w_val) * 100.0
            c_per = (w_val / b_val) if b_val > 0 else 0.0
            glob_iou = results["global"]["class_details"][c_name]["iou"]
            glob_str = f"{glob_iou*100.0:.2f}%" if glob_iou is not None else "—"
            md.append(
                f"| **{c_name}** | {glob_str} | {BIOME_NAMES[w_b][:28]} ({w_val*100.0:.2f}%) | {BIOME_NAMES[b_b][:28]} ({b_val*100.0:.2f}%) | **{gap:.2f} pp** | **{c_per:.4f}** |"
            )
    md.append("\n---\n")

    # 9. Table 6: Per-Biome Ground-Truth Class Frequency Distribution (%)
    md.append("## 9. Table 6: Per-Biome Ground-Truth Class Frequency Distribution (%) (Reviewer ADVW / AC Heatmap Data)\n")
    md.append("*Documenting the normalized percentage frequency of each class in test tiles to support heatmap visualizations.*\n")
    md.append("| # | Entity Name | Water | Trees | Grass | Flooded | Crops | Shrub | Built | Bare | Snow | Total Pixels |")
    md.append("| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    # 14 Biomes
    for b in range(14):
        bd = b_res[str(b)]
        freq_strs = " | ".join([f"{f:.2f}%" for f in bd["class_frequencies"]])
        md.append(f"| {b} | **{BIOME_NAMES[b][:25]}** | {freq_strs} | {bd['valid_pixels']:,d} |")

    # 6 Macro Groups
    md.append("| — | **--- 6 Macro-Groups ---** | — | — | — | — | — | — | — | — | — | — |")
    for g_name, gd in g_res.items():
        g_freq_strs = " | ".join([f"{f:.2f}%" for f in gd["class_frequencies"]])
        md.append(f"| — | **{g_name[:25]}** | {g_freq_strs} | {gd['valid_pixels']:,d} |")

    # Global Marginal Distribution
    g_tot = results["global"]["valid_pixels"]
    glob_freq_strs = " | ".join([f"{f:.2f}%" for f in results["global"]["class_frequencies"]])
    md.append(f"| **G** | **Planet-Wide Marginal** | {glob_freq_strs} | {g_tot:,d} |")
    md.append("\n---\n")

    # 10. Table 7: Ground Truth Pixel Counts (Train vs. Test)
    md.append("## 10. Table 7: Per-Biome Ground-Truth Pixel Counts (Train vs. Test)\n")
    md.append("*Documenting exact class representation across all 14 biomes in Train ($N=15,960$ tiles) and Test ($N=2,016$ tiles).*\n")
    md.append("| # | Biome Name | Split | Water | Trees | Grass | Flooded | Crops | Shrub | Built | Bare | Snow |")
    md.append("| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    tr_counts = results["pixel_inventory"]["train"]
    te_counts = results["pixel_inventory"]["test"]
    for b in range(14):
        tr_row = " | ".join([f"{tr_counts[b, c]:,d}" for c in range(9)])
        te_row = " | ".join([f"{te_counts[b, c]:,d}" for c in range(9)])
        md.append(f"| {b} | **{BIOME_NAMES[b][:25]}** | **Test** | {te_row} |")
        md.append(f"| | | **Train** | {tr_row} |")

    output_path.write_text("\n".join(md), encoding="utf-8")


def aggregate_and_export_artifacts(
    conf_mat_14: np.ndarray,
    tile_info: Dict[str, Any],
    data_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    ckpt_path: Optional[str] = None,
    ckpt_epoch: Optional[int] = None,
    ece_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Takes a 14x9x9 confusion matrix and exports the complete artifact suite:
    - Canonical reporting.py schema in test_selected.json
    - Hierarchical multi-granularity metrics in test_results.json
    - Multi-section publication report in test_summary.md
    - CSV tabular summary in test_biome_metrics.csv
    - Raw confusion matrix in conf_mat_14x9x9.npy
    """
    # 0. Global Dataset-Wide Pooled Confusion Matrix & Metrics
    cm_global = conf_mat_14.sum(axis=0)
    global_metrics = compute_metrics_from_confmat(cm_global)
    gt_g = cm_global.sum(axis=1).astype(np.float64)
    w_global = (gt_g / gt_g.sum()) if gt_g.sum() > 0 else np.ones(9, dtype=np.float64) / 9.0
    ious_global = np.array([x if x is not None else 0.0 for x in global_metrics["ious"]], dtype=np.float64)

    # 1. Per-Biome Metrics (including Standardized & Difficulty-Normalized Relative Metrics)
    biomes_dict = {}
    biome_present_mious = {}
    biome_all9_mious = {}
    biome_std_mious = {}
    biome_rel_scores = {}
    biome_shared5_mious = {}
    biome_50k_mious = {}
    biome_fwiou = {}
    biome_macro_f1s = {}
    biome_pixel_accs = {}

    per_group_metrics = {}
    per_biome_miou_frac = {}
    per_biome_macro_f1_frac = {}

    for b in range(14):
        metrics = compute_metrics_from_confmat(conf_mat_14[b])
        b_name = BIOME_NAMES[b]
        metrics["biome_id"] = b
        metrics["name"] = b_name

        tp_b = np.diag(conf_mat_14[b]).astype(np.float64)
        gt_b = conf_mat_14[b].sum(axis=1).astype(np.float64)
        pred_b = conf_mat_14[b].sum(axis=0).astype(np.float64)
        union_b = gt_b + pred_b - tp_b
        with np.errstate(divide="ignore", invalid="ignore"):
            ious_b = np.where(union_b > 0, tp_b / union_b, np.nan)
        pres_b = gt_b > 0

        # Standardized Counterfactual mIoU (global marginal reweighted)
        if np.sum(w_global[pres_b]) > 0:
            std_miou = float(np.sum(w_global[pres_b] * ious_b[pres_b]) / np.sum(w_global[pres_b]) * 100.0)
        else:
            std_miou = 0.0
        metrics["std_miou"] = std_miou

        # Relative Performance Index (difficulty-normalized)
        rel_valid = pres_b & (ious_global > 0)
        if np.any(rel_valid):
            rel_score = float(np.mean(ious_b[rel_valid] / ious_global[rel_valid]))
        else:
            rel_score = 0.0
        metrics["rel_score"] = rel_score

        biomes_dict[str(b)] = metrics

        biome_present_mious[b_name] = metrics["miou_present"]
        biome_all9_mious[b_name] = metrics["miou_all9"]
        biome_std_mious[b_name] = std_miou
        biome_rel_scores[b_name] = rel_score
        biome_shared5_mious[b_name] = metrics["miou_shared5"]
        biome_50k_mious[b_name] = metrics["miou_50k"]
        biome_fwiou[b_name] = metrics["fwiou"]
        biome_macro_f1s[b_name] = metrics["macro_f1"]
        biome_pixel_accs[b_name] = metrics["pixel_accuracy"]

        # Canonical fractional representations (0..1) matching reporting.py
        miou_frac = metrics["miou_present"] / 100.0
        f1_frac = metrics["macro_f1"] / 100.0
        acc_frac = metrics["pixel_accuracy"] / 100.0

        per_biome_miou_frac[b_name] = miou_frac
        per_biome_macro_f1_frac[b_name] = f1_frac

        per_class_diag = []
        for c_idx, c_name in enumerate(DYNAMIC_WORLD_CLASSES):
            cd = metrics["class_details"][c_name]
            rel_c = (float(cd["iou"] / ious_global[c_idx])) if (cd["iou"] is not None and ious_global[c_idx] > 0) else None
            per_class_diag.append({
                "class_id": c_idx,
                "true_count": cd["gt_pixels"],
                "predicted_count": cd["pred_pixels"],
                "true_positive_count": cd["tp"],
                "f1": cd["f1"],
                "iou": cd["iou"],
                "relative_score": rel_c,
                "frequency_pct": cd["frequency_pct"],
            })

        per_group_metrics[b_name] = {
            "macro_f1": f1_frac,
            "accuracy": acc_frac,
            "sample_count": 144,
            "miou": miou_frac,
            "valid_pixel_count": metrics["valid_pixels"],
            "micro_f1": acc_frac,
            "diagnostics": {
                "sample_count": 144,
                "low_support": False,
                "ground_truth_class_count": len([c for c in per_class_diag if c["true_count"] > 0]),
                "union_class_count": len([c for c in per_class_diag if c["true_count"] > 0 or c["predicted_count"] > 0]),
                "macro_f1_gt_present": f1_frac,
                "std_miou": std_miou / 100.0,
                "rel_score": rel_score,
                "shared5_miou": metrics["miou_shared5"] / 100.0,
                "per_class": per_class_diag,
            },
        }

    # 2. 6 Macro-Groups Rollup
    macro_dict = {}
    macro_present_mious = {}
    macro_std_mious = {}
    macro_rel_scores = {}
    cm_macro = np.zeros((len(MACRO_GROUPS), 9, 9), dtype=np.int64)
    for g_idx, (g_name, b_indices) in enumerate(MACRO_GROUPS.items()):
        cm_g = conf_mat_14[b_indices].sum(axis=0)
        cm_macro[g_idx] = cm_g
        g_metrics = compute_metrics_from_confmat(cm_g)
        g_metrics["group_name"] = g_name
        g_metrics["biomes"] = b_indices

        tp_g_m = np.diag(cm_g).astype(np.float64)
        gt_g_m = cm_g.sum(axis=1).astype(np.float64)
        pred_g_m = cm_g.sum(axis=0).astype(np.float64)
        union_g_m = gt_g_m + pred_g_m - tp_g_m
        with np.errstate(divide="ignore", invalid="ignore"):
            ious_g_m = np.where(union_g_m > 0, tp_g_m / union_g_m, np.nan)
        pres_g_m = gt_g_m > 0

        std_miou_g = float(np.sum(w_global[pres_g_m] * ious_g_m[pres_g_m]) / np.sum(w_global[pres_g_m]) * 100.0) if np.sum(w_global[pres_g_m]) > 0 else 0.0
        rel_valid_g = pres_g_m & (ious_global > 0)
        rel_score_g = float(np.mean(ious_g_m[rel_valid_g] / ious_global[rel_valid_g])) if np.any(rel_valid_g) else 0.0

        g_metrics["std_miou"] = std_miou_g
        g_metrics["rel_score"] = rel_score_g
        macro_dict[g_name] = g_metrics
        macro_present_mious[g_name] = g_metrics["miou_present"]
        macro_std_mious[g_name] = std_miou_g
        macro_rel_scores[g_name] = rel_score_g

    # 3. 4 Balanced Clusters Rollup
    clusters_dict = {}
    clusters_present_mious = {}
    clusters_std_mious = {}
    clusters_rel_scores = {}
    cm_clusters = np.zeros((len(CLUSTERS_4), 9, 9), dtype=np.int64)
    for c_idx, (c_name, b_indices) in enumerate(CLUSTERS_4.items()):
        cm_c = conf_mat_14[b_indices].sum(axis=0)
        cm_clusters[c_idx] = cm_c
        c_metrics = compute_metrics_from_confmat(cm_c)
        c_metrics["cluster_name"] = c_name
        c_metrics["biomes"] = b_indices

        tp_c_m = np.diag(cm_c).astype(np.float64)
        gt_c_m = cm_c.sum(axis=1).astype(np.float64)
        pred_c_m = cm_c.sum(axis=0).astype(np.float64)
        union_c_m = gt_c_m + pred_c_m - tp_c_m
        with np.errstate(divide="ignore", invalid="ignore"):
            ious_c_m = np.where(union_c_m > 0, tp_c_m / union_c_m, np.nan)
        pres_c_m = gt_c_m > 0

        std_miou_c = float(np.sum(w_global[pres_c_m] * ious_c_m[pres_c_m]) / np.sum(w_global[pres_c_m]) * 100.0) if np.sum(w_global[pres_c_m]) > 0 else 0.0
        rel_valid_c = pres_c_m & (ious_global > 0)
        rel_score_c = float(np.mean(ious_c_m[rel_valid_c] / ious_global[rel_valid_c])) if np.any(rel_valid_c) else 0.0

        c_metrics["std_miou"] = std_miou_c
        c_metrics["rel_score"] = rel_score_c
        clusters_dict[c_name] = c_metrics
        clusters_present_mious[c_name] = c_metrics["miou_present"]
        clusters_std_mious[c_name] = std_miou_c
        clusters_rel_scores[c_name] = rel_score_c

    # 4. Fairness Metrics Suite (across granularities and class-controlled formulations)
    fairness_suite = {
        "biomes_present_miou": compute_group_fairness(biome_present_mious),
        "biomes_all9_miou": compute_group_fairness(biome_all9_mious),
        "biomes_std_miou": compute_group_fairness(biome_std_mious),
        "biomes_rel_score": compute_group_fairness(biome_rel_scores),
        "biomes_shared5_miou": compute_group_fairness(biome_shared5_mious),
        "biomes_50k_miou": compute_group_fairness(biome_50k_mious),
        "biomes_fwiou": compute_group_fairness(biome_fwiou),
        "biomes_macro_f1": compute_group_fairness(biome_macro_f1s),
        "biomes_pixel_acc": compute_group_fairness(biome_pixel_accs),
        "macro_groups_present_miou": compute_group_fairness(macro_present_mious),
        "macro_groups_std_miou": compute_group_fairness(macro_std_mious),
        "macro_groups_rel_score": compute_group_fairness(macro_rel_scores),
        "clusters_4_present_miou": compute_group_fairness(clusters_present_mious),
        "clusters_4_std_miou": compute_group_fairness(clusters_std_mious),
        "clusters_4_rel_score": compute_group_fairness(clusters_rel_scores),
    }

    # 5. Equalized Odds and Demographic Parity Disparity
    eod_dpm_14 = compute_eod_and_dpm(conf_mat_14)
    eod_dpm_macro = compute_eod_and_dpm(cm_macro)
    eod_dpm_clusters = compute_eod_and_dpm(cm_clusters)

    # 6. Class Frequency Distribution Matrix (%)
    class_frequency_matrix = {
        "biomes": {BIOME_NAMES[b]: [float(x) for x in biomes_dict[str(b)]["class_frequencies"]] for b in range(14)},
        "macro_groups": {g_name: [float(x) for x in macro_dict[g_name]["class_frequencies"]] for g_name in MACRO_GROUPS},
        "clusters_4": {c_name: [float(x) for x in clusters_dict[c_name]["class_frequencies"]] for c_name in CLUSTERS_4},
        "global": [float(x) for x in global_metrics["class_frequencies"]],
    }

    # 7. Ground Truth Pixel Inventory
    pixel_inventory = {"train": np.zeros((14, 9), dtype=np.int64), "test": np.zeros((14, 9), dtype=np.int64)}
    if data_root is not None:
        pixel_inventory = get_ground_truth_pixel_counts(data_root)

    # 8. Assemble Canonical Reporting Record (including class-controlled equity)
    f14 = fairness_suite["biomes_present_miou"]
    f14_std = fairness_suite["biomes_std_miou"]
    f14_rel = fairness_suite["biomes_rel_score"]
    f14_sh5 = fairness_suite["biomes_shared5_miou"]
    f14_f1 = fairness_suite["biomes_macro_f1"]

    overall_miou_frac = global_metrics["miou_present"] / 100.0
    overall_acc_frac = global_metrics["pixel_accuracy"] / 100.0
    overall_f1_frac = global_metrics["macro_f1"] / 100.0

    canonical_record = {
        "epoch": int(ckpt_epoch or 0),
        "split": "test",
        "task": "mmearth",
        "metric_protocol": "present_gt_v1",
        "overall_macro_f1": overall_f1_frac,
        "global_macro_f1": overall_f1_frac,
        "overall_accuracy": overall_acc_frac,
        "global_accuracy": overall_acc_frac,
        "overall_micro_f1": overall_acc_frac,
        "overall_ece": ece_value,
        "ece_disparity": None,
        "overall_mAP": None,
        "global_map": None,
        "overall_miou": overall_miou_frac,
        "global_miou": overall_miou_frac,
        "sample_count": 2016,
        "per_group_metrics": per_group_metrics,
        "worst_case_macro_f1": f14_f1["worst"] / 100.0,
        "worst_case_miou": f14["worst"] / 100.0,
        "worst_case": f14["worst"] / 100.0,
        "best_case": f14["best"] / 100.0,
        "group_mean": f14["mean"] / 100.0,
        "group_std": f14["std"] / 100.0,
        "performance_equity_ratio": f14["per"],
        "normalized_fairness_range": f14["nfr"],
        "nfr_miou": f14["nfr"],
        "nfr_macro_f1": f14_f1["nfr"],
        "nar_accuracy": fairness_suite["biomes_pixel_acc"]["nfr"],
        "worst_case_miou_present": f14["worst"] / 100.0,
        "worst_case_miou_all9": fairness_suite["biomes_all9_miou"]["worst"] / 100.0,
        "gini_coefficient": f14["gini"],
        "disparity": f14["spread"] / 100.0,
        "equalized_odds_disparity": eod_dpm_14["eqodds"],
        "demographic_parity_disparity": eod_dpm_14["dpm"],
        "per_biome_miou": per_biome_miou_frac,
        "per_biome_macro_f1": per_biome_macro_f1_frac,
        "macro_f1": overall_f1_frac,
        "biome_wise_miou_table": per_biome_miou_frac,
        "worst_biome_miou": f14["worst"] / 100.0,
        "biome_std": f14["std"] / 100.0,
        "per": f14["per"],
        "gini": f14["gini"],
        "eqodds": eod_dpm_14["eqodds"],
        "dpm": eod_dpm_14["dpm"],
        # Class-controlled variants for AC & Reviewer ADVW
        "worst_case_std_miou": f14_std["worst"] / 100.0,
        "best_case_std_miou": f14_std["best"] / 100.0,
        "std_miou_spread": f14_std["spread"] / 100.0,
        "std_miou_per": f14_std["per"],
        "worst_case_rel_score": f14_rel["worst"],
        "best_case_rel_score": f14_rel["best"],
        "rel_score_spread": f14_rel["spread"],
        "rel_score_per": f14_rel["per"],
        "worst_case_shared5_miou": f14_sh5["worst"] / 100.0,
        "best_case_shared5_miou": f14_sh5["best"] / 100.0,
        "shared5_miou_spread": f14_sh5["spread"] / 100.0,
        "shared5_miou_per": f14_sh5["per"],
        "per_biome_std_miou": {b: v / 100.0 for b, v in biome_std_mious.items()},
        "per_biome_rel_score": biome_rel_scores,
        "per_biome_shared5_miou": {b: v / 100.0 for b, v in biome_shared5_mious.items()},
    }

    results = {
        "canonical": canonical_record,
        "global": global_metrics,
        "biomes": biomes_dict,
        "macro_groups": macro_dict,
        "clusters_4": clusters_dict,
        "fairness": fairness_suite,
        "class_frequencies": class_frequency_matrix,
        "eod_dpm": {
            "biomes_14": eod_dpm_14,
            "macro_groups_6": eod_dpm_macro,
            "clusters_4": eod_dpm_clusters,
        },
        "pixel_inventory": pixel_inventory,
    }

    # 9. Save Artifacts if out_dir is provided
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # A. Save confusion matrix
        np.save(out_dir / "conf_mat_14x9x9.npy", conf_mat_14)

        # B. Save test_selected.json
        selected_content = {
            "checkpoint": str(ckpt_path or (out_dir / "checkpoints/last.ckpt")),
            "checkpoint_epoch": int(ckpt_epoch or 0),
            "selection": "test checkpoint evaluation",
            "metrics": canonical_record,
        }
        with open(out_dir / "test_selected.json", "w", encoding="utf-8") as f:
            json.dump(selected_content, f, indent=2)

        # C. Save test_results.json
        test_results_dict = {
            **canonical_record,
            "global": results["global"],
            "biomes": results["biomes"],
            "macro_groups": results["macro_groups"],
            "clusters_4": results["clusters_4"],
            "fairness": results["fairness"],
            "class_frequencies": results["class_frequencies"],
            "eod_dpm": results["eod_dpm"],
            "pixel_inventory": {
                "train": results["pixel_inventory"]["train"].tolist(),
                "test": results["pixel_inventory"]["test"].tolist(),
            },
        }
        with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results_dict, f, indent=2)

        # D. Save test_biome_metrics.csv
        csv_lines = ["biome_id,biome_name,present_miou,all9_miou,std_miou,rel_score,macro_f1,miou_50k,shared5_miou,fwiou,pixel_acc,valid_pixels"]
        for b in range(14):
            bd = biomes_dict[str(b)]
            csv_lines.append(
                f"{b},\"{bd['name']}\",{bd['miou_present']:.4f},{bd['miou_all9']:.4f},{bd['std_miou']:.4f},{bd['rel_score']:.4f},{bd['macro_f1']:.4f},{bd['miou_50k']:.4f},{bd['miou_shared5']:.4f},{bd['fwiou']:.4f},{bd['pixel_accuracy']:.4f},{bd['valid_pixels']}"
            )
        (out_dir / "test_biome_metrics.csv").write_text("\n".join(csv_lines), encoding="utf-8")

        # E. Save test_summary.md
        generate_markdown_summary(results, out_dir / "test_summary.md")

        print(f"\n[Artifacts Generated Successfully in {out_dir}]:")
        print(f"  • {out_dir / 'test_selected.json'} (Canonical paper schema + class-controlled)")
        print(f"  • {out_dir / 'test_summary.md'} (Full publication report)")
        print(f"  • {out_dir / 'test_results.json'} (Consolidated metrics)")
        print(f"  • {out_dir / 'test_biome_metrics.csv'} (Tabular metrics)")
        print(f"  • {out_dir / 'conf_mat_14x9x9.npy'} (Raw confusion matrix)")

    return results

def evaluate_dataset_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tile_info: Dict[str, Any],
    bolp_module: Optional[Any] = None,
    data_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    ckpt_path: Optional[str] = None,
    ckpt_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates the model over the test set, accumulates the 14x9x9 confusion matrix,
    and produces the complete artifact suite.
    """
    model.eval()
    conf_mat_14 = np.zeros((14, 9, 9), dtype=np.int64)

    print("\n>>> Running Unified 14-Biome Evaluation Loop...")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.amp.autocast(autocast_device):
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            masks = batch["mask"].squeeze(1).cpu().numpy()
            filenames = batch["filename"]

            if bolp_module is not None:
                logits = bolp_module.forward_with_bolp(
                    model, images, tile_ids=list(filenames), tile_info=tile_info
                )
            elif hasattr(model, "_bolp_forward"):
                logits = model._bolp_forward(batch)
            else:
                logits = model(images)
                if hasattr(logits, "output"):
                    logits = logits.output

            preds = logits.argmax(dim=1).detach().cpu().numpy()

            for i, tile_id in enumerate(filenames):
                t_info = tile_info.get(str(tile_id), {})
                biome_id = t_info.get("biome", None)
                if biome_id is None:
                    continue
                b_idx = int(biome_id)
                if not (0 <= b_idx < 14):
                    continue

                target = masks[i]
                pred = preds[i]
                valid = target >= 0

                if np.any(valid):
                    t_v = target[valid]
                    p_v = pred[valid]
                    flat_coords = t_v * 9 + p_v
                    counts = np.bincount(flat_coords, minlength=81)
                    conf_mat_14[b_idx] += counts.reshape(9, 9)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} test batches...")

    print(">>> Finished test evaluation pass. Aggregating multi-granularity metrics...")

    return aggregate_and_export_artifacts(
        conf_mat_14=conf_mat_14,
        tile_info=tile_info,
        data_root=data_root,
        out_dir=out_dir,
        ckpt_path=ckpt_path,
        ckpt_epoch=ckpt_epoch,
    )
