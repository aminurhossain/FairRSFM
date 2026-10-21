"""
FairRSFM Benchmark Fairness & Performance Metric Utilities
===========================================================
Unified, verified metric calculations across single-label classification (EuroSAT),
multi-label classification (BigEarthNet), and semantic segmentation (SA-Crop, MMEarth).
All formulations follow the DOFA reference implementation as the gold-standard ground truth.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERAL STATISTICAL METRICS (GINI, DISPARITY, FAIRNESS SUMMARY)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gini(values: Union[List[float], np.ndarray]) -> float:
    """
    Computes the Gini coefficient across a list of group metric values.
    G = sum_{i=1}^n (2*i - n - 1) * y_i / (n * sum y_i)
    """
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    arr = np.sort(np.array(vals))
    n = len(arr)
    sum_val = float(np.sum(arr))
    if sum_val == 0.0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * arr) / (n * sum_val))


def compute_fairness_summary(
    group_metrics: Dict[str, float],
    global_score: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes worst-case, best-case, group mean, std, disparity (Δ),
    Performance Equity Ratio (PER), Normalized Fairness Range (NFR), and Gini coefficient.
    """
    vals = [float(v) for v in group_metrics.values() if v is not None and not np.isnan(v)]
    if not vals:
        return {
            "worst_case": float("nan"),
            "best_case": float("nan"),
            "mean": float("nan"),
            "std": 0.0,
            "disparity": float("nan"),
            "per": 0.0,
            "nfr": float("nan"),
            "gini": 0.0,
        }
    worst_val = min(vals)
    best_val  = max(vals)
    mean_val  = float(np.mean(vals))
    std_val   = float(np.std(vals))
    disp_val  = float(best_val - worst_val)
    per       = (worst_val / best_val) if (best_val > 0 and not np.isnan(worst_val)) else 0.0

    # NFR normalized by global score if provided, else by group mean
    if global_score is not None and global_score > 0:
        nfr = disp_val / global_score
    else:
        nfr = (disp_val / mean_val) if mean_val > 0 else float("nan")

    gini = compute_gini(vals)
    return {
        "worst_case": worst_val,
        "best_case": best_val,
        "mean": mean_val,
        "std": std_val,
        "disparity": disp_val,
        "per": per,
        "nfr": nfr,
        "gini": gini,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SINGLE-LABEL CLASSIFICATION (EUROSAT)
# ─────────────────────────────────────────────────────────────────────────────

def compute_subgroup_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Macro F1 restricted only to classes present in this subgroup."""
    if len(y_true) == 0:
        return float("nan")
    present = list(np.unique(y_true))
    if not present:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes binary F1 for each class index in 0..num_classes-1."""
    f1s = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        yt_c = (y_true == c).astype(int)
        yp_c = (y_pred == c).astype(int)
        if yt_c.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s[c] = float(f1_score(yt_c, yp_c, zero_division=0))
    return f1s


def compute_ece_single_label(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error for single-label multiclass prediction.
    """
    if len(y_true) == 0:
        return float("nan")
    confidences = probs.max(axis=-1)
    accuracies  = (preds == y_true)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
    return float(ece)

# Alias to match DOFA scripts
compute_ece_metric = compute_ece_single_label


def compute_equalized_odds_single_label(biome_data: dict, num_classes: int) -> float:
    """
    Equalized Odds Disparity: Worst-pair mean absolute difference in Youden's J (TPR - FPR).
    """
    biomes = [b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0]
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.array(biome_data[b]["true"])
        yp = np.array(biome_data[b]["pred"])
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            yt_c = (yt == c).astype(int)
            yp_c = (yp == c).astype(int)
            tp = int(((yp_c == 1) & (yt_c == 1)).sum())
            fn = int(((yp_c == 0) & (yt_c == 1)).sum())
            fp = int(((yp_c == 1) & (yt_c == 0)).sum())
            tn = int(((yp_c == 0) & (yt_c == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            if not (np.isnan(tpr) or np.isnan(fpr)):
                j_b[c] = tpr - fpr
        J[b] = j_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
    return worst

compute_equalized_odds = compute_equalized_odds_single_label


def compute_demographic_parity_single_label(biome_data: dict, num_classes: int) -> float:
    """
    Demographic Parity Disparity: Worst-pair mean absolute difference in Positive Prediction Rates (PPR).
    """
    biomes = [b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0]
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.array(biome_data[b]["pred"])
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = (yp == c).mean()
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst

compute_demographic_parity = compute_demographic_parity_single_label


def compute_map_score_single_label(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Mean Average Precision (mAP) for single-label multiclass probabilities."""
    y_true_oh = np.eye(num_classes)[y_true]
    try:
        aps = []
        for c in range(num_classes):
            if y_true_oh[:, c].sum() > 0:
                ap = average_precision_score(y_true_oh[:, c], probs[:, c])
                aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTI-LABEL CLASSIFICATION (BIGEARTHNET)
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_thresholds(all_labels: np.ndarray, all_probs: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Finds F1-maximizing decision threshold per class over validation set.
    """
    thresholds = np.full(num_classes, 0.5)
    candidates = np.arange(0.1, 0.95, 0.05)
    for c in range(num_classes):
        if all_labels[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            preds_c = (all_probs[:, c] >= t).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1 = f1_score(all_labels[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = float(best_t)
    return thresholds

optimize_thresholds_per_class = find_optimal_thresholds


def compute_per_class_f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes F1 score per label column for multi-label tasks."""
    f1s = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        yt_c = y_true[:, c]
        yp_c = y_pred[:, c]
        if yt_c.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s[c] = float(f1_score(yt_c, yp_c, zero_division=0))
    return f1s


def compute_subgroup_macro_f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Computes Macro F1 across all classes with at least one positive sample in the subgroup."""
    f1s = []
    for c in range(num_classes):
        yt_c = y_true[:, c]
        if yt_c.sum() == 0:
            continue
        yp_c = y_pred[:, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s.append(f1_score(yt_c, yp_c, zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


def compute_ece_multilabel(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray, n_bins: int = 10) -> float:
    """ECE for multi-label prediction averaging across active classes."""
    if len(y_true) == 0:
        return float("nan")
    y_max = y_true.max(axis=1)
    p_max = probs.max(axis=1)
    pred_max = preds.max(axis=1)
    correct = (pred_max == y_max).astype(float)
    ece_val, n = 0.0, len(y_true)
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (p_max >= lo) & (p_max < hi)
        if mask.sum() == 0:
            continue
        ece_val += (mask.sum() / n) * abs(correct[mask].mean() - p_max[mask].mean())
    return float(ece_val)


def compute_map_multilabel(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Computes mean average precision across all classes in multi-label classification."""
    aps = []
    for c in range(num_classes):
        if y_true[:, c].sum() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aps.append(average_precision_score(y_true[:, c], probs[:, c]))
    return float(np.mean(aps)) if aps else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEMANTIC SEGMENTATION (SA-CROP & MMEARTH-20K)
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1
) -> float:
    """Mean Intersection over Union (mIoU) ignoring unlabeled/ignore pixels."""
    mask = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    if not present:
        return float("nan")
    ious = []
    for c in present:
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_per_class_iou(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1
) -> np.ndarray:
    """Per-class IoU array for semantic segmentation."""
    mask = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    ious = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        if (y_true == c).sum() == 0:
            continue
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else float("nan")
    return ious


def compute_macro_f1_seg(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1
) -> float:
    """Macro F1 for dense pixel-level semantic segmentation."""
    mask = y_true_flat != ignore_index
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    if not present:
        return float("nan")
    f1s = []
    for c in present:
        tp    = ((y_pred == c) & (y_true == c)).sum()
        fp    = ((y_pred == c) & (y_true != c)).sum()
        fn    = ((y_pred != c) & (y_true == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in f1s if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_ece_seg(
    y_true_flat: np.ndarray,
    probs_max: np.ndarray,
    preds_flat: np.ndarray,
    ignore_index: int = -1,
    n_bins: int = 10
) -> float:
    """Expected Calibration Error for dense segmentation."""
    mask       = y_true_flat != ignore_index
    y_true     = y_true_flat[mask]
    probs_max  = probs_max[mask]
    preds_flat = preds_flat[mask]
    if len(y_true) == 0:
        return float("nan")
    correct = (preds_flat == y_true).astype(float)
    ece, n  = 0.0, len(y_true)
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (probs_max >= lo) & (probs_max < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - probs_max[m].mean())
    return float(ece)


def compute_eqodds_seg(
    biome_data: dict,
    num_classes: int,
    ignore_index: int = -1
) -> float:
    """
    Equalized Odds Disparity for Segmentation:
    Computes Youden's J per class on valid unignored pixels across biomes and returns worst-pair mean delta.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.concatenate(biome_data[b]["true"])
        yp = np.concatenate(biome_data[b]["pred"])
        mask = yt != ignore_index
        yt_v = yt[mask]
        yp_v = yp[mask]
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            yt_c = (yt_v == c).astype(int)
            yp_c = (yp_v == c).astype(int)
            tp = ((yp_c == 1) & (yt_c == 1)).sum()
            fn = ((yp_c == 0) & (yt_c == 1)).sum()
            fp = ((yp_c == 1) & (yt_c == 0)).sum()
            tn = ((yp_c == 0) & (yt_c == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            if not (np.isnan(tpr) or np.isnan(fpr)):
                j_b[c] = tpr - fpr
        J[b] = j_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i+1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
    return worst


def compute_demographic_parity_seg(
    biome_data: dict,
    num_classes: int,
    ignore_index: int = -1
) -> float:
    """
    Demographic Parity Disparity for Segmentation:
    Computes Positive Prediction Rate (PPR) per class across biomes and returns worst-pair mean delta.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b]["true"]) > 0])
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.concatenate(biome_data[b]["pred"])
        yt = np.concatenate(biome_data[b]["true"])
        mask = yt != ignore_index
        yp_v = yp[mask]
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = (yp_v == c).mean() if len(yp_v) > 0 else 0.0
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i+1:]:
            worst = max(worst, float(np.mean(np.abs(PPR[b1] - PPR[b2]))))
    return worst
