"""
FairRSFM Benchmark Fairness & Performance Metric Utilities
===========================================================
Unified, verified metric calculations across single-label classification (EuroSAT),
multi-label classification (BigEarthNet), and semantic segmentation (SA-Crop, MMEarth).
All formulations follow the DOFA reference implementation as the gold-standard ground truth.

Key enhancements (WACV review remediation):
- Strict sample standard deviation (ddof=1) across all summary and reporting functions.
- Mathematically rigorous per-label Expected Calibration Error (ECE) for multi-label classification.
- Pixel-level ECE, Equalized Odds (EOdd), and Demographic Parity (DPM) for dense segmentation.
- Non-parametric bootstrap confidence interval estimation (ddof=1 and resampling).
- Explicit, harmonized handling of present vs. all classes in Macro-averaging.
"""

import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERAL STATISTICAL METRICS (GINI, SAMPLE STD, BOOTSTRAP, FAIRNESS SUMMARY)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gini(values: Union[Sequence[float], np.ndarray]) -> float:
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


def compute_sample_std(
    values: Union[Sequence[float], np.ndarray],
    ddof: int = 1,
) -> float:
    """
    Computes sample standard deviation (ddof=1 by default) consistently.
    If len(values) <= ddof, returns 0.0.
    """
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) <= ddof:
        return 0.0
    return float(np.std(vals, ddof=ddof))


def compute_bootstrap_ci(
    values: Union[Sequence[float], np.ndarray],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes non-parametric percentile bootstrap confidence intervals [ci_low, ci_high].
    Critical for small test groups (e.g., EuroSAT High-Biomass with N=25).
    """
    arr = np.asarray([float(v) for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        val = float(arr[0])
        return (val, val)

    rng = np.random.RandomState(seed)
    boot_stats = np.empty(n_bootstraps, dtype=float)
    n = len(arr)
    for b in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[b] = stat_fn(sample)

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_stats, alpha * 100.0))
    ci_high = float(np.percentile(boot_stats, (1.0 - alpha) * 100.0))
    return (ci_low, ci_high)


def compute_fairness_summary(
    group_metrics: Dict[str, float],
    global_score: Optional[float] = None,
    ddof: int = 1,
) -> Dict[str, float]:
    """
    Computes worst-case, best-case, group mean, std (sample std ddof=1 by default),
    disparity (Delta = best - worst), Performance Equity Ratio (PER = worst / best),
    Normalized Fairness Range (NFR = Delta / global_score or Delta / group_mean),
    and Gini coefficient.
    
    Guarantees: worst_case <= mean <= best_case.
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
    std_val   = compute_sample_std(vals, ddof=ddof)
    disp_val  = float(best_val - worst_val)
    per       = (worst_val / best_val) if (best_val > 0 and not np.isnan(worst_val)) else 0.0

    # NFR normalized by global score if provided and positive, else by group mean
    if global_score is not None and global_score > 0 and not np.isnan(global_score):
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

def compute_subgroup_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    present_classes_only: bool = True,
) -> float:
    """
    Computes Macro F1 for a subgroup in single-label multiclass classification.
    
    Args:
        y_true: 1D ground truth array
        y_pred: 1D prediction array
        num_classes: Optional total number of dataset classes
        present_classes_only: If True (default fair protocol), averages only over classes
                              present in y_true, so biomes are not penalized for natural
                              biotype absences. If False and num_classes provided, averages
                              across all 0..num_classes-1.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return float("nan")

    if present_classes_only:
        labels_eval = list(np.unique(y_true))
    else:
        labels_eval = list(range(num_classes)) if num_classes is not None else list(np.unique(np.concatenate([y_true, y_pred])))

    if not labels_eval:
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, labels=labels_eval, average="macro", zero_division=0))


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes binary F1 for each class index in 0..num_classes-1."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
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


def compute_ece_single_label(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for single-label multiclass prediction.
    Partitions confidence in [0, 1] into n_bins, properly including boundary at 0.0.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return float("nan")

    confidences = probs.max(axis=-1)
    if preds is None:
        preds = probs.argmax(axis=-1)
    else:
        preds = np.asarray(preds).ravel()

    accuracies = (preds == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_acc = float(np.mean(accuracies[in_bin]))
            ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)

# Alias to match DOFA scripts
compute_ece_metric = compute_ece_single_label


def compute_equalized_odds_single_label(biome_data: dict, num_classes: int, pred_key: str = "pred") -> float:
    """
    Equalized Odds Disparity: Worst-pair mean absolute difference in Youden's J (TPR - FPR).
    Evaluated across biomes on shared active classes.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b].get("true", [])) > 0])
    if len(biomes) < 2:
        return float("nan")
    J = {}
    for b in biomes:
        yt = np.asarray(biome_data[b]["true"]).ravel()
        yp = np.asarray(biome_data[b][pred_key]).ravel()
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
    found_any = False
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
                found_any = True
    return worst if found_any else float("nan")


def compute_demographic_parity_single_label(biome_data: dict, num_classes: int, pred_key: str = "pred") -> float:
    """
    Demographic Parity Disparity: Worst-pair mean absolute difference in Positive Prediction Rates (PPR).
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b].get(pred_key, [])) > 0])
    if len(biomes) < 2:
        return float("nan")
    PPR = {}
    for b in biomes:
        yp = np.asarray(biome_data[b][pred_key]).ravel()
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = float((yp == c).mean()) if len(yp) > 0 else 0.0
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diff = float(np.mean(np.abs(PPR[b1] - PPR[b2])))
            worst = max(worst, diff)
    return worst


def compute_map_score_single_label(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Mean Average Precision (mAP) for single-label multiclass probabilities."""
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return float("nan")
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

def find_optimal_thresholds(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Finds F1-maximizing decision threshold per class over validation set.
    """
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)
    thresholds = np.full(num_classes, 0.5)
    candidates = np.arange(0.05, 0.95, 0.05)
    for c in range(num_classes):
        if all_labels[:, c].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
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


def compute_per_class_f1_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Computes F1 score per label column for multi-label tasks."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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


def compute_subgroup_macro_f1_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    active_only: bool = True,
) -> float:
    """
    Computes Macro F1 for multi-label classification.
    
    Args:
        y_true: Binary array (N, C)
        y_pred: Binary array (N, C)
        num_classes: Number of classes C
        active_only: If True, only averages over classes with >= 1 positive ground truth sample.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(num_classes):
        yt_c = y_true[:, c]
        if active_only and yt_c.sum() == 0:
            continue
        yp_c = y_pred[:, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1s.append(f1_score(yt_c, yp_c, zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


def compute_binary_ece(
    y_true_col: np.ndarray,
    probs_col: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error for a single binary label using sigmoid probabilities.
    Partitions probabilities in [0, 1] into n_bins and computes weighted difference
    between mean predicted probability (confidence) and empirical positive frequency (accuracy).
    """
    y = np.asarray(y_true_col).astype(float).ravel()
    p = np.asarray(probs_col).astype(float).ravel()
    n = len(y)
    if n == 0:
        return float("nan")

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_conf = float(np.mean(p[in_bin]))
            bin_acc = float(np.mean(y[in_bin]))
            ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ece_multilabel(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None,
    n_bins: int = 10,
    active_only: bool = True,
    return_per_class: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE) for multi-label classification (e.g. m-BigEarthNet).
    
    Item 12 remediation: Evaluates per-label binary calibration on sigmoid probabilities
    p_{i,c} in [0, 1] against binary ground truth y_{i,c} in {0, 1}, and macro-averages
    across active classes. Eliminates the erroneous max(axis=1) formulation.
    
    Args:
        y_true: Binary ground truth array of shape (N, C)
        probs: Predicted sigmoid probabilities of shape (N, C) in [0, 1]
        preds: Optional binary predictions (unused, kept for backward compatibility)
        n_bins: Number of probability bins (default: 10)
        active_only: If True, only average over labels with >= 1 positive ground truth sample
        return_per_class: If True, returns (macro_ece, per_class_ece_array)
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return (float("nan"), np.array([])) if return_per_class else float("nan")

    num_classes = y_true.shape[1] if y_true.ndim > 1 else 1
    per_class_ece = np.full(num_classes, float("nan"))

    for c in range(num_classes):
        yt_c = y_true[:, c] if y_true.ndim > 1 else y_true
        p_c = probs[:, c] if probs.ndim > 1 else probs
        if active_only and yt_c.sum() == 0:
            continue
        per_class_ece[c] = compute_binary_ece(yt_c, p_c, n_bins=n_bins)

    valid = [e for e in per_class_ece if not np.isnan(e)]
    macro_ece = float(np.mean(valid)) if valid else float("nan")

    if return_per_class:
        return macro_ece, per_class_ece
    return macro_ece


def compute_equalized_odds_multilabel(
    biome_data: dict,
    num_classes: int,
    pred_key: str = "pred",
) -> float:
    """
    Equalized Odds Disparity for Multi-label Classification:
    Computes TPR and FPR per label column for each biome, forms Youden's J = TPR - FPR,
    and returns worst-pair mean absolute difference across shared active classes.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b].get("true", [])) > 0])
    if len(biomes) < 2:
        return float("nan")

    J = {}
    for b in biomes:
        yt = np.asarray(biome_data[b]["true"])
        yp = np.asarray(biome_data[b].get(pred_key, biome_data[b].get("pred_opt", biome_data[b].get("pred", []))))
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            yt_c = yt[:, c]
            yp_c = yp[:, c]
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
    found_any = False
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
                found_any = True
    return worst if found_any else float("nan")


def compute_demographic_parity_multilabel(
    biome_data: dict,
    num_classes: int,
    pred_key: str = "pred",
) -> float:
    """
    Demographic Parity Disparity for Multi-label Classification:
    Computes Positive Prediction Rate (PPR) per label column across biomes
    and returns worst-pair mean absolute difference.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown"])
    biomes = [b for b in biomes if len(biome_data[b].get(pred_key, biome_data[b].get("pred_opt", []))) > 0]
    if len(biomes) < 2:
        return float("nan")

    PPR = {}
    for b in biomes:
        yp = np.asarray(biome_data[b].get(pred_key, biome_data[b].get("pred_opt", [])))
        ppr_b = np.zeros(num_classes)
        for c in range(num_classes):
            ppr_b[c] = float(yp[:, c].mean()) if len(yp) > 0 else 0.0
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diff = float(np.mean(np.abs(PPR[b1] - PPR[b2])))
            worst = max(worst, diff)
    return worst


def compute_map_multilabel(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Computes mean average precision across all classes in multi-label classification."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return float("nan")
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
    ignore_index: int = -1,
    present_classes_only: bool = True,
) -> float:
    """
    Mean Intersection over Union (mIoU) ignoring unlabeled/ignore pixels.
    
    Args:
        y_true_flat: Ground truth pixel array (1D or multi-dimensional)
        y_pred_flat: Predicted pixel array (1D or multi-dimensional)
        num_classes: Total number of classes
        ignore_index: Label index to ignore
        present_classes_only: If True, average only over classes present in ground truth
    """
    yt = np.asarray(y_true_flat).ravel()
    yp = np.asarray(y_pred_flat).ravel()
    mask = yt != ignore_index
    y_true = yt[mask]
    y_pred = yp[mask]

    if present_classes_only:
        present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    else:
        present = list(range(num_classes))

    if not present:
        return float("nan")

    ious = []
    for c in present:
        tp    = int(((y_pred == c) & (y_true == c)).sum())
        fp    = int(((y_pred == c) & (y_true != c)).sum())
        fn    = int(((y_pred != c) & (y_true == c)).sum())
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_per_class_iou(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Per-class IoU array for semantic segmentation."""
    yt = np.asarray(y_true_flat).ravel()
    yp = np.asarray(y_pred_flat).ravel()
    mask = yt != ignore_index
    y_true = yt[mask]
    y_pred = yp[mask]
    ious = np.full(num_classes, float("nan"))
    for c in range(num_classes):
        if (y_true == c).sum() == 0:
            continue
        tp    = int(((y_pred == c) & (y_true == c)).sum())
        fp    = int(((y_pred == c) & (y_true != c)).sum())
        fn    = int(((y_pred != c) & (y_true == c)).sum())
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else float("nan")
    return ious


def compute_macro_f1_seg(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
    present_classes_only: bool = True,
) -> float:
    """Macro F1 for dense pixel-level semantic segmentation."""
    yt = np.asarray(y_true_flat).ravel()
    yp = np.asarray(y_pred_flat).ravel()
    mask = yt != ignore_index
    y_true = yt[mask]
    y_pred = yp[mask]

    if present_classes_only:
        present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    else:
        present = list(range(num_classes))

    if not present:
        return float("nan")

    f1s = []
    for c in present:
        tp    = int(((y_pred == c) & (y_true == c)).sum())
        fp    = int(((y_pred == c) & (y_true != c)).sum())
        fn    = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else float("nan"))
    valid = [v for v in f1s if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_ece_seg(
    y_true_flat: np.ndarray,
    probs_max: np.ndarray,
    preds_flat: Optional[np.ndarray] = None,
    ignore_index: int = -1,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for dense semantic segmentation computed at the pixel level.
    Properly handles flattening, ignore indices, and boundary at 0.0.
    
    Args:
        y_true_flat: Ground truth pixel array (any shape)
        probs_max: Maximum confidence array or full class probability tensor
        preds_flat: Optional predicted class labels (any shape). Inferred if None.
        ignore_index: Masked pixel label (default -1)
        n_bins: Number of probability bins
    """
    yt = np.asarray(y_true_flat).ravel()
    p = np.asarray(probs_max)

    if preds_flat is not None:
        yp = np.asarray(preds_flat).ravel()
        conf = p.ravel()
    else:
        # Check if p has class dimension
        if p.ndim > yt.ndim or (p.ndim > 1 and p.shape[-1] > 1 and p.size != yt.size):
            p_flat = p.reshape(-1, p.shape[-1])
            yp = np.argmax(p_flat, axis=-1)
            conf = np.max(p_flat, axis=-1)
        else:
            conf = p.ravel()
            yp = conf.round().astype(int)

    mask = yt != ignore_index
    y_true = yt[mask]
    conf_v = conf[mask]
    preds_v = yp[mask]

    n = len(y_true)
    if n == 0:
        return float("nan")

    correct = (preds_v == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (conf_v >= lo) & (conf_v <= hi) if i == 0 else (conf_v > lo) & (conf_v <= hi)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_conf = float(np.mean(conf_v[in_bin]))
            bin_acc = float(np.mean(correct[in_bin]))
            ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_eqodds_seg(
    biome_data: dict,
    num_classes: int,
    ignore_index: int = -1,
    pred_key: str = "pred",
) -> float:
    """
    Equalized Odds Disparity for Dense Semantic Segmentation:
    Computes Youden's J per class on valid unignored pixels across biomes and returns worst-pair mean delta.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown" and len(biome_data[b].get("true", [])) > 0])
    if len(biomes) < 2:
        return float("nan")

    J = {}
    for b in biomes:
        true_list = biome_data[b]["true"]
        pred_list = biome_data[b].get(pred_key, biome_data[b].get("pred", []))
        
        # Robust flattening across list of 2D or 3D patches
        yt = np.concatenate([np.asarray(x).ravel() for x in true_list]) if isinstance(true_list, list) else np.asarray(true_list).ravel()
        yp = np.concatenate([np.asarray(x).ravel() for x in pred_list]) if isinstance(pred_list, list) else np.asarray(pred_list).ravel()

        mask = yt != ignore_index
        yt_v = yt[mask]
        yp_v = yp[mask]
        j_b = np.full(num_classes, float("nan"))
        for c in range(num_classes):
            tp = int(((yp_v == c) & (yt_v == c)).sum())
            fn = int(((yp_v != c) & (yt_v == c)).sum())
            fp = int(((yp_v == c) & (yt_v != c)).sum())
            tn = int(((yp_v != c) & (yt_v != c)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            if not (np.isnan(tpr) or np.isnan(fpr)):
                j_b[c] = tpr - fpr
        J[b] = j_b

    worst = 0.0
    found_any = False
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diffs = [abs(J[b1][c] - J[b2][c]) for c in range(num_classes)
                     if not (np.isnan(J[b1][c]) or np.isnan(J[b2][c]))]
            if diffs:
                worst = max(worst, float(np.mean(diffs)))
                found_any = True
    return worst if found_any else float("nan")


def compute_demographic_parity_seg(
    biome_data: dict,
    num_classes: int,
    ignore_index: int = -1,
    pred_key: str = "pred",
) -> float:
    """
    Demographic Parity Disparity for Dense Semantic Segmentation:
    Computes Positive Prediction Rate (PPR) per class across biomes and returns worst-pair mean delta.
    """
    biomes = sorted([b for b in biome_data.keys() if b != "Unknown"])
    biomes = [b for b in biomes if len(biome_data[b].get(pred_key, biome_data[b].get("pred", []))) > 0]
    if len(biomes) < 2:
        return float("nan")

    PPR = {}
    for b in biomes:
        pred_list = biome_data[b].get(pred_key, biome_data[b].get("pred", []))
        true_list = biome_data[b].get("true", [])

        yp = np.concatenate([np.asarray(x).ravel() for x in pred_list]) if isinstance(pred_list, list) else np.asarray(pred_list).ravel()
        if true_list:
            yt = np.concatenate([np.asarray(x).ravel() for x in true_list]) if isinstance(true_list, list) else np.asarray(true_list).ravel()
            mask = yt != ignore_index
            yp_v = yp[mask]
        else:
            yp_v = yp

        ppr_b = np.zeros(num_classes)
        n = len(yp_v)
        if n > 0:
            for c in range(num_classes):
                ppr_b[c] = float((yp_v == c).mean())
        PPR[b] = ppr_b

    worst = 0.0
    for i, b1 in enumerate(biomes):
        for b2 in biomes[i + 1:]:
            diff = float(np.mean(np.abs(PPR[b1] - PPR[b2])))
            worst = max(worst, diff)
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMART DISPATCHERS & BACKWARD COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_equalized_odds(
    biome_data: dict,
    num_classes: int,
    task_type: Optional[str] = None,
    ignore_index: int = -1,
    pred_key: str = "pred",
) -> float:
    """
    Unified Equalized Odds Dispatcher.
    Automatically identifies whether data is single-label classification,
    multi-label classification, or semantic segmentation.
    """
    if task_type == "segmentation":
        return compute_eqodds_seg(biome_data, num_classes, ignore_index=ignore_index, pred_key=pred_key)
    elif task_type == "multilabel":
        return compute_equalized_odds_multilabel(biome_data, num_classes, pred_key=pred_key)
    elif task_type == "single_label":
        return compute_equalized_odds_single_label(biome_data, num_classes, pred_key=pred_key)

    # Auto-detection
    for b, data in biome_data.items():
        if b == "Unknown":
            continue
        true_samples = data.get("true", [])
        if len(true_samples) > 0:
            first = true_samples[0]
            if isinstance(first, np.ndarray) and first.ndim >= 2:
                # 2D mask or 2D image -> segmentation
                return compute_eqodds_seg(biome_data, num_classes, ignore_index=ignore_index, pred_key=pred_key)
            first_arr = np.asarray(first)
            if first_arr.ndim == 1 and len(first_arr) == num_classes:
                # Multi-label one-hot/multi-hot vector per sample
                return compute_equalized_odds_multilabel(biome_data, num_classes, pred_key=pred_key)
            if isinstance(true_samples, np.ndarray) and true_samples.ndim == 2:
                # Array of shape (N, C) -> multi-label
                return compute_equalized_odds_multilabel(biome_data, num_classes, pred_key=pred_key)
            break

    return compute_equalized_odds_single_label(biome_data, num_classes, pred_key=pred_key)


def compute_demographic_parity(
    biome_data: dict,
    num_classes: int,
    task_type: Optional[str] = None,
    ignore_index: int = -1,
    pred_key: str = "pred",
) -> float:
    """
    Unified Demographic Parity Dispatcher.
    """
    if task_type == "segmentation":
        return compute_demographic_parity_seg(biome_data, num_classes, ignore_index=ignore_index, pred_key=pred_key)
    elif task_type == "multilabel":
        return compute_demographic_parity_multilabel(biome_data, num_classes, pred_key=pred_key)
    elif task_type == "single_label":
        return compute_demographic_parity_single_label(biome_data, num_classes, pred_key=pred_key)

    # Auto-detection
    for b, data in biome_data.items():
        if b == "Unknown":
            continue
        pred_samples = data.get(pred_key, data.get("pred", []))
        if len(pred_samples) > 0:
            first = pred_samples[0]
            if isinstance(first, np.ndarray) and first.ndim >= 2:
                return compute_demographic_parity_seg(biome_data, num_classes, ignore_index=ignore_index, pred_key=pred_key)
            first_arr = np.asarray(first)
            if first_arr.ndim == 1 and len(first_arr) == num_classes:
                return compute_demographic_parity_multilabel(biome_data, num_classes, pred_key=pred_key)
            if isinstance(pred_samples, np.ndarray) and pred_samples.ndim == 2:
                return compute_demographic_parity_multilabel(biome_data, num_classes, pred_key=pred_key)
            break

    return compute_demographic_parity_single_label(biome_data, num_classes, pred_key=pred_key)
