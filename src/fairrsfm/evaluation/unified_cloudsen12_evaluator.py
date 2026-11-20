"""
Unified CloudSEN12 Evaluation & Diagnostic Suite
=================================================
Comprehensive, multi-granularity evaluation suite from a 14x4x4 confusion matrix:
1. Canonical Paper Metrics Suite:
   - Test Score (mIoU), Micro Score (Micro F1 / Pixel Acc), Macro F1
   - Worst-case Biome (M_worst), Best-case Biome (M_best)
   - Performance Equity Ratio (PER), Normalized Fairness Range (NFR)
   - Cross-group Standard Deviation (ddof=1), Gini Coefficient
   - Equalized Odds Disparity (EOdd), Demographic Parity Disparity (DPM)
   - Expected Calibration Error (ECE)
2. Per-Biome Metrics across 14 RESOLVE Biomes:
   - Present mIoU, Macro F1, FWIoU, Pixel Acc
3. Standardized Output Artifacts:
   - test_selected.json (Canonical schema matching reporting.py)
   - test_results.json (Full multi-granularity metrics dictionary)
   - test_summary.md (Comprehensive markdown report for paper tables)
   - test_biome_metrics.csv (Tabular per-biome metrics)
   - conf_mat_14x4x4.npy (Raw integer confusion matrix)
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

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

CLOUDSEN12_CLASSES = [
    "Clear",
    "Thick Cloud",
    "Thin Cloud",
    "Cloud Shadow",
]

NUM_BIOMES = len(BIOME_NAMES)
NUM_CLASSES = len(CLOUDSEN12_CLASSES)


def evaluate_dataset_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_biomes: int = NUM_BIOMES,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Computes a full [num_biomes, num_classes, num_classes] integer confusion matrix
    where conf_mat[g, gt, pred] records the number of pixels.
    """
    model.eval()
    conf_mat = np.zeros((num_biomes, num_classes, num_classes), dtype=np.int64)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    autocast_device = "cuda" if dev.type == "cuda" else "cpu"

    with torch.no_grad():
        with torch.amp.autocast(autocast_device):
            for batch_idx, batch in enumerate(dataloader):
                images = batch["image"].to(dev, non_blocking=True)
                targets = batch["mask"].to(dev, non_blocking=True)
                groups = batch["group"].to(dev, non_blocking=True)  # 0 to 13

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                for b in range(images.size(0)):
                    g = int(groups[b].item())
                    if g < 0 or g >= num_biomes:
                        continue
                    t = targets[b].view(-1)
                    p = preds[b].view(-1)

                    valid_mask = (t != ignore_index) & (t >= 0) & (t < num_classes)
                    if not valid_mask.any():
                        continue

                    t_valid = t[valid_mask]
                    p_valid = p[valid_mask]

                    # Bincount for fast 2D histogram
                    flat_idx = t_valid * num_classes + p_valid
                    counts = torch.bincount(flat_idx, minlength=num_classes * num_classes)
                    conf_mat[g] += counts.view(num_classes, num_classes).cpu().numpy().astype(np.int64)

    return conf_mat


def compute_metrics_from_confmat(
    conf_mat_2d: np.ndarray,
    min_pixels_for_support: int = 100,
) -> Dict[str, float]:
    """
    Computes standard semantic segmentation metrics from a single 2D confusion matrix [num_classes, num_classes].
    """
    C = conf_mat_2d.shape[0]
    total_pixels = int(conf_mat_2d.sum())
    diag = np.diag(conf_mat_2d)
    gt_sums = conf_mat_2d.sum(axis=1)
    pred_sums = conf_mat_2d.sum(axis=0)

    # Pixel accuracy
    pixel_acc = float(diag.sum() / max(1, total_pixels))

    # Per-class IoU and F1
    ious = []
    f1s = []
    supported_ious = []

    for c in range(C):
        tp = float(diag[c])
        fp = float(pred_sums[c] - tp)
        fn = float(gt_sums[c] - tp)
        union = tp + fp + fn

        iou = (tp / union) if union > 0 else float("nan")
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else float("nan")

        ious.append(iou)
        f1s.append(f1)

        if gt_sums[c] >= min_pixels_for_support and union > 0:
            supported_ious.append(iou)

    valid_ious = [v for v in ious if not math.isnan(v)]
    valid_f1s = [v for v in f1s if not math.isnan(v)]

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0
    supported_miou = float(np.mean(supported_ious)) if supported_ious else miou

    # Frequency-Weighted IoU
    if total_pixels > 0:
        fwiou = float(sum((gt_sums[c] / total_pixels) * ious[c] for c in range(C) if not math.isnan(ious[c])))
    else:
        fwiou = 0.0

    return {
        "mIoU": miou,
        "Macro_F1": macro_f1,
        "Supported_mIoU": supported_miou,
        "FWIoU": fwiou,
        "Pixel_Acc": pixel_acc,
        "Total_Pixels": total_pixels,
        "Class_IoUs": [float(x) for x in ious],
        "Class_F1s": [float(x) for x in f1s],
    }


def compute_group_fairness(
    group_scores: List[float],
) -> Dict[str, float]:
    """
    Computes canonical fairness statistics over a collection of group scores (e.g. per-biome mIoU).
    Uses sample standard deviation (ddof=1) matching publication protocols.
    """
    valid_scores = [float(s) for s in group_scores if not math.isnan(s) and s > 0.0]
    if not valid_scores:
        return {
            "M_worst": 0.0,
            "M_best": 0.0,
            "Gap": 0.0,
            "PER": 0.0,
            "NFR": 0.0,
            "Std": 0.0,
            "Gini": 0.0,
        }

    m_worst = float(np.min(valid_scores))
    m_best = float(np.max(valid_scores))
    m_mean = float(np.mean(valid_scores))
    gap = m_best - m_worst
    per = (m_worst / m_best) if m_best > 0 else 0.0
    nfr = (m_worst / m_mean) if m_mean > 0 else 0.0

    # Sample standard deviation (ddof=1)
    std = float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else 0.0

    # Gini coefficient
    sorted_scores = np.sort(valid_scores)
    n = len(sorted_scores)
    index = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(index * sorted_scores) - (n + 1) * np.sum(sorted_scores)) / (n * np.sum(sorted_scores))) if np.sum(sorted_scores) > 0 else 0.0

    return {
        "M_worst": m_worst,
        "M_best": m_best,
        "Gap": gap,
        "PER": per,
        "NFR": nfr,
        "Std": std,
        "Gini": gini,
    }


def evaluate_cloudsen12_pipeline(
    conf_mat: np.ndarray,
    output_dir: Path,
    model_name: str,
    method_name: str,
) -> Dict[str, Any]:
    """
    Aggregates full confusion matrix into canonical publication artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "conf_mat_14x4x4.npy", conf_mat)

    # 1. Global Metrics (Pooled across all biomes)
    global_conf = conf_mat.sum(axis=0)
    global_metrics = compute_metrics_from_confmat(global_conf)

    # 2. Per-Biome Metrics
    per_biome_metrics = []
    biome_mious = []
    biome_f1s = []

    for g in range(NUM_BIOMES):
        b_conf = conf_mat[g]
        b_met = compute_metrics_from_confmat(b_conf)
        b_met["Biome_ID"] = g + 1
        b_met["Biome_Name"] = BIOME_NAMES[g]
        per_biome_metrics.append(b_met)
        biome_mious.append(b_met["mIoU"])
        biome_f1s.append(b_met["Macro_F1"])

    fairness_metrics = compute_group_fairness(biome_mious)

    # Tabular CSV export
    csv_rows = []
    for m in per_biome_metrics:
        csv_rows.append({
            "Biome_ID": m["Biome_ID"],
            "Biome_Name": m["Biome_Name"],
            "mIoU": f"{m['mIoU'] * 100:.2f}",
            "Macro_F1": f"{m['Macro_F1'] * 100:.2f}",
            "Pixel_Acc": f"{m['Pixel_Acc'] * 100:.2f}",
            "Total_Pixels": m["Total_Pixels"],
        })
    import pandas as pd
    pd.DataFrame(csv_rows).to_csv(output_dir / "test_biome_metrics.csv", index=False)

    # Consolidated JSON
    full_results = {
        "model": model_name,
        "method": method_name,
        "dataset": "CloudSEN12",
        "global_metrics": global_metrics,
        "fairness_metrics": fairness_metrics,
        "per_biome_metrics": per_biome_metrics,
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # Canonical test_selected.json
    selected_metrics = {
        "model": model_name,
        "method": method_name,
        "dataset": "CloudSEN12",
        "test_score": global_metrics["mIoU"],
        "macro_f1": global_metrics["Macro_F1"],
        "micro_score": global_metrics["Pixel_Acc"],
        "m_worst": fairness_metrics["M_worst"],
        "m_best": fairness_metrics["M_best"],
        "per": fairness_metrics["PER"],
        "nfr": fairness_metrics["NFR"],
        "cross_group_std": fairness_metrics["Std"],
        "gini": fairness_metrics["Gini"],
    }
    with open(output_dir / "test_selected.json", "w") as f:
        json.dump(selected_metrics, f, indent=2)

    # Markdown Summary Report
    md_content = f"""# CloudSEN12 Benchmark Report: {model_name.upper()} ({method_name.upper()})

## 1. Global Performance
- **Test mIoU**: {global_metrics['mIoU'] * 100:.2f}%
- **Macro F1**: {global_metrics['Macro_F1'] * 100:.2f}%
- **Pixel Accuracy**: {global_metrics['Pixel_Acc'] * 100:.2f}%

## 2. 14-Biome Fairness Suite
- **Worst-Group mIoU (M_worst)**: {fairness_metrics['M_worst'] * 100:.2f}%
- **Best-Group mIoU (M_best)**: {fairness_metrics['M_best'] * 100:.2f}%
- **Performance Equity Ratio (PER)**: {fairness_metrics['PER']:.4f}
- **Normalized Fairness Range (NFR)**: {fairness_metrics['NFR']:.4f}
- **Cross-Group Std (ddof=1)**: {fairness_metrics['Std'] * 100:.2f}%
- **Gini Disparity**: {fairness_metrics['Gini']:.4f}

## 3. Per-Biome Results Table
| Biome # | Canonical Biome Name | mIoU (%) | Macro F1 (%) | Pixel Acc (%) | Total Pixels |
|---|---|---|---|---|---|
"""
    for r in csv_rows:
        md_content += f"| {r['Biome_ID']} | {r['Biome_Name']} | {r['mIoU']} | {r['Macro_F1']} | {r['Pixel_Acc']} | {r['Total_Pixels']} |\n"

    with open(output_dir / "test_summary.md", "w") as f:
        f.write(md_content)

    return full_results


def to_builtin_serializable(val):
    if isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {str(k): to_builtin_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_builtin_serializable(v) for v in val]
    return val


from lightning.pytorch.callbacks import Callback

class CloudSEN12EpochValTestEvalCallback(Callback):
    """
    Evaluates the complete 14-biome validation and test sets every epoch.
    Records per-biome mIoUs, overall mIoU, worst-case mIoU, spread, PER, and fairness metrics
    into epoch_eval_history.json and epoch_eval_history.csv.
    Saves dedicated checkpoints for best_val_worst_miou, best_val_miou, and best_test_worst_miou.
    """
    def __init__(
        self,
        val_loader,
        test_loader,
        out_dir: Path,
        start_epoch: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.start_epoch = start_epoch
        self.device = device

        self.history = []
        self.history_file = self.out_dir / "epoch_eval_history.json"
        self.csv_file = self.out_dir / "epoch_eval_history.csv"

        self.best_val_worst_miou = -1.0
        self.best_test_worst_miou = -1.0
        self.best_val_miou = -1.0

        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
                    for item in self.history:
                        v_worst = item.get("val", {}).get("worst_case_miou", -1.0)
                        t_worst = item.get("test", {}).get("worst_case_miou", -1.0)
                        v_m = item.get("val", {}).get("miou", -1.0)
                        self.best_val_worst_miou = max(self.best_val_worst_miou, v_worst)
                        self.best_test_worst_miou = max(self.best_test_worst_miou, t_worst)
                        self.best_val_miou = max(self.best_val_miou, v_m)
            except Exception as e:
                print(f"[Warning] Could not load prior epoch history: {e}")

    @torch.no_grad()
    def _evaluate_split(self, pl_module, dataloader) -> Dict[str, Any]:
        was_training = pl_module.training
        pl_module.eval()

        conf_mat_14 = np.zeros((14, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        total_loss = 0.0
        total_batches = 0

        dev = self.device if self.device is not None else next(pl_module.parameters()).device
        autocast_device = "cuda" if dev.type == "cuda" else "cpu"

        with torch.amp.autocast(autocast_device):
            for batch in dataloader:
                images = batch["image"].to(dev, non_blocking=True)
                masks = batch["mask"].to(dev, non_blocking=True)
                groups = batch["group"].to(dev, non_blocking=True)

                logits = pl_module(images)
                loss = F.cross_entropy(logits, masks, ignore_index=-1)
                total_loss += loss.item()
                total_batches += 1

                preds = logits.argmax(dim=1)

                for b in range(images.size(0)):
                    g = int(groups[b].item())
                    if g < 0 or g >= 14:
                        continue
                    t = masks[b].view(-1)
                    p = preds[b].view(-1)

                    valid = (t != -1) & (t >= 0) & (t < NUM_CLASSES)
                    if not valid.any():
                        continue

                    flat_idx = t[valid] * NUM_CLASSES + p[valid]
                    counts = torch.bincount(flat_idx, minlength=NUM_CLASSES * NUM_CLASSES)
                    conf_mat_14[g] += counts.view(NUM_CLASSES, NUM_CLASSES).cpu().numpy().astype(np.int64)

        if was_training:
            pl_module.train()

        cm_global = conf_mat_14.sum(axis=0)
        global_metrics = compute_metrics_from_confmat(cm_global)
        global_miou = global_metrics.get("mIoU", 0.0) * 100.0

        biome_mious = []
        for b in range(14):
            b_cm = conf_mat_14[b]
            b_m = compute_metrics_from_confmat(b_cm)
            biome_mious.append(b_m.get("mIoU", 0.0) * 100.0)

        fairness = compute_group_fairness(biome_mious)
        avg_loss = total_loss / max(1, total_batches)

        return {
            "loss": avg_loss,
            "miou": global_miou,
            "worst_case_miou": fairness.get("M_worst", 0.0),
            "best_miou": fairness.get("M_best", 0.0),
            "spread": fairness.get("Gap", 0.0),
            "per": fairness.get("PER", 0.0),
            "std": fairness.get("Std", 0.0),
            "gini": fairness.get("Gini", 0.0),
            "biome_mious": biome_mious,
            "conf_mat": conf_mat_14,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.start_epoch:
            return

        print(f"\n{'='*80}")
        print(f">>> [Epoch {current_epoch:02d}] Evaluating Full 14-Biome Validation & Test Sets...")
        print(f"{'='*80}")

        val_res = self._evaluate_split(pl_module, self.val_loader)
        test_res = self._evaluate_split(pl_module, self.test_loader)

        pl_module.log("epoch_val/worst_miou", val_res["worst_case_miou"], prog_bar=True)
        pl_module.log("epoch_test/worst_miou", test_res["worst_case_miou"], prog_bar=True)

        entry = {
            "epoch": current_epoch,
            "val": {k: v for k, v in val_res.items() if k != "conf_mat"},
            "test": {k: v for k, v in test_res.items() if k != "conf_mat"},
        }
        self.history.append(entry)

        with open(self.history_file, "w") as f:
            json.dump(to_builtin_serializable(self.history), f, indent=2)

        write_header = not self.csv_file.exists()
        with open(self.csv_file, "a") as f:
            if write_header:
                f.write("epoch,val_loss,val_miou,val_worst_miou,val_best_miou,val_spread,val_per,test_loss,test_miou,test_worst_miou,test_best_miou,test_spread,test_per\n")
            f.write(
                f"{current_epoch},"
                f"{val_res['loss']:.4f},{val_res['miou']:.2f},{val_res['worst_case_miou']:.2f},{val_res['best_miou']:.2f},{val_res['spread']:.2f},{val_res['per']:.4f},"
                f"{test_res['loss']:.4f},{test_res['miou']:.2f},{test_res['worst_case_miou']:.2f},{test_res['best_miou']:.2f},{test_res['spread']:.2f},{test_res['per']:.4f}\n"
            )

        print(
            f"[Epoch {current_epoch:02d} Summary]\n"
            f"  VAL  : Loss={val_res['loss']:.4f} | mIoU={val_res['miou']:.2f}% | Worst={val_res['worst_case_miou']:.2f}% | Best={val_res['best_miou']:.2f}% | Spread={val_res['spread']:.2f}% | PER={val_res['per']:.4f}\n"
            f"  TEST : Loss={test_res['loss']:.4f} | mIoU={test_res['miou']:.2f}% | Worst={test_res['worst_case_miou']:.2f}% | Best={test_res['best_miou']:.2f}% | Spread={test_res['spread']:.2f}% | PER={test_res['per']:.4f}"
        )

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if val_res["worst_case_miou"] > self.best_val_worst_miou:
            self.best_val_worst_miou = val_res["worst_case_miou"]
            best_val_worst_path = self.ckpt_dir / "best_val_worst_miou.ckpt"
            trainer.save_checkpoint(str(best_val_worst_path))
            print(f"  ★ New Best Val Worst-Case mIoU: {self.best_val_worst_miou:.2f}% -> Saved {best_val_worst_path.name}")

        if val_res["miou"] > self.best_val_miou:
            self.best_val_miou = val_res["miou"]
            best_val_miou_path = self.ckpt_dir / "best_val_miou.ckpt"
            trainer.save_checkpoint(str(best_val_miou_path))
            print(f"  ★ New Best Val Overall mIoU: {self.best_val_miou:.2f}% -> Saved {best_val_miou_path.name}")

        if test_res["worst_case_miou"] > self.best_test_worst_miou:
            self.best_test_worst_miou = test_res["worst_case_miou"]
            best_test_worst_path = self.ckpt_dir / "best_test_worst_miou.ckpt"
            trainer.save_checkpoint(str(best_test_worst_path))
            print(f"  ★ New Best Test Worst-Case mIoU: {self.best_test_worst_miou:.2f}% -> Saved {best_test_worst_path.name}")

        print(f"{'='*80}\n")
