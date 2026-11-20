"""One reporting protocol for every backbone and method.

Per-epoch test evaluation is intentional. Only validation scores select
checkpoints and multi-label thresholds. Undefined JSON values are null.
"""

from contextlib import contextmanager
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from fairrsfm.evaluation import metrics as m
from fairrsfm.evaluation.serialization import json_safe


def record_config(output_dir, script, args):
    script = Path(script)
    package = Path(__file__).resolve().parents[1]
    sources = [p for folder in ("training", "methods", "data", "evaluation")
               for p in sorted((package / folder).glob("*.py"))]
    write_json(Path(output_dir) / "training_config.json", {
        "script": str(script.resolve()), "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "shared_source_sha256": {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "arguments": vars(args), "metric_protocol": "present_gt_v1", "per_epoch_test": True,
    })


def write_json(path, payload):
    """Replace a complete JSON document atomically; keep the previous on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    temporary.replace(path)


@contextmanager
def evaluating(model):
    # Preserve individual submodule modes, including the frozen encoder.
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def macro_f1_present(truth, prediction, **kwargs):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim == 2:
        return m.compute_subgroup_macro_f1_multilabel(truth, prediction, truth.shape[1])
    return m.compute_subgroup_macro_f1(truth, prediction)


def finalize_selected(trainer, model):
    """Evaluate validation-selected checkpoints without restoring callback state.

    Snapshot every path first. Lightning callback state in older checkpoints
    can refer to files subsequently removed by top-k retention.
    """
    callbacks = [c for c in trainer.callbacks if getattr(c, "test_loader", None) is not None
                 and hasattr(c, "_compute_and_log")]
    if len(callbacks) != 1:
        raise RuntimeError(f"Expected one test-reporting callback, found {len(callbacks)}")
    callback = callbacks[0]
    callback.bootstrap_replicates = 1000
    choices = [(str(c.monitor), str(c.best_model_path)) for c in trainer.callbacks
               if getattr(c, "best_model_path", "") and
               (str(c.monitor).startswith("val/benchmark_") or str(c.monitor).startswith("val/worst_case_"))]
    choices.sort(key=lambda item: not item[0].startswith("val/benchmark_"))
    if not choices:
        raise RuntimeError("No validation-selected checkpoint was saved")
    run_test = next(getattr(callback, name) for name in
                    ("_eval_test_split_and_log", "_eval_test_split", "_run_test_evaluation")
                    if hasattr(callback, name))

    def restore(path):
        checkpoint = trainer.strategy.load_checkpoint(path)
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=model.strict_loading)
        return checkpoint

    for monitor, path in choices:
        checkpoint = restore(path)
        epoch = int(checkpoint["epoch"])
        history = getattr(callback, "val_history", [])
        if hasattr(callback, "opt_thresholds"):
            matches = [r for r in history if r.get("epoch") == epoch and "optimal_thresholds" in r]
            if not matches:
                raise ValueError(f"No saved validation thresholds for checkpoint epoch {epoch}")
            callback.opt_thresholds = np.asarray(matches[-1]["optimal_thresholds"])
            callback.threshold_epoch = epoch
        tb = trainer.logger.experiment if trainer.logger else None
        with evaluating(model), trainer.precision_plugin.forward_context():
            run_test(trainer, model, epoch, tb)
        record = callback.test_history[-1]
        target = "test_selected.json" if monitor.startswith("val/benchmark_") else "test_selected_worst.json"
        write_json(Path(callback.output_dir) / target,
                   {"checkpoint": path, "checkpoint_epoch": epoch, "selection": monitor, "metrics": record})
    # Keep the model and any subsequent weight export on the primary checkpoint.
    restore(choices[0][1])


def bootstrap_group_f1(source, truth, pred, num_classes, task, ignore_index, replicates):
    """Resample images/patches using cached counts, never independent pixels."""
    if task == "bigearthnet":
        gt, predicted = truth.astype(int), pred.astype(int)
        true_positive = gt * predicted
    elif task == "eurosat":
        gt = np.eye(num_classes, dtype=int)[truth.astype(int)]
        predicted = np.eye(num_classes, dtype=int)[pred.astype(int)]
        true_positive = gt * predicted
    else:
        gt, predicted, true_positive = [], [], []
        for yt, yp in zip(source["true"], source["pred"]):
            yt, yp = np.asarray(yt).ravel(), np.asarray(yp).ravel()
            valid = yt != ignore_index
            yt, yp = yt[valid].astype(int), yp[valid].astype(int)
            gt.append(np.bincount(yt, minlength=num_classes))
            predicted.append(np.bincount(yp, minlength=num_classes))
            true_positive.append(np.bincount(yt[yt == yp], minlength=num_classes))
        gt, predicted, true_positive = map(np.asarray, (gt, predicted, true_positive))
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(replicates):
        idx = rng.integers(0, len(gt), size=len(gt))
        g, p, tp = gt[idx].sum(0), predicted[idx].sum(0), true_positive[idx].sum(0)
        active = g > 0
        if active.any():
            scores.append(float(np.mean(2 * tp[active] / (g[active] + p[active]))))
    return np.percentile(scores, [2.5, 97.5]).tolist() if scores else [None, None]


def _array(value, multilabel=False):
    a = np.asarray(value)
    return a.reshape(-1, a.shape[-1]) if multilabel else a.reshape(-1)


def support_diagnostics(truth, pred, num_classes, *, multilabel=False, sample_count=None):
    """Confusion counts, GT/union F1, and support without a per-pixel Python loop."""
    if multilabel:
        true_count = truth.sum(axis=0).astype(np.int64)
        pred_count = pred.sum(axis=0).astype(np.int64)
        tp = ((truth == 1) & (pred == 1)).sum(axis=0)
    else:
        true_count = np.bincount(truth.astype(int), minlength=num_classes)
        pred_count = np.bincount(pred.astype(int), minlength=num_classes)
        tp = np.bincount(truth[truth == pred].astype(int), minlength=num_classes)
    den = true_count + pred_count
    scores = np.divide(2 * tp, den, out=np.zeros(num_classes, dtype=float), where=den > 0)
    active, union = true_count > 0, den > 0
    return {
        "sample_count": sample_count if sample_count is not None else len(truth),
        "low_support": sample_count is not None and sample_count < 30,
        "ground_truth_class_ids": np.flatnonzero(active).tolist(),
        "union_class_ids": np.flatnonzero(union).tolist(),
        "ground_truth_class_count": int(active.sum()),
        "union_class_count": int(union.sum()),
        "macro_f1_gt_present": float(scores[active].mean()) if active.any() else None,
        "macro_f1_union": float(scores[union].mean()) if union.any() else None,
        "per_class": [
            {"class_id": c, "true_count": int(true_count[c]), "predicted_count": int(pred_count[c]),
             "true_positive_count": int(tp[c]), "f1": float(scores[c]) if union[c] else None}
            for c in range(num_classes)
        ],
    }


def report_epoch(callback, biome_data, epoch, tb, tb_prefix, label, pl_module=None,
                 *, task, ignore_index=-1, persist=True):
    """Score a validation or test split and persist one consistent epoch record."""
    if label not in {"val", "test"}:
        raise ValueError(f"Unsupported split {label}")
    multilabel, segmentation = task == "bigearthnet", task in {"sa_crop", "mmearth"}
    num_classes = callback.num_classes
    data = {}
    for group, source in biome_data.items():
        if not len(source["true"]):
            continue
        truth = _array(source["true"], multilabel)
        row = {"true": truth}
        if "pred" in source and len(source["pred"]):
            row["pred"] = _array(source["pred"], multilabel)
        if "probs" in source and len(source["probs"]):
            probs = np.asarray(source["probs"])
            row["probs"] = probs.reshape(-1, num_classes)
        elif "probs_max" in source:
            row["probs"] = _array(source["probs_max"])
        row["sample_count"] = len(source.get("sample_ids", source["true"]))
        if segmentation:
            valid = truth != ignore_index
            row = {k: (v[valid] if isinstance(v, np.ndarray) else v) for k, v in row.items()}
        data[group] = row
    if not data:
        raise ValueError(f"No observations for {task} {label}, epoch {epoch}")
    all_true = np.concatenate([d["true"] for d in data.values()])
    if multilabel:
        if label == "val":
            callback.opt_thresholds = m.find_optimal_thresholds(
                all_true, np.concatenate([d["probs"] for d in data.values()]), num_classes)
            callback.threshold_epoch = int(epoch)
        if getattr(callback, "threshold_epoch", None) != int(epoch):
            raise ValueError("Test thresholds must come from validation of the same epoch")
        for row in data.values():
            row["pred"] = (row["probs"] >= callback.opt_thresholds).astype(int)
    all_pred = np.concatenate([d["pred"] for d in data.values()])

    def score(row):
        yt, yp = row["true"], row["pred"]
        if multilabel:
            f1 = m.compute_subgroup_macro_f1_multilabel(yt, yp, num_classes)
            acc = float((yt == yp).all(axis=1).mean())
        else:
            f1 = m.compute_subgroup_macro_f1(yt, yp, num_classes)
            acc = float((yt == yp).mean()) if len(yt) else float("nan")
        result = {"macro_f1": f1, "accuracy": acc, "sample_count": row["sample_count"]}
        if segmentation:
            result["miou"] = m.compute_miou(yt, yp, num_classes, ignore_index)
            result["valid_pixel_count"] = len(yt)
        pr = row.get("probs")
        if pr is not None:
            if segmentation:
                confidence = pr.max(axis=-1) if pr.ndim == 2 else pr
                result["ece"] = m.compute_ece_seg(yt, confidence, yp, ignore_index)
            elif multilabel:
                result["ece"] = m.compute_ece_multilabel(yt, pr)
                result["mAP"] = m.compute_map_multilabel(yt, pr, num_classes)
            else:
                result["ece"] = m.compute_ece_single_label(yt, pr, yp)
                result["mAP"] = m.compute_map_score_single_label(yt, pr, num_classes)
        result["micro_f1"] = float(f1_score(yt, yp, average="micro", zero_division=0)) if len(yt) else None
        return result

    pooled = {"true": all_true, "pred": all_pred,
              "sample_count": sum(d["sample_count"] for d in data.values())}
    if all("probs" in d for d in data.values()):
        pooled["probs"] = np.concatenate([d["probs"] for d in data.values()])
    overall = score(pooled)
    groups, diagnostics = {}, {}
    for group, row in data.items():
        if group == "Unknown":
            continue
        groups[group] = score(row)
        diagnostics[group] = support_diagnostics(row["true"], row["pred"], num_classes,
                                               multilabel=multilabel, sample_count=row["sample_count"])
        replicates = getattr(callback, "bootstrap_replicates", 0)
        if replicates:
            diagnostics[group]["macro_f1_gt_present_ci95"] = bootstrap_group_f1(
                biome_data[group], row["true"], row["pred"], num_classes, task, ignore_index, replicates)
            diagnostics[group]["bootstrap_replicates"] = replicates
            diagnostics[group]["bootstrap_unit"] = "patch" if segmentation else "image"
    shared = sorted(set.intersection(*(set(d["ground_truth_class_ids"]) for d in diagnostics.values()))) if diagnostics else []
    for group, diag in diagnostics.items():
        diag["class_matched_ids"] = shared
        diag["macro_f1_class_matched"] = float(np.mean([diag["per_class"][c]["f1"] for c in shared])) if shared else None
        groups[group]["diagnostics"] = diag
    primary = "miou" if segmentation else "macro_f1"
    primary_scores = {g: v[primary] for g, v in groups.items()}
    fairness = m.compute_fairness_summary(primary_scores)
    group_eces = [v["ece"] for v in groups.values()
                  if v.get("ece") is not None and np.isfinite(v["ece"])]
    ece_disparity = max(group_eces) - min(group_eces) if group_eces else None
    known = {g: {"true": d["true"], "pred": d["pred"]}
             for g, d in data.items() if g != "Unknown"}
    if segmentation:
        eq, dp = m.compute_eqodds_seg(known, num_classes, ignore_index), m.compute_demographic_parity_seg(known, num_classes, ignore_index)
    elif multilabel:
        eq, dp = m.compute_equalized_odds_multilabel(known, num_classes), m.compute_demographic_parity_multilabel(known, num_classes)
    else:
        eq, dp = m.compute_equalized_odds_single_label(known, num_classes), m.compute_demographic_parity_single_label(known, num_classes)
    record = {
        "epoch": int(epoch), "split": label, "task": task, "metric_protocol": "present_gt_v1",
        "overall_macro_f1": overall["macro_f1"], "global_macro_f1": overall["macro_f1"],
        "overall_accuracy": overall["accuracy"], "global_accuracy": overall["accuracy"],
        "overall_micro_f1": overall["micro_f1"],
        "overall_ece": overall.get("ece"), "ece_disparity": ece_disparity,
        "overall_mAP": overall.get("mAP"), "global_map": overall.get("mAP"),
        "overall_miou": overall.get("miou"), "global_miou": overall.get("miou"),
        "sample_count": pooled["sample_count"], "per_group_metrics": groups,
        "worst_case_macro_f1": min((v["macro_f1"] for v in groups.values()), default=float("nan")),
        "worst_case_miou": fairness["worst_case"] if segmentation else None,
        "worst_case": fairness["worst_case"], "best_case": fairness["best_case"],
        "group_mean": fairness["mean"], "group_std": fairness["std"],
        "performance_equity_ratio": fairness["per"], "normalized_fairness_range": fairness["nfr"],
        "gini_coefficient": fairness["gini"], "disparity": fairness["disparity"],
        "equalized_odds_disparity": eq, "demographic_parity_disparity": dp,
        "per_biome_miou": primary_scores if segmentation else {},
        "per_biome_macro_f1": {g: v["macro_f1"] for g, v in groups.items()},
    }
    if multilabel:
        record.update(optimal_thresholds=callback.opt_thresholds.tolist(), threshold_epoch=int(epoch),
                      overall_macro_f1_opt=overall["macro_f1"])
    if label == "val" and pl_module is not None:
        pl_module.val_biome_metrics = primary_scores
        pl_module.log(f"val/benchmark_{primary}", overall[primary], on_step=False, on_epoch=True)
        pl_module.log(f"val/worst_case_{primary}", fairness["worst_case"], on_step=False, on_epoch=True)
    history_name = f"{label}_history"
    history = getattr(callback, history_name, [])
    if not isinstance(history, list):
        history = []
    if not history and persist:
        history_path = Path(callback.output_dir) / f"{label}_results.json"
        if history_path.exists():
            stored = json.loads(history_path.read_text())
            if isinstance(stored, list):
                history = stored
            elif isinstance(stored, dict):
                history = [row.get(label, row) for row in stored.values()]
    history = [r for r in history if r.get("epoch") != int(epoch)] + [json_safe(record)]
    setattr(callback, history_name, history)
    if persist:
        write_json(Path(callback.output_dir) / f"{label}_results.json", history)
    print(f"\n[{task} {label}] epoch={epoch} overall {primary}={overall[primary]:.4f} worst={fairness['worst_case']:.4f}")
    for group, v in groups.items():
        diag = diagnostics[group]
        print(f"  {group}: n={v['sample_count']} F1(GT)={v['macro_f1']:.4f} F1(union)={diag['macro_f1_union']} low_support={diag['low_support']}")
        for cls in diag["per_class"]:
            if cls["true_count"] or cls["predicted_count"]:
                print(f"    class {cls['class_id']}: true={cls['true_count']} pred={cls['predicted_count']} F1={cls['f1']:.4f}")
    if tb is not None and hasattr(tb, "add_scalar"):
        for key, value in overall.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                tb.add_scalar(f"{tb_prefix}/{key}", value, epoch)
        for key, value in record.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                tb.add_scalar(f"{tb_prefix}/{key}", value, epoch)
        for group, values in groups.items():
            tb.add_scalar(f"{tb_prefix}/{group}/{primary}", values[primary], epoch)
    return record
