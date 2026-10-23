"""Checks the shared scoring contract used by checkpoint evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fairrsfm.evaluation.checkpoint import load_finetuned_state
from fairrsfm.evaluation.serialization import json_safe
from fairrsfm.evaluation.metrics import (
    compute_ece_multilabel,
    compute_fairness_summary,
    compute_macro_f1_seg,
    compute_miou,
    compute_subgroup_macro_f1,
    compute_subgroup_macro_f1_multilabel,
    single_label_group_diagnostics,
)
from fairrsfm.evaluation.thresholds import load_validation_thresholds, resolve_checkpoint_epoch


class MetricContractTests(unittest.TestCase):
    def test_group_diagnostics_distinguish_spurious_from_missed_classes(self):
        groups = {
            "a": (np.array([0, 0, 1]), np.array([0, 2, 0])),
            "b": (np.array([0, 1]), np.array([0, 1])),
        }
        result = single_label_group_diagnostics(groups, 3, bootstrap_replicates=20)
        # Training writes diagnostics directly to JSON, without a NumPy encoder.
        json.dumps(result, allow_nan=False)
        a = result["a"]
        self.assertEqual(a["per_class"][2]["true_count"], 0)
        self.assertEqual(a["per_class"][2]["predicted_count"], 1)
        self.assertEqual(a["per_class"][1]["true_count"], 1)
        self.assertEqual(a["per_class"][1]["predicted_count"], 0)
        self.assertEqual(a["class_matched_ids"], [0, 1])
        self.assertLess(a["macro_f1_union"], a["macro_f1_gt_present"])
        self.assertEqual(len(a["macro_f1_gt_present_ci95"]), 2)

    def test_group_macro_f1_excludes_classes_absent_from_ground_truth(self):
        truth = np.array([0, 0])
        pred = np.array([0, 1])
        self.assertAlmostEqual(compute_subgroup_macro_f1(truth, pred, num_classes=3), 2 / 3)
        self.assertAlmostEqual(
            compute_subgroup_macro_f1(truth, pred, num_classes=3, present_classes_only=False),
            2 / 9,
        )

    def test_multilabel_group_macro_f1_ignores_absent_label_even_if_predicted(self):
        truth = np.array([[1, 0], [1, 0]])
        pred = np.array([[1, 1], [0, 0]])
        self.assertAlmostEqual(compute_subgroup_macro_f1_multilabel(truth, pred, 2), 2 / 3)
        self.assertAlmostEqual(
            compute_subgroup_macro_f1_multilabel(truth, pred, 2, active_only=False),
            1 / 3,
        )

    def test_segmentation_group_macro_f1_uses_present_valid_pixel_classes(self):
        truth = np.array([0, 0, -1])
        pred = np.array([0, 1, 1])
        self.assertAlmostEqual(compute_macro_f1_seg(truth, pred, 3, ignore_index=-1), 2 / 3)

    def test_nfr_uses_group_mean_even_when_overall_score_is_given(self):
        summary = compute_fairness_summary({"a": 0.2, "b": 0.8}, global_score=0.9)
        self.assertAlmostEqual(summary["nfr"], 1.2)
        self.assertAlmostEqual(summary["per"], 0.25)

    def test_multilabel_ece_is_per_label(self):
        truth = np.array([[1, 0], [0, 1]])
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.assertAlmostEqual(compute_ece_multilabel(truth, probs), 0.15)
        self.assertAlmostEqual(
            compute_subgroup_macro_f1_multilabel(truth, (probs >= 0.5).astype(int), 2), 1.0
        )

    def test_mmearth_ignore_value_does_not_exclude_class_zero(self):
        truth = np.array([0, -1, 1])
        pred = np.array([0, 2, 0])
        # Class 0 IoU is 1/2; class 1 IoU is 0. The ignored pixel contributes nothing.
        self.assertAlmostEqual(compute_miou(truth, pred, 3, ignore_index=-1), 0.25)

    def test_threshold_history_requires_exact_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val_results.json"
            path.write_text(json.dumps([
                {"epoch": 1, "optimal_thresholds": [0.2, 0.3]},
                {"epoch": 2, "optimal_thresholds": [0.4, 0.5]},
            ]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_validation_thresholds(path, 2)
            np.testing.assert_allclose(load_validation_thresholds(path, 2, epoch=2), [0.4, 0.5])

    def test_slim_checkpoint_may_omit_backbone_but_not_head(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 1)

        model = ToyModel()
        load_finetuned_state(model, {"head.weight": model.head.weight, "head.bias": model.head.bias})
        with self.assertRaisesRegex(ValueError, "missing non-backbone keys"):
            load_finetuned_state(model, {})

    def test_checkpoint_epoch_must_match_threshold_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.ckpt"
            torch.save({"epoch": 3, "state_dict": {}}, checkpoint_path)
            self.assertEqual(resolve_checkpoint_epoch(checkpoint_path), 3)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                resolve_checkpoint_epoch(checkpoint_path, requested_epoch=2)

    def test_undefined_metrics_are_standard_json_null(self):
        result = json_safe({"per_class_f1": np.array([1.0, np.nan]), "ece": np.inf})
        self.assertEqual(json.dumps(result, allow_nan=False), '{"per_class_f1": [1.0, null], "ece": null}')


if __name__ == "__main__":
    unittest.main()
