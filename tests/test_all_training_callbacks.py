"""Exercise each actual callback's validation-to-test path without loading backbones."""

import ast
from collections import defaultdict
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import Callback

from fairrsfm.training.reporting import report_epoch, evaluating, macro_f1_present, write_json
from fairrsfm.evaluation.serialization import json_safe
from fairrsfm.evaluation.metrics import find_optimal_thresholds
from fairrsfm.training.protocol import forward_with_sample_ids

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "experiments/scripts").rglob("*.py")
                 if p.name != "mmearth_dynamic_world_datamodule.py")


def callback_class(path):
    tree = ast.parse(path.read_text())
    candidates = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        if "_compute_and_log" in methods and any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "test_loader" for t in n.targets)
            for n in ast.walk(cls)
        ):
            candidates.append(cls)
    assert len(candidates) == 1, path
    cls = candidates[0]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = dict(Callback=Callback, np=np, torch=torch, Path=Path, json=json,
                     defaultdict=defaultdict, IGNORE_INDEX=-1, report_epoch=report_epoch,
                     evaluating=evaluating, macro_f1_present=macro_f1_present,
                     write_json=write_json, json_safe=json_safe,
                     forward_with_sample_ids=forward_with_sample_ids,
                     F=torch.nn.functional)
    namespace["find_optimal_thresholds"] = find_optimal_thresholds
    namespace["get_macro_biome"] = lambda mapping, sid: mapping[str(sid)]
    namespace["tile_id_to_macro_group"] = lambda sid, mapping: mapping[str(sid)]
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[cls.name]


@pytest.mark.parametrize("path", [p for p in SCRIPTS if not ("mmearth_20k" in str(p) and p.name != "mmearth_groupdro.py")], ids=lambda p: str(p.relative_to(ROOT)))
def test_every_validation_epoch_also_scores_test(path, tmp_path):
    task = next(t for t in ("eurosat", "bigearthnet", "sa_crop", "mmearth") if t in str(path))
    seg, multi = task in ("sa_crop", "mmearth"), task == "bigearthnet"
    logits = torch.tensor([[3., -2., -3.], [-2., 3., -3.], [-3., -2., 3.]])
    truth = torch.eye(3) if multi else torch.tensor([0, 1, 2])
    if seg:
        logits = logits[:, :, None, None].expand(-1, -1, 2, 2).contiguous()
        truth = truth[:, None, None].expand(-1, 2, 2).contiguous()
    ids = ["a", "b", "unknown"]
    batch = (logits, truth, ids)
    if task == "mmearth" and "satmae" not in str(path):
        batch = {"image": logits, "mask": truth[:, None], "filename": ids}

    class Model(torch.nn.Module):
        device = torch.device("cpu")
        logger = None

        @property
        def model(self):
            return lambda x: SimpleNamespace(output=x)

        def forward(self, x, sample_ids=None):
            return x

        def log(self, key, value, **kwargs):
            trainer.callback_metrics[key] = value

    model = Model()
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0, logger=None,
                              callback_metrics=defaultdict(float))
    cls = callback_class(path)
    mapping = {"a": "A", "b": "B", "unknown": "Unknown"}
    possible = dict(biome_map=mapping, tile_info_dict=mapping,
                    num_classes=3, class_names=["zero", "one", "two"], output_dir=tmp_path,
                    ignore_index=-1, test_loader=[batch])
    kwargs = {k: v for k, v in possible.items() if k in inspect.signature(cls).parameters}
    cb = cls(**kwargs)
    if hasattr(cb, "on_validation_epoch_start"):
        cb.on_validation_epoch_start(trainer, model)
    outputs = {"logits": logits, "y": truth, "masks": truth, "sample_ids": ids}
    cb.on_validation_batch_end(trainer, model, outputs, batch, 0)
    cb.on_validation_epoch_end(trainer, model)
    val = json.loads((tmp_path / "val_results.json").read_text())[-1]
    test = json.loads((tmp_path / "test_results.json").read_text())[-1]
    assert test["epoch"] == val["epoch"] == 0
    assert test["split"] == "test"
    assert test["overall_macro_f1"] == pytest.approx(1.0)
    assert test["sample_count"] == 3
    assert np.isfinite(test["overall_ece"])
    assert np.isfinite(test["ece_disparity"])
    assert set(test["per_group_metrics"]) == {"A", "B"}
    assert test["per_group_metrics"]["A"]["diagnostics"]["per_class"][0]["true_count"] > 0
    if multi:
        assert test["optimal_thresholds"] == val["optimal_thresholds"]
        assert test["threshold_epoch"] == 0
    assert model.training  # direct test evaluation restores the original mode
