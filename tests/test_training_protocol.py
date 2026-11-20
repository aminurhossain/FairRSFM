import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset

from fairrsfm.training.protocol import (
    MethodTrainingMixin, configure_training_protocol, dataset_sample_ids,
    known_training_dataset, training_loader, MethodObjective,
)


class MetadataDataset(Dataset):
    def __init__(self):
        self.active_partition = SimpleNamespace(partition_dict={"train": ["a", "b", "c", "d"]})
        self.split = "train"

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.ones(2), torch.tensor(i % 2), self.active_partition.partition_dict["train"][i]


def test_training_subset_is_identical_for_all_methods_and_sampler_is_balanced():
    mapping = {"a": "A", "b": "A", "c": "B"}
    signatures = []
    for method in ["erm", "bolp", "dbr", "groupdro"]:
        subset, groups = known_training_dataset(MetadataDataset(), lambda sid: mapping.get(sid))
        signatures.append(dataset_sample_ids(subset))
        loader = training_loader(subset, method, groups=groups, batch_size=2, shuffle=True)
        if method == "groupdro":
            assert loader.sampler.weights.tolist() == [0.5, 0.5, 1.0]
    assert signatures == [["a", "b", "c"]] * 4


@pytest.mark.parametrize("task", ["eurosat", "bigearthnet", "sa_crop", "mmearth"])
@pytest.mark.parametrize("method", ["erm", "bolp", "dbr", "groupdro"])
def test_shared_steps_train_and_evaluate_all_task_method_combinations(task, method):
    class Task(MethodTrainingMixin, pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.model(x)

        def log(self, *args, **kwargs):
            pass

    model = Task()
    configure_training_protocol(model, method, task, {"a": "A", "b": "B"}, SimpleNamespace())
    x = torch.tensor([[2., 0.], [0., 2.]])
    y = torch.tensor([0, 1])
    if task == "bigearthnet":
        y = torch.nn.functional.one_hot(y, 2).float()
    elif task in {"sa_crop", "mmearth"}:
        model.forward = lambda x: model.model(x).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
        y = y[:, None, None].expand(-1, 2, 2).clone()
        y[0, 0, 0] = -1
    batch = (x, y, ["a", "b"])
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.model.weight.grad is not None
    before = {k: v.clone() for k, v in model.method_protocol.state_dict().items() if torch.is_tensor(v)}
    model.validation_step(batch, 0)
    model.test_step(batch, 0)
    after = model.method_protocol.state_dict()
    for key, value in before.items():
        torch.testing.assert_close(value, after[key], equal_nan=True)


def test_all_56_scripts_use_shared_steps_and_filter_training_data():
    root = Path(__file__).resolve().parents[1]
    paths = [p for p in (root / "experiments/scripts").rglob("*.py")
             if p.stem.split("_")[-1] in {"erm", "bolp", "dbr", "groupdro"}]
    assert len(paths) == 56
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        assert "known_training_dataset(" in source, path
        assert "configure_training_protocol(" in source, path
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bases = [ast.unparse(b) for b in cls.bases]
            if "MethodTrainingMixin" in bases:
                assert not any(isinstance(n, ast.FunctionDef) and n.name in
                               {"training_step", "validation_step", "test_step", "on_train_epoch_start"}
                               for n in cls.body), path


def test_all_ignored_patches_do_not_change_dbr_loss_statistics():
    objective = MethodObjective("dbr", "mmearth", {"a": "A"})
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    loss = objective(logits, torch.full((1, 2, 2), -1), ["a"])
    assert loss.item() == 0
    assert objective.dbr.observation_count.item() == 0
    loss.backward()


def test_non_strict_checkpoint_load_requires_head_and_method_state():
    class Task(MethodTrainingMixin, pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)
            configure_training_protocol(self, "dbr", "eurosat", {"a": "A"}, SimpleNamespace())

    model = Task()
    state = model.state_dict()
    model.load_state_dict(state, strict=False)
    for missing in ["head.weight", "method_protocol.dbr.weights"]:
        incomplete = dict(state)
        del incomplete[missing]
        with pytest.raises(ValueError, match="required head or shared method state"):
            model.load_state_dict(incomplete, strict=False)


def test_resume_rejects_changed_training_biome_assignments():
    original = MethodObjective("erm", "eurosat", {"a": "A", "b": "B"})
    changed = MethodObjective("erm", "eurosat", {"a": "B", "b": "A"})
    with pytest.raises(ValueError, match="training subset differs"):
        changed.load_state_dict(original.state_dict())


def test_dbr_epoch_boundary_and_resume_before_epoch_end_update():
    class Task(MethodTrainingMixin, pl.LightningModule):
        epoch = 0

        @property
        def current_epoch(self):
            return self.epoch

        def __init__(self):
            super().__init__()
            configure_training_protocol(self, "dbr", "eurosat", {"a": "A", "b": "B"}, SimpleNamespace())

    model = Task()
    model.method_protocol.dbr.observe(torch.tensor([2., 8.]), ["A", "B"])
    import copy
    pre_update = copy.deepcopy(model.state_dict())
    model.on_train_epoch_end()
    expected = model.method_protocol.dbr.weights.clone()
    resumed = Task()
    resumed.load_state_dict(pre_update)
    resumed.epoch = 1
    resumed.on_train_epoch_start()
    torch.testing.assert_close(resumed.method_protocol.dbr.weights, expected)
    resumed.on_train_epoch_start()
    torch.testing.assert_close(resumed.method_protocol.dbr.weights, expected)


@pytest.mark.parametrize("task", ["sa_crop", "mmearth"])
def test_actual_dofa_upernet_adapter_transforms_final_layer_only(task):
    from fairrsfm.methods.bolp import BOLPProjector
    from fairrsfm.training.protocol import forward_with_sample_ids

    root = Path(__file__).resolve().parents[1]
    path = root / f"experiments/scripts/dofa/{task}_upernet/bolp.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DOFAUperNetBOLPModel")
    isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = dict(nn=torch.nn, torch=torch, F=torch.nn.functional, SharedBOLPProjector=BOLPProjector)
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)

    class Encoder(torch.nn.Module):
        def forward_intermediates(self, x, wavelengths, out_indices):
            return [x + index for index in range(4)]

    class Head(torch.nn.Module):
        def forward(self, features):
            self.features = features
            return features[-1]

    base = torch.nn.Module()
    base.encoder, base.head = Encoder(), Head()
    base.wavelengths, base.out_indices = [], [0, 1, 2, 3]
    centroids = {"A": torch.tensor([1., 1.]), "B": torch.tensor([-1., -1.])}
    projection = torch.diag(torch.tensor([0., 1.]))
    cls = namespace[cls.name]
    if task == "sa_crop":
        adapter = cls(base, projection, torch.zeros(2), centroids, {})
        rank_ablation = cls(base, torch.eye(2), torch.zeros(2), centroids, {}, rank=0)
        assert rank_ablation.projector.get_extra_state()["k"] == 0
    else:
        adapter = cls(base, BOLPProjector(centroids, torch.zeros(2), projection, 1))

    class Task(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = adapter
            self._protocol_lookup = lambda sid: "A"

        def forward(self, x):
            return self.model(x)

    images = torch.arange(16.).reshape(2, 2, 2, 2)
    output = forward_with_sample_ids(Task(), images, ["a", "b"])
    for index in range(3):
        torch.testing.assert_close(base.head.features[index], images + index)
    torch.testing.assert_close(output[:, 0], torch.zeros(2, 2, 2))
    torch.testing.assert_close(output[:, 1], images[:, 1] + 2)
    assert adapter.projector._sample_biomes is None
