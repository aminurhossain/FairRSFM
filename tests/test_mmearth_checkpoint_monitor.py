"""Exercise real training callbacks through Lightning's validation lifecycle."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import pyproj
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


class TinyTask(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        # TerraTorch owns this name; the benchmark callback must coexist with it.
        self.log("val/mIoU", 0.2, prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("method", ["erm", "bolp", "dbr"])
def test_benchmark_checkpoint_does_not_collide_with_terratorch(tmp_path, method):
    os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
    root = Path(__file__).resolve().parents[1]
    path = root / f"experiments/scripts/prithvi/segmentation/mmearth_20k/mmearth_{method}.py"
    spec = importlib.util.spec_from_file_location(f"mmearth_monitor_{method}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=2)
    callback = module.EpochMetricsSaverCallback(
        val_loader=loader, tile_info={}, num_classes=2, out_dir=tmp_path,
        device=torch.device("cpu"),
    )
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints", monitor="val/benchmark_miou", mode="max",
        filename="{epoch:02d}-{val/benchmark_miou:.4f}", auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[callback, checkpoint], num_sanity_val_steps=0,
    )
    with patch.object(module, "evaluate_split", return_value={"global_miou": 0.73}):
        trainer.fit(TinyTask(), loader, loader)
    assert Path(checkpoint.best_model_path).is_file()
    assert "0.7300" in checkpoint.best_model_path
    assert checkpoint.best_model_score.item() == pytest.approx(0.73)
    saved = json.loads((tmp_path / "val_results.json").read_text())
    assert saved["epoch_0"]["val"]["global_miou"] == 0.73
