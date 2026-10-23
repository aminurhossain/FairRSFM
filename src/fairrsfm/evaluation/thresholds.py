"""Load validation-selected multi-label thresholds for a test run."""

import json
from pathlib import Path

import numpy as np


def resolve_checkpoint_epoch(checkpoint_path: str | Path, requested_epoch: int | None = None) -> int | None:
    """Use Lightning checkpoint metadata to prevent mismatched threshold epochs."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if recorded_epoch is None:
        return requested_epoch
    recorded_epoch = int(recorded_epoch)
    if requested_epoch is not None and recorded_epoch != requested_epoch:
        raise ValueError(
            f"Checkpoint epoch {recorded_epoch} disagrees with --thresholds-epoch {requested_epoch}"
        )
    return recorded_epoch


def load_validation_thresholds(path: str | Path, num_classes: int, epoch: int | None = None) -> np.ndarray:
    """Read a threshold vector or select one exact epoch from val_results.json.

    An epoch must be specified when the JSON contains more than one epoch so a
    test checkpoint cannot silently use thresholds from a different checkpoint.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if epoch is None and len(data) != 1:
            raise ValueError("Threshold history has multiple epochs; pass --thresholds-epoch for the selected checkpoint")
        matches = [row for row in data if row.get("epoch") == epoch] if epoch is not None else data
        if len(matches) != 1:
            raise ValueError(f"Expected one validation threshold record for epoch {epoch}, found {len(matches)}")
        data = matches[0]
    if isinstance(data, dict):
        if epoch is not None and "epoch" in data and data["epoch"] != epoch:
            raise ValueError(f"Validation threshold epoch {data['epoch']} disagrees with checkpoint epoch {epoch}")
        data = data.get("optimal_thresholds")
    thresholds = np.asarray(data, dtype=float)
    if thresholds.shape != (num_classes,) or not np.all(np.isfinite(thresholds)):
        raise ValueError(f"Expected {num_classes} finite class thresholds, got shape {thresholds.shape}")
    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Thresholds must lie in [0, 1]")
    return thresholds
