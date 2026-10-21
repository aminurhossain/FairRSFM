from __future__ import annotations

import argparse
import json
import os
from datetime import date
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
_env_root = os.environ.get("MMEARTH_DATA_DIR", "/data1/aminur/om/MMearth20k/hardcoded")
DEFAULT_DATA_ROOT = _env_root if Path(_env_root).is_dir() else str(SCRIPT_DIR / "hardcoded")

SENTINEL2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
]

DEFAULT_SENTINEL2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B8",
    "B9",
    "B11",
    "B12",
]

DYNAMIC_WORLD_CLASSES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {root}, found {len(matches)}.")
    return matches[0]


def validate_indices(name: str, indices: Sequence[int], num_rows: int) -> list[int]:
    normalized = [int(index) for index in indices]
    invalid = [index for index in normalized if index < 0 or index >= num_rows]
    if invalid:
        preview = invalid[:5]
        raise ValueError(
            f"Split {name!r} contains indices outside HDF5 row range 0..{num_rows - 1}: {preview}"
        )
    return normalized


class MMEarthDynamicWorldSegmentationDataset(Dataset):
    """MMEarth Sentinel-2 to Dynamic World semantic segmentation dataset."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        sentinel2_bands: Sequence[str] | None = None,
        image_size: int | None = None,
        normalize: bool = True,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.h5_path = find_one(self.data_root, "*fixed.h5")
        self.splits_path = find_one(self.data_root, "*splits.json")
        self.band_stats_path = find_one(self.data_root, "*band_stats.json")

        self.splits = json.loads(self.splits_path.read_text(encoding="utf-8"))
        if indices is None:
            if split not in self.splits:
                raise ValueError(f"Unknown split {split!r}; expected one of {sorted(self.splits)}.")
            self.indices = list(self.splits[split])
        else:
            self.indices = list(indices)

        self.sentinel2_bands = list(sentinel2_bands or DEFAULT_SENTINEL2_BANDS)
        unknown_bands = [band for band in self.sentinel2_bands if band not in SENTINEL2_BANDS]
        if unknown_bands:
            raise ValueError(f"Unknown Sentinel-2 bands {unknown_bands}; expected subset of {SENTINEL2_BANDS}.")
        self.band_indices = [SENTINEL2_BANDS.index(band) for band in self.sentinel2_bands]
        self.image_size = image_size
        self.normalize = normalize
        self.band_stats = json.loads(self.band_stats_path.read_text(encoding="utf-8"))
        self.tile_info = json.loads(find_one(self.data_root, "*tile_info.json").read_text(encoding="utf-8"))
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row_index = self.indices[index]
        h5 = self._file()

        metadata = h5["metadata"][row_index]
        tile_id = metadata["tile_id"].decode("utf-8")
        s2_type = metadata["S2_type"].decode("utf-8")
        tile_info = self.tile_info[tile_id]

        image = np.asarray(h5["sentinel2"][row_index], dtype=np.float32)[self.band_indices]
        if self.normalize:
            stats_key = "sentinel2_l2a" if s2_type == "l2a" else "sentinel2_l1c"
            means = np.asarray(self.band_stats[stats_key]["mean"], dtype=np.float32)[self.band_indices]
            stds = np.asarray(self.band_stats[stats_key]["std"], dtype=np.float32)[self.band_indices]
            image = (image - means[:, None, None]) / stds[:, None, None]

        # MMEarth convention: raw Dynamic World 0 is no-data, 1..9 become classes 0..8.
        mask = np.asarray(h5["dynamic_world"][row_index, 0], dtype=np.int64)
        mask = np.where(mask == 0, -1, mask - 1)
        mask = np.where(mask > 8, -1, mask)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if self.image_size is not None and image_tensor.shape[-1] != self.image_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "temporal_coords": self._temporal_coords(tile_info["S2_DATE"]),
            "location_coords": torch.tensor([tile_info["lat"], tile_info["lon"]], dtype=torch.float32),
            "filename": tile_id,
        }

    @staticmethod
    def _temporal_coords(s2_date: str) -> torch.Tensor:
        parsed = date.fromisoformat(s2_date)
        return torch.tensor([[parsed.year, parsed.timetuple().tm_yday - 1]], dtype=torch.float32)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class MMEarthDynamicWorldSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        sentinel2_bands: Sequence[str] | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        image_size: int | None = None,
        normalize: bool = True,
        fallback_val_fraction: float = 0.1,
        fallback_test_fraction: float = 0.0,
        seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.sentinel2_bands = list(sentinel2_bands or DEFAULT_SENTINEL2_BANDS)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.normalize = normalize
        self.fallback_val_fraction = fallback_val_fraction
        self.fallback_test_fraction = fallback_test_fraction
        self.seed = seed
        self._split_indices: dict[str, list[int]] | None = None
        self.train_dataset: MMEarthDynamicWorldSegmentationDataset | None = None
        self.val_dataset: MMEarthDynamicWorldSegmentationDataset | None = None
        self.test_dataset: MMEarthDynamicWorldSegmentationDataset | None = None
        self.predict_dataset: MMEarthDynamicWorldSegmentationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self._split_indices = self._make_split_indices()
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("val")
        if stage in (None, "validate"):
            self.val_dataset = self._dataset("val")
        if stage in (None, "test"):
            self.test_dataset = self._dataset("test")
        if stage in (None, "predict"):
            self.predict_dataset = self._dataset("test")

    def _dataset(self, split: str) -> MMEarthDynamicWorldSegmentationDataset:
        indices = None if self._split_indices is None else self._split_indices[split]
        return MMEarthDynamicWorldSegmentationDataset(
            data_root=self.data_root,
            split=split,
            sentinel2_bands=self.sentinel2_bands,
            image_size=self.image_size,
            normalize=self.normalize,
            indices=indices,
        )

    def _make_split_indices(self) -> dict[str, list[int]]:
        data_root = Path(self.data_root)
        splits_path = find_one(data_root, "*splits.json")
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        with h5py.File(find_one(data_root, "*fixed.h5"), "r") as h5:
            num_rows = int(h5["metadata"].shape[0])

        train_indices = validate_indices("train", splits.get("train", []), num_rows)
        val_indices = validate_indices("val", splits.get("val", []), num_rows)
        test_indices = validate_indices("test", splits.get("test", []), num_rows)

        needs_val = not val_indices and self.fallback_val_fraction > 0
        needs_test = not test_indices and self.fallback_test_fraction > 0
        if not needs_val and not needs_test:
            return {"train": train_indices, "val": val_indices, "test": test_indices}

        if not train_indices:
            return {"train": [], "val": [], "test": test_indices}

        rng = np.random.default_rng(self.seed)
        permuted = np.asarray(train_indices)
        rng.shuffle(permuted)
        cursor = 0

        if needs_val:
            val_size = max(1, int(round(len(permuted) * self.fallback_val_fraction)))
            val_indices = permuted[cursor : cursor + val_size].astype(int).tolist()
            cursor += val_size

        if needs_test:
            test_size = max(1, int(round(len(permuted) * self.fallback_test_fraction)))
            test_indices = permuted[cursor : cursor + test_size].astype(int).tolist()
            cursor += test_size

        if cursor >= len(permuted):
            raise ValueError(
                "Fallback val/test fractions leave no training samples. "
                f"val_fraction={self.fallback_val_fraction}, test_fraction={self.fallback_test_fraction}"
            )

        val_set = set(val_indices)
        test_set = set(test_indices)
        fallback_train = [idx for idx in train_indices if idx not in val_set and idx not in test_set]
        return {"train": fallback_train, "val": val_indices, "test": test_indices}

    def teardown(self, stage: str | None = None) -> None:
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset):
            if dataset is not None:
                dataset.close()

    @staticmethod
    def _collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "temporal_coords": torch.stack([item["temporal_coords"] for item in batch]),
            "location_coords": torch.stack([item["location_coords"] for item in batch]),
            "filename": [str(item["filename"]) for item in batch],
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def _batch_summary(name: str, batch: dict[str, torch.Tensor | list[str]]) -> None:
    print(f"{name}:")
    print(f"  image: {tuple(batch['image'].shape)} {batch['image'].dtype}")
    print(f"  mask: {tuple(batch['mask'].shape)} {batch['mask'].dtype}")
    print(f"  temporal_coords: {tuple(batch['temporal_coords'].shape)} {batch['temporal_coords'].dtype}")
    print(f"  location_coords: {tuple(batch['location_coords'].shape)} {batch['location_coords'].dtype}")
    print(f"  filenames: {batch['filename'][: min(3, len(batch['filename']))]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test MMEarth Dynamic World dataloaders.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fallback-val-fraction", type=float, default=0.1)
    parser.add_argument("--fallback-test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B2", "B3", "B4", "B8A", "B11", "B12"],
        help="Sentinel-2 bands to load.",
    )
    args = parser.parse_args()

    datamodule = MMEarthDynamicWorldSegmentationDataModule(
        data_root=args.data_root,
        sentinel2_bands=args.bands,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        normalize=True,
        fallback_val_fraction=args.fallback_val_fraction,
        fallback_test_fraction=args.fallback_test_fraction,
    )
    datamodule.setup(None)

    print(f"data_root: {args.data_root}")
    print(f"train_len: {len(datamodule.train_dataset)}")
    print(f"val_len: {len(datamodule.val_dataset)}")
    print(f"test_len: {len(datamodule.test_dataset)}")

    _batch_summary("train_batch", next(iter(datamodule.train_dataloader())))
    _batch_summary("val_batch", next(iter(datamodule.val_dataloader())))
    _batch_summary("test_batch", next(iter(datamodule.test_dataloader())))

    datamodule.teardown(None)
    print("dataloader_smoke_test_ok=True")


if __name__ == "__main__":
    # Example:
    # python mmearth_dynamic_world_datamodule.py --data-root "/data1/aminur/om/MMearth20k/hardcoded" --batch-size 2 --num-workers 0 --image-size 224
    main()
