"""
CloudSEN12 Stratified 14-Biome DataModule for FairRSFM
======================================================
Provides a unified PyTorch Dataset and LightningDataModule for CloudSEN12
dense cloud & cloud shadow semantic segmentation (4 classes: clear, thick cloud, thin cloud, cloud shadow)
with canonical RESOLVE 2017 14-Biome geographic group assignments.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import tacoreader

# Set PROJ environment variables for rasterio / pyproj compatibility
import pyproj.datadir
proj_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"] = proj_dir

DEFAULT_CLOUDSEN12_TORTILLA = "/data1/aminur/dataset/geobench2/cloudsen12/geobench_cloudsen12.tortilla"
DEFAULT_SPLITS_JSON = "/data1/aminur/dataset/geobench2/cloudsen12/cloudsen12_stratified_14biomes_splits.json"
DEFAULT_BIOMES_CSV = "/data1/aminur/om/prithvi/FairRSFM_repo/scripts/cloudsen12_samples_with_biomes.csv"

CLOUDSEN12_CLASSES = [
    "clear",          # 0
    "thick_cloud",    # 1
    "thin_cloud",     # 2
    "cloud_shadow",   # 3
]
NUM_CLASSES = len(CLOUDSEN12_CLASSES)

RESOLVE_BIOME_NAMES = {
    1: "Tropical & Subtropical Moist Broadleaf Forests",
    2: "Tropical & Subtropical Dry Broadleaf Forests",
    3: "Tropical & Subtropical Coniferous Forests",
    4: "Temperate Broadleaf & Mixed Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests/Taiga",
    7: "Tropical & Subtropical Grasslands, Savannas & Shrublands",
    8: "Temperate Grasslands, Savannas & Shrublands",
    9: "Flooded Grasslands & Savannas",
    10: "Montane Grasslands & Shrublands",
    11: "Tundra",
    12: "Mediterranean Forests, Woodlands & Scrub",
    13: "Deserts & Xeric Shrublands",
    14: "Mangroves"
}

# S2 12-band order in CloudSEN12
S2_12_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


class CloudSEN12SegmentationDataset(Dataset):
    """
    CloudSEN12 Dataset yielding normalized optical tensors, dense segmentation masks,
    and 14-biome geographic group labels.
    """
    def __init__(
        self,
        tortilla_path: str = DEFAULT_CLOUDSEN12_TORTILLA,
        splits_json: str = DEFAULT_SPLITS_JSON,
        biomes_csv: str = DEFAULT_BIOMES_CSV,
        split: str = "train",
        image_size: int = 224,
        selected_bands: Optional[Sequence[str]] = None,
        band_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.tortilla_path = tortilla_path
        self.split = split
        self.image_size = image_size

        if not os.path.exists(tortilla_path):
            raise FileNotFoundError(f"CloudSEN12 tortilla not found: {tortilla_path}")
        if not os.path.exists(splits_json):
            raise FileNotFoundError(f"Splits JSON not found: {splits_json}")
        if not os.path.exists(biomes_csv):
            raise FileNotFoundError(f"Biomes CSV not found: {biomes_csv}")

        # Load metadata and splits
        with open(splits_json, "r") as f:
            splits_data = json.load(f)

        if split not in splits_data:
            raise ValueError(f"Unknown split '{split}'. Available: {list(splits_data.keys())}")
        self.indices = splits_data[split]

        biomes_df = pd.read_csv(biomes_csv)
        self.biome_map = dict(zip(biomes_df["index"], biomes_df["BIOME_NUM"]))
        self.lon_map = dict(zip(biomes_df["index"], biomes_df["lon"]))
        self.lat_map = dict(zip(biomes_df["index"], biomes_df["lat"]))

        self.taco_df = tacoreader.load(tortilla_path)

        # Selected bands configuration
        if band_indices is not None:
            self.band_indices = list(band_indices)
            self.selected_bands = [S2_12_BANDS[i] for i in self.band_indices if i < len(S2_12_BANDS)]
        elif selected_bands is not None:
            self.selected_bands = list(selected_bands)
            self.band_indices = [S2_12_BANDS.index(b) for b in self.selected_bands if b in S2_12_BANDS]
        else:
            self.selected_bands = list(S2_12_BANDS)
            self.band_indices = list(range(len(S2_12_BANDS)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.indices[idx]
        sample_row = self.taco_df.read(global_idx)

        img_path = sample_row.read(0)
        mask_path = sample_row.read(1)

        with rasterio.open(img_path) as src:
            img_arr = src.read()  # Shape: (14, 512, 512) or (12, 512, 512)
        with rasterio.open(mask_path) as src:
            mask_arr = src.read(1)  # Shape: (512, 512)

        # Take first 12 optical bands if file has 14
        if img_arr.shape[0] > 12:
            img_arr = img_arr[:12]

        # Select requested bands
        if len(self.band_indices) != img_arr.shape[0]:
            img_arr = img_arr[self.band_indices]

        img_t = torch.from_numpy(img_arr).float()
        mask_t = torch.from_numpy(mask_arr).long()

        # Resize to target image_size
        if self.image_size is not None and (img_t.shape[1] != self.image_size or img_t.shape[2] != self.image_size):
            img_t = F.interpolate(
                img_t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="nearest"
            ).squeeze(0).squeeze(0).long()

        # Retrieve biome and coordinates
        biome_num = int(self.biome_map.get(global_idx, 1))
        # Ensure 0-indexed group id: 0 to 13
        group_id = max(0, min(13, biome_num - 1))

        return {
            "image": img_t,
            "mask": mask_t,
            "group": torch.tensor(group_id, dtype=torch.long),
            "biome_num": torch.tensor(biome_num, dtype=torch.long),
            "lon": torch.tensor(float(self.lon_map.get(global_idx, 0.0)), dtype=torch.float32),
            "lat": torch.tensor(float(self.lat_map.get(global_idx, 0.0)), dtype=torch.float32),
            "sample_id": global_idx,
        }


class CloudSEN12SegmentationDataModule(LightningDataModule):
    """
    Lightning DataModule providing train, val, test loaders for CloudSEN12.
    """
    def __init__(
        self,
        tortilla_path: str = DEFAULT_CLOUDSEN12_TORTILLA,
        splits_json: str = DEFAULT_SPLITS_JSON,
        biomes_csv: str = DEFAULT_BIOMES_CSV,
        image_size: int = 224,
        batch_size: int = 16,
        num_workers: int = 4,
        selected_bands: Optional[Sequence[str]] = None,
        band_indices: Optional[Sequence[int]] = None,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.tortilla_path = tortilla_path
        self.splits_json = splits_json
        self.biomes_csv = biomes_csv
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.selected_bands = selected_bands
        self.band_indices = band_indices
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = CloudSEN12SegmentationDataset(
                tortilla_path=self.tortilla_path,
                splits_json=self.splits_json,
                biomes_csv=self.biomes_csv,
                split="train",
                image_size=self.image_size,
                selected_bands=self.selected_bands,
                band_indices=self.band_indices,
            )
            self.val_dataset = CloudSEN12SegmentationDataset(
                tortilla_path=self.tortilla_path,
                splits_json=self.splits_json,
                biomes_csv=self.biomes_csv,
                split="validation",
                image_size=self.image_size,
                selected_bands=self.selected_bands,
                band_indices=self.band_indices,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = CloudSEN12SegmentationDataset(
                tortilla_path=self.tortilla_path,
                splits_json=self.splits_json,
                biomes_csv=self.biomes_csv,
                split="test",
                image_size=self.image_size,
                selected_bands=self.selected_bands,
                band_indices=self.band_indices,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
