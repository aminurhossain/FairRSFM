"""
GEO-Bench Dataset Wrappers with Biome Group Annotations
======================================================
Wraps GEO-Bench classification and segmentation datasets, linking sample IDs
to RESOLVE macro-biome groups and filtering unmapped instances.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import geobench
    import geobench.task
    GEOBENCH_AVAILABLE = True
except ImportError:
    GEOBENCH_AVAILABLE = False


# Default band indices for Prithvi / Sentinel-2 (B2, B3, B4, B8A, B11, B12)
PRITHVI_BAND_INDICES = [1, 2, 3, 8, 11, 12]


class GeoBenchDataset(Dataset):
    """
    Standardized dataset wrapper for GEO-Bench tasks with biome group labels.
    """
    def __init__(
        self,
        task_name: str,
        split: str = "train",
        biome_map: Optional[Dict[str, str]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        band_indices: Optional[List[int]] = None,
        img_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        augment: bool = False,
    ):
        super().__init__()
        if not GEOBENCH_AVAILABLE:
            raise ImportError(
                "geobench is required. Install it via `pip install geobench`."
            )

        self.task_name = task_name
        self.split = split
        self.biome_map = biome_map or {}
        self.band_indices = band_indices or PRITHVI_BAND_INDICES
        self.img_size = img_size
        self.augment = augment

        # Load GEO-Bench task
        tasks = geobench.task.load_task_converters(data_dir=data_dir)
        matching = [t for t in tasks if t.dataset_name == task_name]
        if not matching:
            raise ValueError(f"Task '{task_name}' not found in GEO-Bench directory: {data_dir}")
        self.task = matching[0]
        self.dataset = self.task.get_dataset(split=split)
        self.band_names = self.dataset[0].band_names

        # Normalization constants
        default_mean = [1353.7, 1117.2, 1041.8, 2265.4, 2257.4, 1595.5]
        default_std = [65.3, 154.0, 187.9, 400.0, 597.0, 584.0]
        self.mean = torch.tensor(mean or default_mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std or default_std, dtype=torch.float32)[:, None, None]

        # Filter valid indices with known biome annotations
        self.valid_indices = []
        try:
            split_ids = self.dataset.active_partition.partition_dict[split]
            id_to_index = {str(self.dataset[i].sample_name): i for i in range(len(self.dataset))}
            for sid in split_ids:
                if self.biome_map.get(str(sid), "Unknown") != "Unknown":
                    if str(sid) in id_to_index:
                        self.valid_indices.append(id_to_index[str(sid)])
        except Exception:
            for i in range(len(self.dataset)):
                sid = str(self.dataset[i].sample_name)
                if self.biome_map.get(sid, "Unknown") != "Unknown":
                    self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        real_idx = self.valid_indices[idx]
        sample = self.dataset[real_idx]

        x_data = sample.pack_to_3d(band_names=self.band_names)[0]
        if self.band_indices:
            x_data = x_data[:, :, self.band_indices]
        x = torch.tensor(x_data, dtype=torch.float32).permute(2, 0, 1)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Spatial augmentations for training
        if self.augment:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])

        x = (x - self.mean) / self.std

        # Label extraction
        if hasattr(sample, "label"):
            y = torch.tensor(int(sample.label), dtype=torch.long)
        elif hasattr(sample, "labels"):
            y = torch.tensor(sample.labels, dtype=torch.float32)
        else:
            y = torch.tensor(0, dtype=torch.long)

        sid = str(sample.sample_name)
        return x, y, sid
