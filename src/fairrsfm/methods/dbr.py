"""
Dynamic Biome Reweighting with Validation Feedback (DBR)
========================================================
A loss-level mitigation method that begins with inverse-frequency weights
and dynamically updates them using validation performance across biomes.

Reference:
    FairRSFM: Biome-Aware Benchmark & Debiasing Framework (docs/methods.md)
"""

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class DynamicBiomeReweighter:
    """
    Tracks and updates per-biome loss weights based on validation feedback.
    
    Formula:
        w_g^(0) = (1 / p_g) / ( (1 / |G|) * sum_{g'} 1 / p_g' )
        w_g^(t+1) = w_g^(t) * (1 + alpha * (M_bar^(t) - M_g^(t)) / (M_bar^(t) + eps))
        Followed by unit-mean normalization and EMA smoothing.
    """
    def __init__(
        self,
        train_biomes: List[str],
        alpha: float = 0.5,
        ema: float = 0.8,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        self.alpha = alpha
        self.ema = ema
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Compute empirical frequencies
        counts = defaultdict(int)
        for b in train_biomes:
            if b != "Unknown":
                counts[b] += 1
        self.biomes = sorted(counts.keys())
        total = sum(counts.values())

        if len(self.biomes) == 0:
            self.w_current = {}
            return

        # Inverse-frequency initial weights normalized to mean 1.0
        inv_freq = {b: total / (len(self.biomes) * counts[b]) for b in self.biomes}
        mean_if = np.mean(list(inv_freq.values()))
        self.w_current: Dict[str, float] = {b: float(inv_freq[b] / mean_if) for b in self.biomes}

    def get_weights_tensor(
        self,
        sample_ids: List[str],
        biome_map: Dict[str, str],
        device: torch.device
    ) -> torch.Tensor:
        """Get per-sample weight tensor for a mini-batch."""
        w_list = [self.w_current.get(biome_map.get(str(sid), "Unknown"), 1.0) for sid in sample_ids]
        return torch.tensor(w_list, dtype=torch.float32, device=device)

    def update_from_val(self, val_group_metrics: Dict[str, float]):
        """
        Update biome weights using validation metric feedback (e.g. F1 or mIoU).
        Underperforming groups receive higher weights, overperforming receive lower.
        """
        valid = {
            b: val for b, val in val_group_metrics.items()
            if b in self.w_current and val is not None and not np.isnan(val)
        }
        if not valid:
            return

        mean_val = np.mean(list(valid.values()))
        w_raw = {}
        for b in self.biomes:
            if b in valid:
                rel_diff = (mean_val - valid[b]) / (mean_val + 1e-6)
                w_target = self.w_current[b] * (1.0 + self.alpha * rel_diff)
                w_raw[b] = max(self.min_weight, min(self.max_weight, w_target))
            else:
                w_raw[b] = self.w_current[b]

        mean_w = np.mean(list(w_raw.values()))
        w_norm = {b: w_raw[b] / mean_w for b in self.biomes}

        for b in self.biomes:
            w_old = self.w_current[b]
            w_new = self.ema * w_old + (1.0 - self.ema) * w_norm[b]
            self.w_current[b] = float(w_new)
