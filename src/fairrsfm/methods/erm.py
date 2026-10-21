"""
Empirical Risk Minimization (ERM)
=================================
Standard unweighted linear probing baseline. Biome metadata is not used
during training and is evaluated post-hoc across subgroups.

Reference:
    FairRSFM: Biome-Aware Benchmark & Debiasing Framework (docs/methods.md)
"""

from typing import Optional
import torch
import torch.nn as nn


class ERMLoss(nn.Module):
    """Standard unweighted empirical risk loss wrapper."""
    def __init__(self, task: str = "classification", class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.task = task
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif task == "multilabel":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif task == "segmentation":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)
