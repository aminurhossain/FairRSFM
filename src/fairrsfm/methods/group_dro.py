"""
Group Distributionally Robust Optimization (GroupDRO)
=====================================================
Minimizes worst-case empirical risk across terrestrial biomes via online
exponentiated gradient ascent over group weights.

Reference:
    Sagawa et al., Distributionally Robust Neural Networks (ICLR 2020)
    FairRSFM: Biome-Aware Benchmark & Debiasing Framework (docs/methods.md)
"""

from typing import List, Optional
import torch
import torch.nn as nn


class GroupDROLossTracker(nn.Module):
    """
    Maintains and updates group weights q_g online using exponentiated gradient ascent.
    
    Formula:
        loss = sum_g q_g * loss_g
        q_g <- q_g * exp(group_lr * loss_g)
        q <- q / sum(q)
    """
    def __init__(self, group_names: List[str], group_lr: float = 0.01):
        super().__init__()
        self.group_names = sorted(group_names)
        self.group_lr = group_lr
        self.num_groups = len(self.group_names)
        self.group_to_idx = {g: i for i, g in enumerate(self.group_names)}

        # Initialize uniform group weights
        initial_q = torch.ones(self.num_groups, dtype=torch.float32) / max(1, self.num_groups)
        self.register_buffer("q", initial_q)

    def forward(
        self,
        per_sample_loss: torch.Tensor,
        sample_biomes: List[str]
    ) -> torch.Tensor:
        """
        Compute weighted GroupDRO loss for the current batch and update q online.
        
        Args:
            per_sample_loss: Unreduced loss tensor of shape (B,)
            sample_biomes: List of biome group names for each sample in the batch
        """
        device = per_sample_loss.device
        present_indices = []
        present_losses = []

        for g, idx in self.group_to_idx.items():
            mask = torch.tensor([b == g for b in sample_biomes], dtype=torch.bool, device=device)
            if mask.sum() > 0:
                l_g = per_sample_loss[mask].mean()
                present_indices.append(idx)
                present_losses.append(l_g)

        if not present_indices:
            return per_sample_loss.mean()

        # Weighted loss combination: sum_g q_g * loss_g (detach q to avoid in-place graph conflict)
        q_weights = self.q.clone().detach()
        weighted_loss = sum(q_weights[idx] * l_g for idx, l_g in zip(present_indices, present_losses))

        # Online exponentiated gradient ascent update: q_g <- q_g * exp(eta_q * loss_g)
        with torch.no_grad():
            for idx, l_g in zip(present_indices, present_losses):
                self.q[idx] = self.q[idx] * torch.exp(self.group_lr * l_g.detach())
            self.q.copy_(self.q / self.q.sum())

        return weighted_loss
