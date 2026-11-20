"""Epoch-level DBR driven by EMA-smoothed, unweighted group training losses."""

from collections import Counter
from collections.abc import Mapping

import torch
from torch import nn


class DynamicBiomeReweighter(nn.Module):
    """Persist weights, EMA losses, and partial-epoch accumulators in checkpoints."""

    def __init__(self, train_biomes, alpha=0.5, beta=0.9, epsilon=1e-6):
        super().__init__()
        if alpha < 0 or not 0 <= beta < 1 or epsilon <= 0:
            raise ValueError("DBR requires alpha >= 0, 0 <= beta < 1, epsilon > 0")
        counts = dict(train_biomes) if isinstance(train_biomes, Mapping) else Counter(train_biomes)
        counts = {g: int(n) for g, n in counts.items() if g and g != "Unknown" and n > 0}
        self.biomes = sorted(counts)
        self.group_to_idx = {g: i for i, g in enumerate(self.biomes)}
        self.alpha, self.beta, self.epsilon = float(alpha), float(beta), float(epsilon)
        frequency = torch.tensor([counts[g] for g in self.biomes], dtype=torch.float64)
        initial = frequency.reciprocal()
        if len(initial):
            initial /= initial.mean()
        self.register_buffer("weights", initial)
        self.register_buffer("ema_loss", torch.full_like(initial, float("nan")))
        self.register_buffer("loss_sum", torch.zeros_like(initial))
        self.register_buffer("observation_count", torch.zeros_like(initial))
        self.register_buffer("last_update_epoch", torch.tensor(-1, dtype=torch.long))

    def get_extra_state(self):
        return {"protocol": "training_loss_ema_v1", "groups": self.biomes,
                "alpha": self.alpha, "beta": self.beta, "epsilon": self.epsilon}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("DBR checkpoint groups or hyperparameters differ from this run")

    def weights_for_groups(self, groups, device=None, dtype=torch.float32):
        target = device if device is not None else self.weights.device
        indices = torch.tensor([self.group_to_idx.get(g, -1) for g in groups],
                               device=self.weights.device, dtype=torch.long)
        values = self.weights.new_ones(len(groups))
        known = indices >= 0
        values[known] = self.weights[indices[known]]
        return values.to(device=target, dtype=dtype)

    def get_weights_tensor(self, sample_ids, biome_map, device):
        return self.weights_for_groups([biome_map.get(str(s), "Unknown") for s in sample_ids], device)

    @torch.no_grad()
    def observe(self, per_sample_loss, groups):
        """Accumulate raw task losses; never pass losses already weighted by DBR."""
        values = per_sample_loss.detach().reshape(-1).to(self.loss_sum)
        if len(values) != len(groups):
            raise ValueError("One group label is required per sample loss")
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("DBR observations must be finite nonnegative losses")
        for group in set(groups):
            idx = self.group_to_idx.get(group)
            if idx is None:
                continue
            mask = torch.tensor([g == group for g in groups], device=values.device)
            self.loss_sum[idx] += values[mask].sum()
            self.observation_count[idx] += mask.sum()

    @torch.no_grad()
    def finish_epoch(self, epoch):
        """Update once per epoch; absent groups retain their previous EMA loss."""
        if int(epoch) <= int(self.last_update_epoch):
            raise ValueError("DBR epoch was already updated or is out of order")
        sums, counts = self.loss_sum.clone(), self.observation_count.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(counts)
        active = counts > 0
        first = active & torch.isnan(self.ema_loss)
        previous = active & ~first
        mean = sums / counts.clamp_min(1)
        self.ema_loss[first] = mean[first]
        self.ema_loss[previous] = (self.beta * self.ema_loss[previous]
                                   + (1 - self.beta) * mean[previous])
        initialized = ~torch.isnan(self.ema_loss)
        if active.any():
            raw = self.weights.clone()
            raw[initialized] = (self.ema_loss[initialized] + self.epsilon).pow(self.alpha)
            self.weights.copy_(raw / raw.mean())
        self.loss_sum.zero_()
        self.observation_count.zero_()
        self.last_update_epoch.fill_(int(epoch))
