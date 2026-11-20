"""The common per-observation task losses for every mitigation method."""

import torch
import torch.nn.functional as F


def per_sample_task_loss(logits, target, task, ignore_index=-1):
    if task == "bigearthnet":
        return F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none").mean(-1)
    if task == "eurosat":
        return F.cross_entropy(logits, target.long(), reduction="none")
    if task not in {"sa_crop", "mmearth"}:
        raise ValueError(f"Unknown task {task}")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    valid = target != ignore_index
    counts = valid.flatten(1).sum(1)
    losses = F.cross_entropy(logits, target.long(), ignore_index=ignore_index, reduction="none")
    # Empty patches have no observation, rather than an artificial zero loss.
    means = losses.flatten(1).sum(1) / counts.clamp_min(1)
    return torch.where(counts > 0, means, torch.full_like(means, float("nan")))
