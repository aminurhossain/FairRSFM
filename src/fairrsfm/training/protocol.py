"""Common training objectives, known-biome subsets, and Lightning steps."""

from collections import Counter
import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from fairrsfm.methods.dbr import DynamicBiomeReweighter
from fairrsfm.methods.group_dro import GroupDROLossTracker
from fairrsfm.methods.task_loss import per_sample_task_loss
from fairrsfm.methods.bolp import BOLPProjector


def dataset_sample_ids(dataset):
    """Read split metadata rather than images, respecting wrappers and subsets."""
    if isinstance(dataset, Subset):
        ids = dataset_sample_ids(dataset.dataset)
        return [ids[int(i)] for i in dataset.indices]
    if hasattr(dataset, "h5_path") and hasattr(dataset, "indices"):
        import h5py
        with h5py.File(dataset.h5_path, "r") as handle:
            names = handle["metadata"].fields("tile_id")[:]
        return [names[int(i)].decode("utf-8") for i in dataset.indices]
    if hasattr(dataset, "active_partition"):
        names = dataset.active_partition.partition_dict.get(getattr(dataset, "split", "train"))
        if names is not None and len(names) == len(dataset):
            return list(map(str, names))
    if hasattr(dataset, "dataset"):
        names = dataset_sample_ids(dataset.dataset)
        if hasattr(dataset, "valid_indices"):
            names = [names[int(i)] for i in dataset.valid_indices]
        if len(names) == len(dataset):
            return names
    # Metadata-free third-party datasets still work, with an explicit scan.
    names = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict):
            name = item["filename"]
        elif hasattr(item, "sample_name"):
            name = item.sample_name
        else:
            name = item[-1]
        names.append(str(name))
    return names


class KnownBiomeSubset(Subset):
    def close(self):
        if hasattr(self.dataset, "close"):
            self.dataset.close()


def known_training_dataset(dataset, lookup):
    ids = dataset_sample_ids(dataset)
    if len(ids) != len(dataset) or len(set(ids)) != len(ids):
        raise ValueError("Training sample IDs must be unique and match the dataset")
    groups = {sid: lookup(sid) or "Unknown" for sid in ids}
    indices = [i for i, sid in enumerate(ids) if groups[sid] != "Unknown"]
    groups = {ids[i]: groups[ids[i]] for i in indices}
    if not indices:
        raise ValueError("No training samples have known biome labels")
    print(f"[Protocol] known-biome training subset: {len(indices)}/{len(ids)}; {dict(Counter(groups.values()))}")
    return KnownBiomeSubset(dataset, indices), groups


def lookup_biome(mapping, sid):
    return mapping.get(str(sid)) or mapping.get(Path(str(sid)).stem) or "Unknown"


def training_loader(dataset, method, **kwargs):
    """Official GroupDRO example setting: sample groups uniformly in expectation."""
    groups = kwargs.pop("groups")
    if method == "groupdro":
        ids = dataset_sample_ids(dataset)
        counts = Counter(groups.values())
        weights = torch.tensor([1 / counts[groups[sid]] for sid in ids], dtype=torch.double)
        kwargs.pop("shuffle", None)
        kwargs["sampler"] = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(dataset, **kwargs)


def balanced_group_loader(loader, groups):
    return training_loader(loader.dataset, "groupdro", groups=groups,
        batch_size=loader.batch_size, num_workers=loader.num_workers,
        collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
        drop_last=loader.drop_last, worker_init_fn=loader.worker_init_fn,
        persistent_workers=loader.persistent_workers)


def forward_with_sample_ids(module, images, sample_ids):
    """Keep sample grouping available through nested backbone adapters."""
    lookup = getattr(module, "_protocol_lookup", None)
    groups = [lookup(str(sid)) or "Unknown" for sid in sample_ids] if lookup else None
    projectors = [m for m in module.modules() if isinstance(m, BOLPProjector)]
    for projector in projectors:
        projector._sample_biomes = groups
    try:
        if "sample_ids" in inspect.signature(module.forward).parameters:
            return module(images, sample_ids=sample_ids)
        return module(images)
    finally:
        for projector in projectors:
            projector._sample_biomes = None


class MethodObjective(nn.Module):
    def __init__(self, method, task, groups, ignore_index=-1, alpha=0.5, beta=0.9, eta=0.01):
        super().__init__()
        self.method, self.task, self.ignore_index = method, task, ignore_index
        self.groups = dict(groups)
        self.dbr = DynamicBiomeReweighter(list(groups.values()), alpha=alpha, beta=beta) if method == "dbr" else None
        self.dro = GroupDROLossTracker(list(set(groups.values())), group_lr=eta) if method == "groupdro" else None
        self.config = {"method": method, "task": task, "ignore_index": ignore_index,
                       "alpha": alpha if method == "dbr" else None,
                       "beta": beta if method == "dbr" else None,
                       "eta": eta if method == "groupdro" else None,
                       "known_biome_training_only": True,
                       "group_counts": dict(Counter(groups.values())),
                       "training_groups_sha256": hashlib.sha256("\n".join(f"{k}\0{groups[k]}" for k in sorted(groups)).encode()).hexdigest(),
                       "training_ids_sha256": hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()}
        if method == "dbr":
            self.config.update(epsilon=1e-6, update_every_epochs=1,
                               feedback="unweighted_group_training_loss",
                               initialization="inverse_training_group_frequency", clipping=None)
        if method == "groupdro":
            self.config.update(sampling="inverse_group_frequency_with_replacement",
                               update_probabilities="before_weighted_objective",
                               initialization="uniform", adjustment=0, normalize_loss=False)
        if method == "bolp":
            self.config.update(bolp_transform="tokenwise_v1", rank="num_groups_minus_one",
                               residual_centering=True)

    def get_extra_state(self):
        return self.config

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError("Method protocol or known-biome training subset differs from the checkpoint")

    def forward(self, logits, target, sample_ids):
        losses = per_sample_task_loss(logits, target, self.task, self.ignore_index)
        valid = torch.isfinite(losses)
        if not valid.any():
            return logits.sum() * 0
        groups = [self.groups[str(sid)] for i, sid in enumerate(sample_ids) if bool(valid[i])]
        losses = losses[valid]
        if self.dbr is not None:
            self.dbr.observe(losses, groups)
            return (losses * self.dbr.weights_for_groups(groups, losses.device, losses.dtype)).mean()
        if self.dro is not None:
            return self.dro(losses, groups)
        return losses.mean()


def configure_training_protocol(module, method, task, groups, args, ignore_index=-1, lookup=None):
    module._protocol_lookup = lookup
    module.method_protocol = MethodObjective(
        method, task, groups, ignore_index,
        alpha=getattr(args, "alpha", getattr(args, "dbr_alpha", 0.5)),
        beta=getattr(args, "beta", getattr(args, "ema_decay", getattr(args, "dbr_ema", 0.9))),
        eta=getattr(args, "eta_q", getattr(args, "group_lr", 0.01)))
    module.hparams["method_protocol"] = module.method_protocol.config
    module.register_load_state_dict_pre_hook(_require_protocol_state)
    for name, child in module.named_modules():
        if name.split(".")[-1] in {"encoder", "backbone"}:
            child.requires_grad_(False)
            child.eval()
    update_projection_protocol(module, args)


def _require_protocol_state(module, state_dict, prefix, *unused):
    """Even non-strict backbone loads must restore the head and method state."""
    required = {prefix + "method_protocol." + key
                for key in module.method_protocol.state_dict()}
    required.update(prefix + name for name, value in module.named_parameters()
                    if value.requires_grad)
    for name, child in module.named_modules():
        if isinstance(child, BOLPProjector):
            required.update(prefix + name + "." + key for key in child.state_dict())
    missing = sorted(required.difference(state_dict))
    if missing:
        raise ValueError("Checkpoint lacks required head or shared method state; "
                         "historical runs cannot resume under the new protocol. "
                         f"Missing keys: {missing}")


def update_projection_protocol(module, args):
    if module.method_protocol.method == "bolp":
        projectors = [p for p in module.modules() if isinstance(p, BOLPProjector)]
        module.method_protocol.config["projectors"] = [p.get_extra_state() for p in projectors]
        module.method_protocol.config["feature_layer"] = "final_encoder_layer"
    args.method_protocol = module.method_protocol.config


class MethodTrainingMixin:
    """Architecture adapters supply forward; every method shares these steps."""

    def on_train_epoch_start(self):
        dbr = self.method_protocol.dbr
        if dbr is not None and self.current_epoch > 0:
            if int(dbr.last_update_epoch) < self.current_epoch - 1:
                dbr.finish_epoch(self.current_epoch - 1)
        hook = getattr(super(), "on_train_epoch_start", None)
        if hook is not None:
            hook()

    def on_train_epoch_end(self):
        dbr = self.method_protocol.dbr
        if dbr is not None and int(dbr.last_update_epoch) < self.current_epoch:
            dbr.finish_epoch(self.current_epoch)
        hook = getattr(super(), "on_train_epoch_end", None)
        if hook is not None:
            hook()

    def _protocol_step(self, batch, split):
        if isinstance(batch, dict):
            images, target, ids = batch["image"], batch["mask"], batch["filename"]
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]
        else:
            images, target, ids = batch
        for name, child in self.named_modules():
            if name.split(".")[-1] in {"encoder", "backbone"}:
                child.eval()
        if hasattr(self, "_bolp_forward"):
            logits = self._bolp_forward(batch)
        elif any(c.__name__ == "SemanticSegmentationTask" for c in type(self).__mro__):
            logits = self.model(images).output
        else:
            logits = forward_with_sample_ids(self, images, ids)
        if split == "train":
            loss = self.method_protocol(logits, target, ids)
        else:
            values = per_sample_task_loss(logits, target, self.method_protocol.task,
                                          self.method_protocol.ignore_index)
            loss = values.nanmean() if torch.isfinite(values).any() else logits.sum() * 0
        metrics = getattr(self, f"{split}_metrics", None)
        if metrics is not None:
            if split == "test" and isinstance(metrics, (list, nn.ModuleList)):
                metrics = metrics[0]
            metrics.update(logits.argmax(1), target)
        self.log(f"{split}/loss", loss, on_step=split == "train", on_epoch=True,
                 prog_bar=True, batch_size=images.shape[0])
        return loss if split == "train" else {"logits": logits, "y": target,
                                              "masks": target, "sample_ids": ids}

    def training_step(self, batch, batch_idx):
        return self._protocol_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._protocol_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._protocol_step(batch, "test")
