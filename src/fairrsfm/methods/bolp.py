"""
Biome-Orthogonal Linear Probing (BOLP)
======================================
Projects frozen representations into a subspace orthogonal to biome-discriminative
directions, isolating inter-group shortcuts while preserving task signal.

Reference:
    FairRSFM: Biome-Aware Benchmark & Debiasing Framework (docs/methods.md)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BOLPProjector(nn.Module):
    """
    Applies residual centering and orthogonal subspace projection to features.
    
    Formula:
        z_tilde = z - mu_g + mu_0   (if residualization enabled)
        z_proj = z_tilde @ P_k^T
    """
    def __init__(
        self,
        biome_centroids: Dict[str, torch.Tensor],
        global_centroid: torch.Tensor,
        Pk: torch.Tensor,
        k: int,
        skip_residualization: bool = False,
    ):
        super().__init__()
        self.skip_residualization = skip_residualization
        self.k = k
        self.register_buffer("mu0", global_centroid)
        self.register_buffer("Pk", Pk)

        biomes_sorted = sorted(biome_centroids.keys())
        self.biome_index = {b: i for i, b in enumerate(biomes_sorted)}
        mu_stack = torch.stack([biome_centroids[b] for b in biomes_sorted], dim=0)
        self.register_buffer("mu_table", mu_stack)

    def get_extra_state(self):
        return {"biome_index": self.biome_index, "k": self.k,
                "skip_residualization": self.skip_residualization,
                "transform": "tokenwise_v1"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("BOLP checkpoint groups, rank, or residualization differ from this run")

    def forward(self, z: torch.Tensor, biome_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        Project feature tensor z of shape (B, d).
        """
        if biome_ids is None:
            biome_ids = getattr(self, "_sample_biomes", None)
        if z.ndim not in (2, 3, 4):
            raise ValueError("BOLP expects image vectors, tokens, or spatial feature maps")
        z_tilde = z.movedim(1, -1).clone() if z.ndim == 4 else z.clone()
        if not self.skip_residualization and biome_ids is not None:
            for i, biome in enumerate(biome_ids):
                idx = self.biome_index.get(biome, None)
                if idx is not None:
                    z_tilde[i] = z_tilde[i] - self.mu_table[idx] + self.mu0
        result = z_tilde @ self.Pk.T
        return result.movedim(-1, 1) if z.ndim == 4 else result

    def variance_retained(self, z: torch.Tensor) -> float:
        """Compute the fraction of feature variance retained after projection."""
        with torch.no_grad():
            var_before = z.var(dim=0).sum().item()
            z_proj = z @ self.Pk.T
            var_after = z_proj.var(dim=0).sum().item()
        return var_after / var_before if var_before > 0 else float("nan")


def compute_bolp_projection(
    embeddings: torch.Tensor,
    sample_biomes: List[str],
    skip_residualization: bool = False,
) -> BOLPProjector:
    """
    Compute SVD-based projection matrix P_k from pre-extracted training embeddings.
    
    Args:
        embeddings: Tensor of shape (N, d)
        sample_biomes: List of biome group names for each of the N samples
        skip_residualization: If True, only apply Pk projection without centroid shifting
    """
    d = embeddings.shape[1]
    biome_to_indices = defaultdict(list)
    for i, b in enumerate(sample_biomes):
        if b != "Unknown":
            biome_to_indices[b].append(i)

    known_biomes = {b: idxs for b, idxs in biome_to_indices.items() if len(idxs) > 0}
    if not known_biomes:
        raise ValueError("BOLP requires known training biomes")

    biome_centroids = {b: embeddings[idxs].mean(dim=0) for b, idxs in known_biomes.items()}
    mu0 = torch.stack(list(biome_centroids.values()), dim=0).mean(dim=0)
    B_count = len(biome_centroids)

    # Rank k = |G| - 1
    k = min(B_count - 1, d)

    # Inter-group mean-difference matrix M in R^{d x |G|}
    M = torch.stack([biome_centroids[b] - mu0 for b in sorted(biome_centroids.keys())], dim=1)

    # Economy SVD
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Uk = U[:, :k]  # (d, k)

    # Orthogonal projection onto the null space of inter-group differences
    Id = torch.eye(d, dtype=M.dtype, device=M.device)
    Pk = Id - Uk @ Uk.T

    projector = BOLPProjector(
        biome_centroids=biome_centroids,
        global_centroid=mu0,
        Pk=Pk,
        k=k,
        skip_residualization=skip_residualization,
    )
    return projector


class LayerwiseBOLP(nn.Module):
    def __init__(self, projectors):
        super().__init__()
        self.layers = nn.ModuleList(projectors)

    def forward(self, features, biome_ids=None):
        if isinstance(features, (tuple, list)):
            if len(features) != len(self.layers):
                raise ValueError("BOLP feature-layer count changed since fitting")
            values = [p(f, biome_ids) for p, f in zip(self.layers, features)]
            return tuple(values) if isinstance(features, tuple) else values
        if len(self.layers) != 1:
            raise ValueError("Expected multiple feature layers for this BOLP fit")
        return self.layers[0](features, biome_ids)

    @property
    def biome_index(self):
        return self.layers[-1].biome_index

    @property
    def k(self):
        return self.layers[-1].k

    @property
    def skip_residualization(self):
        return self.layers[-1].skip_residualization


@torch.no_grad()
def fit_bolp_layers(extractor, dataloader, lookup, device, all_layers=False,
                   skip_residualization=False, rank=None):
    """Fit image-weighted centroids per layer using a streaming training scan."""
    sums, counts = [], []
    for batch in dataloader:
        images, ids = (batch["image"], batch["filename"]) if isinstance(batch, dict) else (batch[0], batch[2])
        features = extractor(images.to(device))
        if isinstance(features, (tuple, list)):
            features = list(features) if all_layers else [features[-1]]
        else:
            features = [features]
        if not sums:
            sums = [defaultdict(lambda: None) for _ in features]
            counts = [defaultdict(int) for _ in features]
        if len(features) != len(sums):
            raise ValueError("Encoder feature structure changed during BOLP fitting")
        for layer, value in enumerate(features):
            if value.ndim == 4:
                value = value.mean((2, 3))
            elif value.ndim == 3:
                value = value.mean(1)
            elif value.ndim != 2:
                raise ValueError(f"Unsupported BOLP features: {value.shape}")
            value = value.detach().double().cpu()
            for i, sid in enumerate(ids):
                group = lookup(str(sid)) or "Unknown"
                if group == "Unknown":
                    continue
                sums[layer][group] = value[i].clone() if sums[layer][group] is None else sums[layer][group] + value[i]
                counts[layer][group] += 1
    if not sums:
        raise ValueError("No training batches for BOLP fitting")
    projectors = []
    for layer in range(len(sums)):
        centroids = {g: total / counts[layer][g] for g, total in sums[layer].items()}
        if not centroids:
            raise ValueError("No known training groups for BOLP fitting")
        mu0 = torch.stack(list(centroids.values())).mean(0)
        matrix = torch.stack([centroids[g] - mu0 for g in sorted(centroids)], 1)
        u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        k = min(len(centroids) - 1, matrix.shape[0])
        if rank is not None:
            if not 0 <= rank <= k:
                raise ValueError(f"Requested BOLP rank {rank} exceeds the bound {k}")
            k = rank
        basis = u[:, :k]
        projection = torch.eye(matrix.shape[0], dtype=matrix.dtype) - basis @ basis.T
        projector = BOLPProjector({g: v.float() for g, v in centroids.items()},
            mu0.float(), projection.float(), k, skip_residualization)
        projectors.append(projector.to(device))
        print(f"[BOLP] layer={layer} groups={len(centroids)} rank={k} observations={dict(counts[layer])}")
    return LayerwiseBOLP(projectors)


class PrithviFinalLayerBOLP(nn.Module):
    """Prithvi adapter with checkpointed final-layer centroids and projector."""

    def __init__(self, k=None, device="cpu", tile_lookup=None):
        super().__init__()
        self.k, self.device, self.tile_lookup = k, torch.device(device), tile_lookup
        self.projector = None

    @staticmethod
    def _get_encoder(model):
        inner = getattr(model, "model", model)
        for name in ("encoder", "backbone", "encoder_backbone"):
            if hasattr(inner, name):
                return getattr(inner, name)
        raise AttributeError("Prithvi adapter cannot locate the frozen encoder")

    def fit(self, model, dataloader, tile_info):
        encoder = self._get_encoder(model)
        encoder.eval()
        fitted = fit_bolp_layers(encoder, dataloader,
            lambda sid: self.tile_lookup(sid, tile_info), self.device, rank=self.k)
        self.projector = fitted.layers[0]

    def project_tokens(self, tokens, tile_ids, tile_info):
        if self.projector is None:
            raise RuntimeError("Fit BOLP on training features before applying it")
        if self.projector.mu0.device != tokens.device:
            self.projector.to(tokens.device)
        return self.projector(tokens, [self.tile_lookup(str(sid), tile_info) for sid in tile_ids])

    def forward_with_bolp(self, model, images, tile_ids=None, tile_info=None):
        tile_ids = tile_ids if tile_ids is not None else [None] * len(images)
        tile_info = tile_info or {}
        dev = next(model.parameters()).device
        if images.device != dev:
            images = images.to(dev, non_blocking=True)
        def hook(module, args, output):
            if isinstance(output, (list, tuple)):
                values = list(output)
                values[-1] = self.project_tokens(values[-1], tile_ids, tile_info)
                return tuple(values) if isinstance(output, tuple) else values
            return self.project_tokens(output, tile_ids, tile_info)
        handle = self._get_encoder(model).register_forward_hook(hook)
        try:
            output = getattr(model, "model", model)(images)
            return output.output if hasattr(output, "output") else output
        finally:
            handle.remove()
