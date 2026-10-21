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

    def forward(self, z: torch.Tensor, biome_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        Project feature tensor z of shape (B, d).
        """
        z_tilde = z.clone()
        if not self.skip_residualization and biome_ids is not None:
            for i, biome in enumerate(biome_ids):
                idx = self.biome_index.get(biome, None)
                if idx is not None:
                    z_tilde[i] = z[i] - self.mu_table[idx] + self.mu0
        return z_tilde @ self.Pk.T

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
    if len(known_biomes) < 2:
        raise ValueError(f"BOLP requires at least 2 distinct known biomes, found {len(known_biomes)}")

    biome_centroids = {b: embeddings[idxs].mean(dim=0) for b, idxs in known_biomes.items()}
    mu0 = torch.stack(list(biome_centroids.values()), dim=0).mean(dim=0)
    B_count = len(biome_centroids)

    # Rank k = |G| - 1
    k = B_count - 1

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
