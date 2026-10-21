"""
Prithvi EO V2 Model Wrapper for FairRSFM Benchmark
==================================================
Provides a unified interface for NASA/IBM Prithvi EO 2.0 (300M parameters),
supporting both classification (EuroSAT, BigEarthNet) and semantic segmentation
(m-SA-crop-type, MMEarth-20k).

References:
- IBM / NASA Granite Geospatial Prithvi EO 2.0
- TerraTorch: https://github.com/IBM/terratorch
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn as nn

try:
    from terratorch.models import PrithviModelFactory
    from terratorch.models.backbones.prithvi_vit import HLSBands
    TERRATORCH_AVAILABLE = True
except Exception:
    TERRATORCH_AVAILABLE = False
    PrithviModelFactory = None
    HLSBands = None


# Default bands for Sentinel-2 / Landsat in Prithvi EO V2
DEFAULT_S2_BANDS = [
    "BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"
]


class PrithviModelWrapper(nn.Module):
    """
    Unified Wrapper for Prithvi EO V2 300M Foundation Model.
    """
    def __init__(
        self,
        task: str = "classification",
        num_classes: int = 10,
        bands: Optional[List[str]] = None,
        decoder: str = "MLPDecoder",
        checkpoint_path: Optional[Union[str, Path]] = None,
        freeze_backbone: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        if not TERRATORCH_AVAILABLE:
            raise ImportError(
                "terratorch is required to build Prithvi models. "
                "Install it via `pip install terratorch`."
            )

        self.task = task
        self.num_classes = num_classes
        self.bands = bands or DEFAULT_S2_BANDS
        self.decoder = decoder
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.freeze_backbone = freeze_backbone

        factory = PrithviModelFactory()

        # Build underlying model via TerraTorch factory
        build_kwargs = {
            "task": task,
            "backbone": "prithvi_eo_v2_300_tl",
            "decoder": decoder,
            "bands": self.bands,
            "num_classes": num_classes,
            "pretrained": True if self.checkpoint_path else False,
        }
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            build_kwargs["backbone_ckpt_path"] = self.checkpoint_path
        elif self.checkpoint_path:
            print(f"[Prithvi] Warning: Checkpoint '{self.checkpoint_path}' not found on disk. Initializing without checkpoint.")

        self.model = factory.build_model(**build_kwargs)

        if freeze_backbone and hasattr(self.model, "encoder"):
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, T, H, W)
        Returns:
            Logits tensor
        """
        if self.freeze_backbone and hasattr(self.model, "encoder"):
            self.model.encoder.eval()
        return self.model(x)


def get_prithvi_model(
    task: str = "classification",
    num_classes: int = 10,
    bands: Optional[List[str]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    freeze_backbone: bool = True
) -> nn.Module:
    """Convenience helper to instantiate Prithvi EO V2 300M."""
    decoder = "MLPDecoder" if task == "classification" else "FCNDecoder"
    return PrithviModelWrapper(
        task=task,
        num_classes=num_classes,
        bands=bands,
        decoder=decoder,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone
    )
