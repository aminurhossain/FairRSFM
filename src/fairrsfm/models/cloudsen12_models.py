"""
Unified CloudSEN12 Model Factory for All Remote Sensing Foundation Models
========================================================================
Supports: TerraFM, Prithvi EO V2 300M, SatMAE, DOFA.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import pyproj.datadir
proj_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"]  = proj_dir

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairrsfm.models.terrafm_wrapper import TerraFMUPerNetSegmentationClassifier
from fairrsfm.models.dofa_wrapper import DOFAUperNetSegmentationClassifier
from fairrsfm.models.satmae_wrapper import SatMAEUperNetSegmentationClassifier

REPO_DIR = Path(__file__).resolve().parents[3]

DOFA_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 0.842, 0.945, 1.610, 2.190]

DEFAULT_CHECKPOINTS = {
    "terrafm": REPO_DIR / "checkpoints/terrafm/TerraFM-B.pth",
    "dofa":    REPO_DIR / "checkpoints/dofa/dofa_base_patch16_224.pth",
    "satmae":  REPO_DIR / "checkpoints/satmae/pretrain-vit-base-e199.pth",
    "prithvi": REPO_DIR / "checkpoints/Prithvi_EO_V2_300M_TL.pt",
}

# Band indices selected from S2 12-band order:
# ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MODEL_BAND_INDICES = {
    "terrafm": list(range(12)),                  # 12 optical bands
    "dofa":    list(range(12)),                  # 12 optical bands
    "satmae":  [1, 2, 3, 4, 5, 6, 7, 8, 10, 11], # 10 optical bands (B02..B08, B8A, B11, B12)
    "prithvi": [1, 2, 3, 8, 10, 11],             # 6 HLS bands (B02, B03, B04, B8A, B11, B12)
}


class PrithviCloudSEN12Wrapper(nn.Module):
    """
    Wraps Prithvi EO V2 300M UPerNet segmentation model for CloudSEN12.
    Ensures input of shape [B, 6, H, W] outputs logits [B, num_classes, H, W].
    """
    def __init__(
        self,
        num_classes: int = 4,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        from terratorch.models import PrithviModelFactory
        from terratorch.models.backbones.prithvi_vit import HLSBands

        factory = PrithviModelFactory()
        self.model = factory.build_model(
            task="segmentation",
            backbone="prithvi_eo_v2_300_tl",
            decoder="UperNetDecoder",
            bands=[
                HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
                HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2
            ],
            num_classes=num_classes,
            pretrained=True if checkpoint_path else False,
            backbone_ckpt_path=checkpoint_path,
            decoder_channels=256,
            decoder_scale_modules=True,
            backbone_num_frames=1,
            rescale=True,
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone and hasattr(self.model, "encoder"):
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone and hasattr(self.model, "encoder"):
            self.model.encoder.eval()
        out = self.model(x)
        return out.output if hasattr(out, "output") else out


def get_cloudsen12_band_indices(model_name: str) -> List[int]:
    return MODEL_BAND_INDICES[model_name.lower()]


def build_cloudsen12_model(
    model_name: str,
    num_classes: int = 4,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    name = model_name.lower()
    ckpt = checkpoint_path or str(DEFAULT_CHECKPOINTS.get(name, ""))
    if name == "terrafm":
        return TerraFMUPerNetSegmentationClassifier(
            num_classes=num_classes,
            checkpoint_path=ckpt,
            freeze_backbone=freeze_backbone,
            fpn_dim=256,
        )
    elif name == "dofa":
        return DOFAUperNetSegmentationClassifier(
            num_classes=num_classes,
            checkpoint_path=ckpt,
            wavelengths=DOFA_WAVELENGTHS,
            freeze_backbone=freeze_backbone,
            fpn_dim=512,
        )
    elif name == "satmae":
        return SatMAEUperNetSegmentationClassifier(
            num_classes=num_classes,
            checkpoint_path=ckpt,
            freeze_backbone=freeze_backbone,
            fpn_dim=512,
        )
    elif name == "prithvi":
        return PrithviCloudSEN12Wrapper(
            num_classes=num_classes,
            checkpoint_path=ckpt,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
