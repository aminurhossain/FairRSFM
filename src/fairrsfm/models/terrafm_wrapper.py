"""
TerraFM Model Wrapper for FairRSFM Benchmark
=============================================
Provides a unified PyTorch nn.Module for TerraFM (MBZUAI Oryx ViT-Base / ViT-Large),
handling input channel adaptation, weight loading, and feature extraction
for downstream classification (EuroSAT, BigEarthNet, So2Sat) and dense
semantic segmentation (m-SA-Crop-Type, MMEarth-20k, CloudSEN12).

References:
- TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation (ICLR 2026, arXiv:2506.06281)
- MBZUAI Oryx: https://github.com/mbzuai-oryx/TerraFM
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add upstream TerraFM implementation to path
UPSTREAM_TERRAFM_DIR = Path(__file__).resolve().parent / "upstream" / "terrafm"
if str(UPSTREAM_TERRAFM_DIR) not in sys.path and UPSTREAM_TERRAFM_DIR.is_dir():
    sys.path.insert(0, str(UPSTREAM_TERRAFM_DIR))

try:
    from terrafm import terrafm_base, terrafm_large, normalize, S2_BANDS, STATS
except ImportError:
    terrafm_base = None
    terrafm_large = None
    normalize = None
    S2_BANDS = None
    STATS = None

from fairrsfm.models.dofa_wrapper import UPerNetHead


class TerraFMFeatureExtractor(nn.Module):
    """
    TerraFM Backbone Feature Extractor.
    Extracts high-level feature representations and multi-scale feature maps
    using pre-trained TerraFM weights.
    """
    def __init__(
        self,
        model_size: str = "base",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        sensor: str = "s2_l2a",
        is_l2a: bool = True,
        out_indices: List[int] = [3, 5, 7, 11],
    ):
        super().__init__()
        if terrafm_base is None:
            raise ImportError(
                f"TerraFM module not found at {UPSTREAM_TERRAFM_DIR}. "
                "Ensure upstream/terrafm is populated."
            )

        self.model_size = model_size.lower()
        self.sensor = sensor
        self.is_l2a = is_l2a
        self.out_indices = out_indices
        self.freeze_backbone = freeze_backbone

        if self.model_size == "large":
            self.backbone = terrafm_large()
            self.embed_dim = 1024
        else:
            self.backbone = terrafm_base()
            self.embed_dim = 768

        # Load weights if checkpoint is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)
        elif checkpoint_path:
            print(f"[TerraFM] Warning: Checkpoint path '{checkpoint_path}' does not exist.")

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def load_pretrained_weights(self, checkpoint_path: str):
        print(f"[TerraFM] Loading pretrained weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"[TerraFM] Weights loaded successfully: {msg}")

    def _prepare_input(self, x: Tensor) -> Tensor:
        """
        Adapts arbitrary input bands to TerraFM's 12-channel Sentinel-2 format.
        If fewer than 12 channels are provided, missing channels are zero-padded.
        """
        B, C, H, W = x.shape
        if C == 12:
            return normalize(x, self.sensor)
        elif C == 2:  # Sentinel-1
            return normalize(x, "s1")
        elif C < 12:
            # Pad missing bands to 12
            pad = torch.zeros(B, 12 - C, H, W, dtype=x.dtype, device=x.device)
            padded = torch.cat([x, pad], dim=1)
            return normalize(padded, self.sensor)
        else:
            return normalize(x[:, :12], self.sensor)

    def forward(self, x: Tensor) -> Tensor:
        """Extracts CLS token embedding: [B, embed_dim]"""
        if self.freeze_backbone:
            self.backbone.eval()
        x_norm = self._prepare_input(x)
        return self.backbone(x_norm, is_l2a=self.is_l2a)

    def extract_features(self, x: Tensor) -> List[Tensor]:
        """Extracts multi-scale 2D feature maps for UPerNet: List of 4 x [B, embed_dim, H/16, W/16]"""
        if self.freeze_backbone:
            self.backbone.eval()
        x_norm = self._prepare_input(x)
        return self.backbone.extract_feature(x_norm, out_indices=self.out_indices, is_l2a=self.is_l2a)


class TerraFMClassificationClassifier(nn.Module):
    """
    Standard Linear Probe classifier for TerraFM on frozen representations.
    """
    def __init__(
        self,
        num_classes: int,
        model_size: str = "base",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = TerraFMFeatureExtractor(
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
        )
        embed_dim = self.feature_extractor.embed_dim
        layers = []
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(embed_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        cls_feat = self.feature_extractor(x)
        return self.head(cls_feat)


class TerraFMUPerNetSegmentationClassifier(nn.Module):
    """
    Standard UPerNet semantic segmentation model for TerraFM on frozen representations.
    """
    def __init__(
        self,
        num_classes: int,
        model_size: str = "base",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        fpn_dim: int = 256,
    ):
        super().__init__()
        self.feature_extractor = TerraFMFeatureExtractor(
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
        )
        embed_dim = self.feature_extractor.embed_dim
        self.upernet_head = UPerNetHead(
            in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
            fpn_dim=fpn_dim,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[2], x.shape[3]
        feats = self.feature_extractor.extract_features(x)
        logits = self.upernet_head(feats)
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
