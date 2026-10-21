"""
SatMAE Model Wrapper for FairRSFM Benchmark
============================================
Provides a unified PyTorch nn.Module for SatMAE (ViT-Base / ViT-Large),
handling input channel adaptation, weight loading, and feature extraction
for downstream classification (EuroSAT, BigEarthNet, So2Sat) and segmentation.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

# Locate upstream SatMAE modules relative to this file
UPSTREAM_SATMAE_DIR = Path(__file__).resolve().parent / "upstream" / "SatMAE"
if str(UPSTREAM_SATMAE_DIR) not in sys.path and UPSTREAM_SATMAE_DIR.is_dir():
    sys.path.insert(0, str(UPSTREAM_SATMAE_DIR))

try:
    import models_vit_group_channels
    import models_vit
except ImportError:
    models_vit_group_channels = None
    models_vit = None


class SatMAEFeatureExtractor(nn.Module):
    """
    SatMAE Backbone Feature Extractor.
    Extracts high-level feature representations from satellite imagery using pre-trained SatMAE weights.
    """
    def __init__(
        self,
        model_type: str = "vit_base_patch16",
        multispectral: bool = True,
        in_chans: int = 10,
        patch_size: int = 16,
        checkpoint_path: Optional[str] = None,
        img_size: int = 96,
        freeze_backbone: bool = True
    ):
        super().__init__()
        if models_vit_group_channels is None and multispectral:
            raise ImportError(
                f"SatMAE group channels module not found at {UPSTREAM_SATMAE_DIR}. "
                "Ensure upstream/SatMAE is populated."
            )

        self.model_type = model_type
        self.multispectral = multispectral
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone

        if multispectral:
            # SatMAE multi-spectral uses 10 bands grouped into 3 spectral sets:
            # Group 0: (0, 1, 2, 6) -> e.g. B2, B3, B4, B8
            # Group 1: (3, 4, 5, 7) -> e.g. B5, B6, B7, B8A
            # Group 2: (8, 9)       -> e.g. B11, B12
            channel_groups = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
            if hasattr(models_vit_group_channels, model_type):
                fn = getattr(models_vit_group_channels, model_type)
                self.backbone = fn(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    channel_groups=channel_groups,
                    global_pool=True
                )
            else:
                raise ValueError(f"Unknown model_type '{model_type}' in models_vit_group_channels")
        else:
            if hasattr(models_vit, model_type):
                fn = getattr(models_vit, model_type)
                self.backbone = fn(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    global_pool=True
                )
            else:
                raise ValueError(f"Unknown model_type '{model_type}' in models_vit")

        self.embed_dim = self.backbone.embed_dim

        # Load weights if checkpoint is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)
        elif checkpoint_path:
            print(f"[SatMAE] Warning: Checkpoint path '{checkpoint_path}' does not exist. Initializing with random weights.")

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def load_pretrained_weights(self, checkpoint_path: str):
        print(f"[SatMAE] Loading pretrained weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        
        model_dict = self.backbone.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        missing, unexpected = self.backbone.load_state_dict(filtered_dict, strict=False)
        print(f"[SatMAE] Loaded {len(filtered_dict)}/{len(model_dict)} layers successfully.")
        if missing:
            print(f"[SatMAE] Missing keys: {len(missing)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor of shape (B, C, H, W)
        Output: Feature tensor of shape (B, embed_dim)
        """
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)
        return feats


class SatMAEClassifier(nn.Module):
    """
    Linear Probe / Classifier wrapping SatMAE feature extractor.
    """
    def __init__(
        self,
        num_classes: int,
        model_type: str = "vit_base_patch16",
        multispectral: bool = True,
        in_chans: int = 10,
        patch_size: int = 16,
        checkpoint_path: Optional[str] = None,
        img_size: int = 96,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.encoder = SatMAEFeatureExtractor(
            model_type=model_type,
            multispectral=multispectral,
            in_chans=in_chans,
            patch_size=patch_size,
            checkpoint_path=checkpoint_path,
            img_size=img_size,
            freeze_backbone=freeze_backbone
        )
        self.head = nn.Linear(self.encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits
