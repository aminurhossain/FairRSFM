"""
DOFA Model Wrapper for FairRSFM Benchmark
=========================================
Standalone, self-contained implementation of Dynamic One-For-All (DOFA)
vision transformer for satellite imagery. Eliminates any torchgeo / GDAL / PROJ
runtime dependencies.

Reference:
  DOFA: Dynamic One-For-All (arXiv:2403.15356)
"""

import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from timm.models.vision_transformer import Block


# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE DOFA ARCHITECTURE (STANDALONE PYTORCH)
# ─────────────────────────────────────────────────────────────────────────────

def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute 1D sine/cosine position embedding for wavelengths."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


class TransformerWeightGenerator(nn.Module):
    """Dynamic weight generator for DOFA per-band patch embedding."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    """Fully-connected residual layer for wavelength projection."""
    def __init__(self, linear_size: int = 128) -> None:
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class DOFAEmbedding(nn.Module):
    """Dynamic One-For-All (DOFA) wavelength-conditioned patch embedding."""
    def __init__(
        self, dynamic_embed_dim: int = 128, kernel_size: int = 16, embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)

        self.weight_generator = TransformerWeightGenerator(
            dynamic_embed_dim, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(dynamic_embed_dim)
        self._init_weights()

    def _init_weight(self, m: object) -> None:
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        self.weight_generator.apply(self._init_weight)
        self.fclayer.apply(self._init_weight)

    def forward(self, x: Tensor, wavelengths: Tensor) -> Tuple[Tensor, Tensor]:
        inplanes = wavelengths.size(0)
        # Convert wavelength to nm (micrometers * 1000)
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            x, weights, bias=bias, stride=self.kernel_size, padding=0, dilation=1
        )
        x = dynamic_out.flatten(2).transpose(1, 2)
        return x, waves


class DOFA(nn.Module):
    """Dynamic One-For-All (DOFA) Vision Transformer."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        drop_rate: float = 0.0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dynamic_embed_dim: int = 128,
        num_classes: int = 0,
        global_pool: bool = True,
        mlp_ratio: float = 4.0,
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dynamic_embed_dim = dynamic_embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.mlp_ratio = mlp_ratio

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = DOFAEmbedding(
            dynamic_embed_dim=dynamic_embed_dim,
            kernel_size=patch_size,
            embed_dim=embed_dim,
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x: Tensor, wavelengths: List[float]) -> Tensor:
        wavelist = torch.tensor(wavelengths, device=x.device).float()
        x, _ = self.patch_embed(x, wavelist)

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome: Tensor = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_patch_features(self, x: Tensor, wavelengths: List[float]) -> Tensor:
        wavelist = torch.tensor(wavelengths, device=x.device).float()
        x, _ = self.patch_embed(x, wavelist)

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        patches = x[:, 1:, :]
        if hasattr(self, "fc_norm") and self.fc_norm is not None:
            patches = self.fc_norm(patches)
        elif hasattr(self, "norm") and self.norm is not None:
            patches = self.norm(patches)
        return patches

    def forward_intermediate_features(
        self, x: Tensor, wavelengths: List[float], out_indices: List[int] = [2, 5, 8, 11]
    ) -> List[Tensor]:
        wavelist = torch.tensor(wavelengths, device=x.device).float()
        x, _ = self.patch_embed(x, wavelist)

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in out_indices:
                patches = x[:, 1:, :]
                b, l, d = patches.shape
                h = w = int(l ** 0.5)
                feat = patches.permute(0, 2, 1).reshape(b, d, h, w)
                features.append(feat)
        return features

    def forward(self, x: Tensor, wavelengths: List[float]) -> Tensor:
        x = self.forward_features(x, wavelengths)
        x = self.head_drop(x)
        x = self.head(x)
        return x


def dofa_base_patch16_224(**kwargs) -> DOFA:
    defaults = {"patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12}
    defaults.update(kwargs)
    return DOFA(**defaults)


def dofa_large_patch16_224(**kwargs) -> DOFA:
    defaults = {"patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16}
    defaults.update(kwargs)
    return DOFA(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOFA BENCHMARK WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

class DOFAFeatureExtractor(nn.Module):
    """
    DOFA Backbone Feature Extractor.
    Extracts representations from satellite imagery using dynamic wavelength-conditioned embeddings.
    """
    def __init__(
        self,
        model_type: str = "dofa_base_patch16_224",
        checkpoint_path: Optional[str] = None,
        wavelengths: Optional[List[float]] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_type = model_type
        self.freeze_backbone = freeze_backbone
        self.wavelengths = wavelengths

        if model_type == "dofa_base_patch16_224":
            self.backbone = dofa_base_patch16_224()
        elif model_type == "dofa_large_patch16_224":
            self.backbone = dofa_large_patch16_224()
        else:
            raise ValueError(f"Unknown DOFA model_type: {model_type}")

        self.embed_dim = self.backbone.embed_dim

        # Load weights if checkpoint is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)
        elif checkpoint_path:
            print(f"Warning: Checkpoint path '{checkpoint_path}' does not exist. Initializing with random weights.")

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def load_pretrained_weights(self, checkpoint_path: str):
        print(f"[DOFA] Loading pretrained weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"[DOFA] Pretrained weights loaded successfully (Missing: {len(missing)}, Unexpected: {len(unexpected)}).")

    def forward(self, x: torch.Tensor, wavelengths: Optional[List[float]] = None) -> torch.Tensor:
        wl = wavelengths if wavelengths is not None else self.wavelengths
        if wl is None:
            raise ValueError("Wavelengths must be provided either at initialization or in forward().")

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone.forward_features(x, wl)
        else:
            feats = self.backbone.forward_features(x, wl)
        return feats

    def forward_patches(self, x: torch.Tensor, wavelengths: Optional[List[float]] = None) -> torch.Tensor:
        wl = wavelengths if wavelengths is not None else self.wavelengths
        if wl is None:
            raise ValueError("Wavelengths must be provided either at initialization or in forward().")

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                patches = self.backbone.forward_patch_features(x, wl)
        else:
            patches = self.backbone.forward_patch_features(x, wl)
        return patches

    def forward_intermediates(
        self, x: torch.Tensor, wavelengths: Optional[List[float]] = None, out_indices: List[int] = [2, 5, 8, 11]
    ) -> List[torch.Tensor]:
        wl = wavelengths if wavelengths is not None else self.wavelengths
        if wl is None:
            raise ValueError("Wavelengths must be provided either at initialization or in forward().")

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone.forward_intermediate_features(x, wl, out_indices=out_indices)
        else:
            feats = self.backbone.forward_intermediate_features(x, wl, out_indices=out_indices)
        return feats


class DOFAClassifier(nn.Module):
    """
    Linear Probe / Classifier wrapping DOFA feature extractor.
    Follows official DOFA protocol:
      BatchNorm1d(embed_dim, affine=False, eps=1e-6) + Linear(embed_dim, num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        model_type: str = "dofa_base_patch16_224",
        checkpoint_path: Optional[str] = None,
        wavelengths: Optional[List[float]] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.wavelengths = wavelengths
        self.encoder = DOFAFeatureExtractor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            wavelengths=wavelengths,
            freeze_backbone=freeze_backbone,
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6),
            nn.Linear(self.encoder.embed_dim, num_classes),
        )
        init.trunc_normal_(self.head[1].weight, std=0.01)
        if self.head[1].bias is not None:
            nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor, wavelengths: Optional[List[float]] = None) -> torch.Tensor:
        feats = self.encoder(x, wavelengths)
        logits = self.head(feats)
        return logits


class DOFASegmentationClassifier(nn.Module):
    """
    Linear Probe / Segmentation Head wrapping DOFA patch feature extractor.
    Follows DOFA semantic segmentation protocol:
      BatchNorm2d(embed_dim, affine=False, eps=1e-6) + Conv2d(embed_dim, num_classes, 1) + Bilinear Interpolation
    """
    def __init__(
        self,
        num_classes: int,
        model_type: str = "dofa_base_patch16_224",
        checkpoint_path: Optional[str] = None,
        wavelengths: Optional[List[float]] = None,
        freeze_backbone: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.wavelengths = wavelengths
        self.img_size = img_size
        self.grid_size = img_size // 16
        self.encoder = DOFAFeatureExtractor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            wavelengths=wavelengths,
            freeze_backbone=freeze_backbone,
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(self.encoder.embed_dim, affine=False, eps=1e-6),
            nn.Conv2d(self.encoder.embed_dim, num_classes, kernel_size=1),
        )
        init.trunc_normal_(self.head[1].weight, std=0.01)
        if self.head[1].bias is not None:
            nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor, wavelengths: Optional[List[float]] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        patches = self.encoder.forward_patches(x, wavelengths)  # (B, 196, 768)
        spatial_feats = patches.permute(0, 2, 1).view(b, self.encoder.embed_dim, self.grid_size, self.grid_size)
        logits = self.head(spatial_feats)  # (B, num_classes, 14, 14)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


class PPM(nn.Module):
    """Pyramid Pooling Module for UPerNet."""
    def __init__(self, in_dim: int, out_dim: int, pool_scales: Tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ) for scale in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + len(pool_scales) * out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        outs = [x]
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(outs, dim=1))


class UPerNetHead(nn.Module):
    """
    Unified Perceptual Parsing (UPerNet) Decoder for Vision Transformers.
    Fuses multi-level features from layers [2, 5, 8, 11] into a Feature Pyramid.
    """
    def __init__(
        self,
        in_channels: List[int] = [768, 768, 768, 768],
        fpn_dim: int = 512,
        num_classes: int = 10,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        self.ppm = PPM(in_channels[3], fpn_dim)

        # Lateral convs (1x1)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        # FPN output convs (3x3)
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        # Multi-scale rescaler to transform single-scale ViT tokens to pyramid scales
        self.rescale_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels[0], in_channels[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.rescale_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels[1], in_channels[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.rescale_2 = nn.Identity()
        self.rescale_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fused classification head
        self.cls_seg = nn.Sequential(
            nn.Conv2d(4 * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: 4 feature maps at (B, 768, 14, 14)
        c0 = self.rescale_0(feats[0])  # (B, 768, 56, 56)
        c1 = self.rescale_1(feats[1])  # (B, 768, 28, 28)
        c2 = self.rescale_2(feats[2])  # (B, 768, 14, 14)
        c3 = self.rescale_3(feats[3])  # (B, 768, 7, 7)

        p3 = self.ppm(c3)  # (B, 512, 7, 7)

        p2 = self.lateral_convs[2](c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p2 = self.fpn_convs[2](p2)

        p1 = self.lateral_convs[1](c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        p1 = self.fpn_convs[1](p1)

        p0 = self.lateral_convs[0](c0) + F.interpolate(p1, size=c0.shape[2:], mode="bilinear", align_corners=False)
        p0 = self.fpn_convs[0](p0)

        # Upsample all stages to p0 resolution (56x56)
        p1_up = F.interpolate(p1, size=p0.shape[2:], mode="bilinear", align_corners=False)
        p2_up = F.interpolate(p2, size=p0.shape[2:], mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=p0.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([p0, p1_up, p2_up, p3_up], dim=1)  # (B, 2048, 56, 56)
        logits = self.cls_seg(fused)  # (B, num_classes, 56, 56)
        return logits


class DOFAUperNetSegmentationClassifier(nn.Module):
    """
    Official DOFA Semantic Segmentation Model with UPerNet Decoder.
    Extracts multi-stage features from blocks [2, 5, 8, 11] and decodes via UPerNet.
    """
    def __init__(
        self,
        num_classes: int,
        model_type: str = "dofa_base_patch16_224",
        checkpoint_path: Optional[str] = None,
        wavelengths: Optional[List[float]] = None,
        freeze_backbone: bool = True,
        fpn_dim: int = 512,
        out_indices: List[int] = [2, 5, 8, 11],
    ):
        super().__init__()
        self.wavelengths = wavelengths
        self.out_indices = out_indices
        self.encoder = DOFAFeatureExtractor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            wavelengths=wavelengths,
            freeze_backbone=freeze_backbone,
        )
        d = self.encoder.embed_dim
        self.head = UPerNetHead(
            in_channels=[d, d, d, d],
            fpn_dim=fpn_dim,
            num_classes=num_classes,
            dropout_ratio=0.1,
        )

    def forward(self, x: torch.Tensor, wavelengths: Optional[List[float]] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        multi_feats = self.encoder.forward_intermediates(x, wavelengths, out_indices=self.out_indices)
        logits = self.head(multi_feats)  # (B, num_classes, 56, 56)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


