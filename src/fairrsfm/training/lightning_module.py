"""
Unified PyTorch Lightning Module for FairRSFM Probes
====================================================
Supports frozen backbone linear probing with ERM, DBR, BOLP, and GroupDRO.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from fairrsfm.methods.dbr import DynamicBiomeReweighter
from fairrsfm.methods.bolp import BOLPProjector
from fairrsfm.methods.group_dro import GroupDROLossTracker


class FairRSFMProbe(L.LightningModule):
    """
    Unified downstream probe wrapper supporting all 4 debiasing methods.
    """
    def __init__(
        self,
        model: nn.Module,
        method: str = "erm",
        num_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        class_weights: Optional[List[float]] = None,
        biome_map: Optional[Dict[str, str]] = None,
        reweighter: Optional[DynamicBiomeReweighter] = None,
        bolp_projector: Optional[BOLPProjector] = None,
        group_names: Optional[List[str]] = None,
        group_lr: float = 0.01,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "reweighter", "bolp_projector"])
        self.model = model
        self.method = method.lower()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.biome_map = biome_map or {}
        self.freeze_backbone = freeze_backbone

        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=w)
        self.loss_fn_none = nn.CrossEntropyLoss(weight=w, reduction="none")

        # Method-specific components
        self.reweighter = reweighter
        self.bolp_projector = bolp_projector
        if self.method == "groupdro" and group_names:
            self.group_dro = GroupDROLossTracker(group_names=group_names, group_lr=group_lr)
        else:
            self.group_dro = None

        if self.freeze_backbone and hasattr(self.model, "encoder"):
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()

    def forward(self, x: torch.Tensor, sample_ids: Optional[List[str]] = None) -> torch.Tensor:
        if self.freeze_backbone and hasattr(self.model, "encoder"):
            self.model.encoder.eval()

        if self.method == "bolp" and self.bolp_projector is not None:
            # Extract features and project
            if hasattr(self.model, "encoder"):
                feats = self.model.encoder(x)
                if isinstance(feats, (list, tuple)):
                    feats = feats[-1]
                if feats.dim() == 4:
                    feats = feats.mean(dim=[-2, -1])
                elif feats.dim() == 3:
                    feats = feats.mean(dim=1)
                biome_ids = [self.biome_map.get(str(s), "Unknown") for s in (sample_ids or [])]
                proj_feats = self.bolp_projector(feats, biome_ids=biome_ids)
                if hasattr(self.model, "head"):
                    return self.model.head(proj_feats)
                return proj_feats

        out = self.model(x)
        return out.logits if hasattr(out, "logits") else (out.output if hasattr(out, "output") else out)

    def training_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x, sample_ids=sample_ids)

        if self.method == "dbr" and self.reweighter is not None:
            loss_per_sample = self.loss_fn_none(logits, y)
            sample_w = self.reweighter.get_weights_tensor(sample_ids, self.biome_map, device=x.device)
            loss = (loss_per_sample * sample_w).mean()
        elif self.method == "groupdro" and self.group_dro is not None:
            loss_per_sample = self.loss_fn_none(logits, y)
            sample_biomes = [self.biome_map.get(str(s), "Unknown") for s in sample_ids]
            loss = self.group_dro(loss_per_sample, sample_biomes)
        else:
            loss = self.loss_fn(logits, y)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x, sample_ids=sample_ids)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def test_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x, sample_ids=sample_ids)
        loss = self.loss_fn(logits, y)
        self.log("test/loss", loss, batch_size=x.size(0))
        return {"logits": logits, "y": y, "sample_ids": sample_ids}

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
