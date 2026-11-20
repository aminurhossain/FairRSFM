#!/usr/bin/env python3
"""
Universal CloudSEN12 Benchmark & Mitigation Pipeline
===================================================
A unified entry point for Foundation Models (TerraFM, Prithvi, SatMAE, DOFA)
and Fairness Mitigation Methods (ERM, Group DRO, DBR, BOLP) on the
Stratified 14-Biome CloudSEN12 Benchmark (4-class semantic segmentation).

Usage:
  python run_cloudsen12.py --model terrafm --method <erm|groupdro|dbr|bolp> --mode <train|eval>
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Environment & Proj Setup
os.environ["HUGGINGFACE_HUB_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]     = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

import pyproj.datadir
proj_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"]  = proj_dir

REPO_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from fairrsfm.data.cloudsen12_datamodule import (
    CloudSEN12SegmentationDataModule,
    NUM_CLASSES,
    RESOLVE_BIOME_NAMES,
)
from fairrsfm.models.terrafm_wrapper import TerraFMUPerNetSegmentationClassifier
from fairrsfm.evaluation.unified_cloudsen12_evaluator import (
    evaluate_dataset_confusion_matrix,
    evaluate_cloudsen12_pipeline,
    CloudSEN12EpochValTestEvalCallback,
)

from cloudsen12_erm import CloudSEN12ERMModule
from cloudsen12_groupdro import CloudSEN12GroupDROModule
from cloudsen12_dbr import CloudSEN12DBRModule
from cloudsen12_bolp import CloudSEN12BOLPModule

DEFAULT_CHECKPOINT = REPO_DIR / "checkpoints/terrafm/TerraFM-B.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Universal CloudSEN12 Benchmark Pipeline")
    p.add_argument("--model", choices=["terrafm"], default="terrafm",
                   help="Foundation model backbone")
    p.add_argument("--method", choices=["erm", "groupdro", "dbr", "bolp"], default="erm",
                   help="Fairness optimization / mitigation method")
    p.add_argument("--mode", choices=["train", "eval"], default="train",
                   help="Run mode: train from scratch or evaluate existing checkpoint")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to pretrained model checkpoint or trained evaluation checkpoint")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (defaults to Final_cloudsen12/<model>/<method>/)")
    p.add_argument("--gpu", type=str, default="0",
                   help="CUDA GPU device ID (e.g. 0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of epochs (default 50, no early stopping)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for AdamW")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay for AdamW")
    p.add_argument("--group-lr", type=float, default=0.01,
                   help="Step size for Group DRO exponentiated gradient ascent")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="DBR smoothing exponent gamma")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="BOLP convex combination weight alpha")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker count")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_dir = Path(args.output_dir) if args.output_dir else REPO_DIR / f"Final_cloudsen12/{args.model}/{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f" UNIVERSAL CLOUDSEN12 BENCHMARK")
    print(f" Model       : {args.model.upper()}")
    print(f" Method      : {args.method.upper()}")
    print(f" Mode        : {args.mode.upper()}")
    print(f" Output Dir  : {out_dir}")
    print(f" Device      : {device}")
    print("=" * 80 + "\n")

    dm = CloudSEN12SegmentationDataModule(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    dm.setup()

    # Pretrained backbone checkpoint
    backbone_ckpt = args.checkpoint or str(DEFAULT_CHECKPOINT)

    # ── TRAINING MODE ─────────────────────────────────────────────────────────
    if args.mode == "train":
        L.seed_everything(args.seed, workers=True)

        if args.method == "erm":
            model = CloudSEN12ERMModule(
                checkpoint_path=backbone_ckpt,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )
        elif args.method == "groupdro":
            model = CloudSEN12GroupDROModule(
                checkpoint_path=backbone_ckpt,
                lr=args.lr,
                group_lr=args.group_lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )
        elif args.method == "dbr":
            train_groups = [dm.train_dataset.biome_map[idx] - 1 for idx in dm.train_dataset.indices]
            group_counts = {g: train_groups.count(g) for g in range(14)}
            model = CloudSEN12DBRModule(
                group_counts=group_counts,
                checkpoint_path=backbone_ckpt,
                lr=args.lr,
                gamma=args.gamma,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )
        elif args.method == "bolp":
            model = CloudSEN12BOLPModule(
                checkpoint_path=backbone_ckpt,
                lr=args.lr,
                alpha=args.alpha,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
            )

        ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir / "checkpoints"),
            filename="checkpoint-{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        lr_cb = LearningRateMonitor(logging_interval="epoch")
        prog_cb = TQDMProgressBar(refresh_rate=20)
        logger = TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")

        eval_cb = CloudSEN12EpochValTestEvalCallback(
            val_loader=dm.val_dataloader(),
            test_loader=dm.test_dataloader(),
            out_dir=out_dir,
            start_epoch=0,
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=args.epochs,
            callbacks=[ckpt_cb, lr_cb, prog_cb, eval_cb],
            logger=logger,
            log_every_n_steps=10,
        )

        print(f"[Training] Launching {args.epochs} epochs of {args.method.upper()} on CloudSEN12...")
        trainer.fit(model, datamodule=dm)

        # Select best checkpoint
        best_candidate = out_dir / "checkpoints/best_val_worst_miou.ckpt"
        if best_candidate.exists():
            best_ckpt = str(best_candidate)
        else:
            best_ckpt = ckpt_cb.best_model_path or str(out_dir / "checkpoints/last.ckpt")
        print(f"[Evaluation] Selected best checkpoint: {best_ckpt}")
        eval_model = model.model
    else:
        # Evaluation only mode
        eval_ckpt = args.checkpoint or str(out_dir / "checkpoints/last.ckpt")
        print(f"[Evaluation] Loading evaluation checkpoint: {eval_ckpt}")
        ckpt_dict = torch.load(eval_ckpt, map_location="cpu")
        state_dict = ckpt_dict.get("state_dict", ckpt_dict)

        model_core = TerraFMUPerNetSegmentationClassifier(
            num_classes=NUM_CLASSES,
            checkpoint_path=backbone_ckpt,
            freeze_backbone=True,
            fpn_dim=256,
        )
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}
        model_core.load_state_dict(clean_state_dict if clean_state_dict else state_dict, strict=False)
        eval_model = model_core

    eval_model.to(device)
    eval_model.eval()

    test_loader = dm.test_dataloader()
    print("[Evaluation] Computing 14-Biome 4x4 Confusion Matrix on Test Split...")
    conf_mat = evaluate_dataset_confusion_matrix(eval_model, test_loader, device=device)

    print(f"[Artifacts] Generating publication metrics into {out_dir}...")
    results = evaluate_cloudsen12_pipeline(
        conf_mat=conf_mat,
        output_dir=out_dir,
        model_name=args.model,
        method_name=args.method,
    )

    print(f"\n{'='*80}")
    print(f" [DONE] {args.model.upper()} ({args.method.upper()}) CloudSEN12 Pipeline Complete!")
    print(f" Test mIoU      : {results['global_metrics']['mIoU']*100:.2f}%")
    print(f" Macro F1       : {results['global_metrics']['Macro_F1']*100:.2f}%")
    print(f" Worst Biome    : {results['fairness_metrics']['M_worst']*100:.2f}%")
    print(f" PER Disparity  : {results['fairness_metrics']['PER']:.4f}")
    print(f" Cross-Group Std: {results['fairness_metrics']['Std']*100:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
