#!/usr/bin/env python3
"""
TerraFM ViT-Base — MMEarth-20k 14-Biome Semantic Segmentation (BOLP)
====================================================================
Protocol:
  • Biome-Orthogonal Linear Probing (BOLP) via null-space projection of biome centroids
  • Frozen ViT-Base Backbone + Multi-Level UPerNet Decoder (fpn_dim=256)
  • 12 Sentinel-2 Optical Channels (224x224)
  • Full 50 Epochs Training (NO early stopping)
  • Full 14-Biome Validation & Test evaluation saved every epoch from Epoch 0
  • Outputs: test_selected.json, test_results.json, test_summary.md,
             test_biome_metrics.csv, conf_mat_14x9x9.npy, epoch_eval_history.json
"""

import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_DIR))

from experiments.mmearth.run_unified_mmearth import main

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "terrafm"])
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "bolp"])
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "train"])
    main()
