#!/usr/bin/env python3
"""
TerraFM ViT-Base — South Africa Crop Type (m-SA-crop-type) ERM Baseline
"""
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[5]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.scripts.terrafm.segmentation.sa_crop.run_sa_crop import main

if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "erm"])
    main()
