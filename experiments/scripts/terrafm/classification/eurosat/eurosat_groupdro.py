#!/usr/bin/env python3
"""
TerraFM ViT-Base — EuroSAT Classification (m-eurosat) Group DRO
"""
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[5]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.scripts.terrafm.classification.eurosat.run_eurosat import main

if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "groupdro"])
    main()
