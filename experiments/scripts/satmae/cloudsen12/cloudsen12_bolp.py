#!/usr/bin/env python3
"""
SatMAE ViT-Base — CloudSEN12 BOLP
"""
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[4]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.scripts.cloudsen12.run_cloudsen12 import main

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "satmae"])
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "bolp"])
    main()
