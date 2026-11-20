#!/usr/bin/env python3
"""
TerraFM ViT-Base — MMEarth-20k CLI Runner
Delegates to run_unified_mmearth.py with --model terrafm
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
    main()
