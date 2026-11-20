#!/usr/bin/env python3
"""
Prithvi EO V2 300M — CloudSEN12 DBR
"""
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[5]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.scripts.cloudsen12.run_cloudsen12 import main

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "prithvi"])
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "dbr"])
    main()
