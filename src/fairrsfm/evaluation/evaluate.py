#!/usr/bin/env python3
"""
FairRSFM Unified Checkpoint Evaluator
=====================================
Single entrypoint to evaluate any checkpoint across all 3 Remote Sensing
Foundation Models (Prithvi, SatMAE, DOFA) and all 4 Benchmark Datasets.

Usage:
    python evaluation/evaluate.py --model satmae --dataset eurosat --ckpt path/to/ckpt.ckpt
    python evaluation/evaluate.py --model dofa --dataset sa_crop --ckpt path/to/ckpt.ckpt
    python evaluation/evaluate.py --model prithvi --dataset bigearthnet --ckpt path/to/ckpt.ckpt
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Mapping of (model, dataset) to dedicated test script in evaluation/
EVAL_DISPATCH = {
    ("satmae", "eurosat"):      ROOT_DIR / "satmae/test_eurosat.py",
    ("satmae", "bigearthnet"):  ROOT_DIR / "satmae/test_bigearthnet.py",
    ("satmae", "sa_crop"):      ROOT_DIR / "satmae/test_sa_crop.py",
    ("satmae", "mmearth"):      ROOT_DIR / "satmae/test_mmearth.py",

    ("dofa", "eurosat"):        ROOT_DIR / "dofa/test_eurosat.py",
    ("dofa", "bigearthnet"):    ROOT_DIR / "dofa/test_bigearthnet.py",
    ("dofa", "sa_crop"):        ROOT_DIR / "dofa/test_sa_crop.py",
    ("dofa", "mmearth"):        ROOT_DIR / "dofa/test_mmearth.py",

    ("prithvi", "eurosat"):     ROOT_DIR / "prithvi/test_eurosat.py",
    ("prithvi", "bigearthnet"): ROOT_DIR / "prithvi/test_bigearthnet.py",
    ("prithvi", "sa_crop"):     ROOT_DIR / "prithvi/test_sa_crop.py",
    ("prithvi", "mmearth"):     ROOT_DIR / "prithvi/test_mmearth.py",
}


def main():
    parser = argparse.ArgumentParser(description="FairRSFM Unified Checkpoint Evaluator")
    parser.add_argument("--model", type=str, required=True, choices=["prithvi", "satmae", "dofa"], help="Foundation Model")
    parser.add_argument("--dataset", type=str, required=True, choices=["eurosat", "bigearthnet", "sa_crop", "mmearth"], help="Benchmark Dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    parser.add_argument("--out_json", type=str, default=None, help="Output metrics JSON path")
    args, unknown = parser.parse_known_args()

    key = (args.model.lower(), args.dataset.lower())
    if key not in EVAL_DISPATCH:
        print(f"Error: Unsupported combination: model='{args.model}', dataset='{args.dataset}'")
        sys.exit(1)

    script_path = EVAL_DISPATCH[key]
    if not script_path.exists():
        print(f"Error: Evaluator script not found at {script_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(script_path),
        "--ckpt", str(args.ckpt),
        "--gpu", str(args.gpu)
    ]
    if args.data_dir:
        cmd.extend(["--data_dir", str(args.data_dir)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.out_json:
        cmd.extend(["--out_json", str(args.out_json)])
    if unknown:
        cmd.extend(unknown)

    print(f"[Dispatch] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
