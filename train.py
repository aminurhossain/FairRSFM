#!/usr/bin/env python3
"""
FairRSFM Unified Training Entry Point
=====================================
Single CLI entry point to launch reproducible training runs across all 3
Remote Sensing Foundation Models (Prithvi, SatMAE, DOFA), 4 Benchmark Datasets,
and 4 Debiasing Paradigms (ERM, DBR, BOLP, GroupDRO).

Usage:
    python experiments/train.py --model satmae --dataset bigearthnet --method groupdro --seed 42
    python experiments/train.py --model prithvi --dataset eurosat --method dbr --seed 42
    python experiments/train.py --model dofa --dataset sa_crop --method bolp --seed 42
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Resolve repository root by walking up until pyproject.toml is found
_curr = Path(__file__).resolve()
while _curr.parent != _curr:
    if (_curr / "pyproject.toml").is_file():
        break
    _curr = _curr.parent
REPO_ROOT = _curr
SCRIPTS_DIR = REPO_ROOT / "experiments" / "scripts"


def get_script_path(model: str, dataset: str, method: str, head: str = "linear") -> Path:
    model = model.lower()
    dataset = dataset.lower().replace("-", "_")
    method = method.lower()
    head = head.lower() if head else "linear"

    if model == "prithvi":
        if dataset == "eurosat":
            return SCRIPTS_DIR / f"prithvi/classification/eurosat/eurosat_{method}.py"
        elif dataset in ("bigearthnet", "bigearth"):
            return SCRIPTS_DIR / f"prithvi/classification/bigearthnet/bigearthnet_{method}.py"
        elif dataset in ("sa_crop", "sacrop", "sa_crop_type"):
            return SCRIPTS_DIR / f"prithvi/segmentation/sa_crop/sa_crop_{method}.py"
        elif dataset in ("mmearth", "mmearth_20k"):
            return SCRIPTS_DIR / f"prithvi/segmentation/mmearth_20k/mmearth_{method}.py"

    elif model == "satmae":
        if dataset == "eurosat":
            return SCRIPTS_DIR / f"satmae/eurosat/eurosat_{method}.py"
        elif dataset in ("bigearthnet", "bigearth"):
            return SCRIPTS_DIR / f"satmae/bigearthnet/bigearthnet_{method}.py"
        elif dataset in ("sa_crop", "sacrop", "sa_crop_type"):
            return SCRIPTS_DIR / f"satmae/sa_crop/sa_crop_{method}.py"
        elif dataset in ("mmearth", "mmearth_20k"):
            return SCRIPTS_DIR / f"satmae/mmearth/mmearth_{method}.py"

    elif model == "dofa":
        if dataset == "eurosat":
            return SCRIPTS_DIR / f"dofa/eurosat/{method}.py"
        elif dataset in ("bigearthnet", "bigearth"):
            return SCRIPTS_DIR / f"dofa/bigearthnet/{method}.py"
        elif dataset in ("sa_crop_upernet", "sacrop_upernet") or (dataset in ("sa_crop", "sacrop", "sa_crop_type") and head == "upernet"):
            return SCRIPTS_DIR / f"dofa/sa_crop_upernet/{method}.py"
        elif dataset in ("sa_crop", "sacrop", "sa_crop_type"):
            return SCRIPTS_DIR / f"dofa/sa_crop/{method}.py"
        elif dataset in ("mmearth_upernet", "mmearth_20k_upernet") or (dataset in ("mmearth", "mmearth_20k") and head == "upernet"):
            return SCRIPTS_DIR / f"dofa/mmearth_upernet/{method}.py"
        elif dataset in ("mmearth", "mmearth_20k"):
            return SCRIPTS_DIR / f"dofa/mmearth/{method}.py"

    raise ValueError(
        f"Unsupported combination: model='{model}', dataset='{dataset}', method='{method}', head='{head}'.\n"
        f"Supported models: [prithvi, satmae, dofa]\n"
        f"Supported datasets: [eurosat, bigearthnet, sa_crop, mmearth, sa_crop_upernet, mmearth_upernet]\n"
        f"Supported methods: [erm, dbr, bolp, groupdro]\n"
        f"Supported heads: [linear, upernet]"
    )


def detect_supported_args(script_path: Path, env: dict = None) -> str:
    """Inspect script file content directly to determine exact CLI argument flags."""
    try:
        return script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="FairRSFM Unified Training Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["prithvi", "satmae", "dofa"],
        help="Foundation Model Backbone",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eurosat", "bigearthnet", "sa_crop", "mmearth", "sa_crop_upernet", "mmearth_upernet"],
        help="Benchmark Dataset",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["erm", "dbr", "bolp", "groupdro"],
        help="Debiasing Method",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="linear",
        choices=["linear", "upernet"],
        help="Segmentation Head (linear probe vs dense upernet)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None, dest="batch_size", help="Batch size")
    parser.add_argument("--epochs", "--max_epochs", "--max-epochs", type=int, default=None, dest="epochs", help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data_dir", "--data-dir", type=str, default=None, dest="data_dir", help="Dataset directory path")
    parser.add_argument("--output_dir", "--out_dir", "--out-dir", type=str, default=None, dest="output_dir", help="Output directory")
    parser.add_argument("--gpu", "--device", type=str, default=None, dest="gpu", help="GPU device ID (e.g. 0)")
    parser.add_argument("--train_samples", "--train-samples", type=int, default=None, dest="train_samples", help="Limit number of training samples for smoke/mock test")

    args, unknown_args = parser.parse_known_args()

    script_path = get_script_path(args.model, args.dataset, args.method, args.head)

    if not script_path.is_file():
        print(f"Error: Target script not found at {script_path}")
        sys.exit(1)

    print("=" * 70)
    print("🛰️  FairRSFM Benchmark Training")
    print(f"  • Backbone : {args.model.upper()}")
    print(f"  • Dataset  : {args.dataset}")
    print(f"  • Method   : {args.method.upper()}")
    print(f"  • Seed     : {args.seed}")
    print(f"  • Script   : {script_path.relative_to(REPO_ROOT)}")
    print("=" * 70)

    # Configure environment
    env = os.environ.copy()
    src_dir = str(REPO_ROOT / "src")
    repo_dir = str(REPO_ROOT)
    upstream_satmae = str(REPO_ROOT / "src/fairrsfm/models/upstream/SatMAE")
    experiments_dir = str(REPO_ROOT / "experiments")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{repo_dir}:{experiments_dir}:{upstream_satmae}:{pythonpath}"

    # Auto-detect local dataset paths if environment variables not explicitly set
    if "GEO_BENCH_DIR" not in env and Path("/data1/aminur/dataset/geobench").is_dir():
        env["GEO_BENCH_DIR"] = "/data1/aminur/dataset/geobench"
    if "MMEARTH_DATA_DIR" not in env and Path("/data1/aminur/om/MMearth20k/hardcoded").is_dir():
        env["MMEARTH_DATA_DIR"] = "/data1/aminur/om/MMearth20k/hardcoded"

    # Auto-detect and configure PROJ database path for geospatial raster libraries
    p_candidates = []
    try:
        import pyproj.datadir
        p_candidates.append(Path(pyproj.datadir.get_data_dir()))
    except Exception:
        pass
    p_candidates.extend([
        Path(sys.prefix) / "share" / "proj",
        Path("/home/aminur/anaconda3/share/proj"),
        Path("/usr/share/proj"),
    ])
    for p_candidate in p_candidates:
        if (p_candidate / "proj.db").is_file():
            env["PROJ_LIB"] = str(p_candidate)
            env["PROJ_DATA"] = str(p_candidate)
            break

    # Detect argument styles used by this specific script
    help_text = detect_supported_args(script_path, env)

    cmd = [sys.executable, str(script_path), "--seed", str(args.seed)]

    # Dynamic translation for batch size
    if args.batch_size is not None:
        if "--batch-size" in help_text:
            cmd.extend(["--batch-size", str(args.batch_size)])
        elif "--batch_size" in help_text:
            cmd.extend(["--batch_size", str(args.batch_size)])

    # Dynamic translation for epochs
    if args.epochs is not None:
        if "--epochs" in help_text:
            cmd.extend(["--epochs", str(args.epochs)])
        elif "--max_epochs" in help_text:
            cmd.extend(["--max_epochs", str(args.epochs)])
        elif "--max-epochs" in help_text:
            cmd.extend(["--max-epochs", str(args.epochs)])

    # Learning rate
    if args.lr is not None and "--lr" in help_text:
        cmd.extend(["--lr", str(args.lr)])

    # Output directory
    if args.output_dir is not None:
        if "--out-dir" in help_text:
            cmd.extend(["--out-dir", str(args.output_dir)])
        elif "--out_dir" in help_text:
            cmd.extend(["--out_dir", str(args.output_dir)])

    # GPU
    if args.gpu is not None and "--gpu" in help_text:
        cmd.extend(["--gpu", str(args.gpu)])

    # Train samples (mock / smoke testing)
    if args.train_samples is not None:
        if "--train-samples" in help_text:
            cmd.extend(["--train-samples", str(args.train_samples)])
        elif "--train_samples" in help_text:
            cmd.extend(["--train_samples", str(args.train_samples)])

    # Pass any extra unknown arguments directly
    if unknown_args:
        cmd.extend(unknown_args)

    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n[FairRSFM] Training interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
