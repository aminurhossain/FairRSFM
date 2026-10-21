#!/usr/bin/env python3
"""
FairRSFM 48-Run Benchmark Orchestrator
======================================
Executes the full 48-run factorial benchmark matrix:
  3 Foundation Models (Prithvi, SatMAE, DOFA)
  x 4 Benchmark Datasets (EuroSAT, BigEarthNet, SA-Crop, MMEarth-20k)
  x 4 Debiasing Paradigms (ERM, DBR, BOLP, GroupDRO)
  = 48 Training Runs

Features:
  • Runs 1 full epoch (or customizable via --epochs) per combination
  • Real-time logging per experiment under runs/benchmark_48_epochs/logs/
  • Extracts and unifies all DOFA-standard performance & fairness metrics
  • Maintains live status in summary.json and summary.md
  • Designed to be detached and monitored inside a tmux session
"""

import sys
import os
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "benchmark_48_epochs"

MODELS = ["prithvi", "satmae", "dofa"]
DATASETS = ["eurosat", "bigearthnet", "sa_crop", "mmearth"]
METHODS = ["erm", "dbr", "bolp", "groupdro"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FairRSFM 48-Run Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs per run")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="Models to run")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="Datasets to run")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, help="Methods to run")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan without executing")
    parser.add_argument("--continue-on-error", action="store_true", default=True, help="Continue next run on failure")
    return parser.parse_args()


def extract_metrics_from_log(log_path: Path) -> dict:
    """Extract standard DOFA-aligned performance & fairness metrics from run log."""
    metrics = {
        "overall_performance": None,
        "worst_group": None,
        "best_group": None,
        "group_mean": None,
        "per": None,
        "gini": None,
        "nfr": None,
        "eq_odds": None,
        "demographic_parity": None,
    }
    if not log_path.exists():
        return metrics

    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        for line in reversed(lines):
            line_str = line.strip()
            # SA-Crop / MMEarth IoU summary
            if "Overall mIoU" in line_str and metrics["overall_performance"] is None:
                for part in line_str.split("|"):
                    if "Overall mIoU:" in part:
                        try:
                            val = float(part.split("Overall mIoU:")[1].replace("%", "").strip())
                            metrics["overall_performance"] = val
                        except Exception:
                            pass
            # EuroSAT / BigEarthNet Macro F1 summary
            elif "Overall Macro F1" in line_str and metrics["overall_performance"] is None:
                for part in line_str.split("|"):
                    if "Overall Macro F1:" in part:
                        try:
                            val = float(part.split("Overall Macro F1:")[1].replace("%", "").strip())
                            metrics["overall_performance"] = val
                        except Exception:
                            pass

            # Worst-case group
            if "Worst-Case Group" in line_str and metrics["worst_group"] is None:
                try:
                    metrics["worst_group"] = float(line_str.split(":")[-1].strip())
                except Exception:
                    pass

            # Best-case group
            if "Best-Case Group" in line_str and metrics["best_group"] is None:
                try:
                    metrics["best_group"] = float(line_str.split(":")[-1].strip())
                except Exception:
                    pass

            # Performance Equity Ratio (PER)
            if "Performance Equity Ratio (PER)" in line_str and metrics["per"] is None:
                try:
                    metrics["per"] = float(line_str.split(":")[-1].strip())
                except Exception:
                    pass

            # Gini Coefficient
            if "Gini Coefficient" in line_str and metrics["gini"] is None:
                try:
                    metrics["gini"] = float(line_str.split(":")[-1].strip())
                except Exception:
                    pass

            # Equalized Odds Disparity
            if "Equalized Odds Disparity" in line_str and metrics["eq_odds"] is None:
                try:
                    metrics["eq_odds"] = float(line_str.split(":")[-1].strip())
                except Exception:
                    pass

            # Demographic Parity
            if ("Demographic Parity" in line_str or "DPM" in line_str) and metrics["demographic_parity"] is None:
                try:
                    metrics["demographic_parity"] = float(line_str.split(":")[-1].strip())
                except Exception:
                    pass

    except Exception as e:
        print(f"Warning parsing metrics from {log_path}: {e}")

    return metrics


def write_markdown_summary(summary_path: Path, results: list, total_count: int):
    """Write human-readable Markdown table summarizing all benchmark runs."""
    lines = [
        "# FairRSFM 48-Run Benchmark Summary (1 Full Epoch)",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Completed Runs:** {len(results)} / {total_count}  ",
        "",
        "| # | Model | Dataset | Method | Status | Duration | Overall Perf | Worst Group | PER | Gini | Eq. Odds | Log |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        m = r["metrics"]
        perf_str = f"{m['overall_performance']:.2f}%" if m.get("overall_performance") is not None else "-"
        worst_str = f"{m['worst_group']:.4f}" if m.get("worst_group") is not None else "-"
        per_str = f"{m['per']:.4f}" if m.get("per") is not None else "-"
        gini_str = f"{m['gini']:.4f}" if m.get("gini") is not None else "-"
        eq_str = f"{m['eq_odds']:.4f}" if m.get("eq_odds") is not None else "-"
        status_icon = "✅ SUCCESS" if r["status"] == "SUCCESS" else "❌ FAILED"
        log_name = Path(r["log_file"]).name

        lines.append(
            f"| {r['index']} | {r['model'].upper()} | {r['dataset']} | {r['method'].upper()} | "
            f"{status_icon} | {r['duration_sec']:.1f}s | {perf_str} | {worst_str} | {per_str} | "
            f"{gini_str} | {eq_str} | `{log_name}` |"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    runs_dir = out_dir / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"

    # Build plan
    plan = []
    for model in args.models:
        for dataset in args.datasets:
            for method in args.methods:
                plan.append((model, dataset, method))

    total_runs = len(plan)
    print("=" * 80)
    print(f"🚀 FairRSFM Benchmark Suite: {total_runs} Sequential Training Runs (Epochs: {args.epochs})")
    print(f"  • Models   : {', '.join(args.models)}")
    print(f"  • Datasets : {', '.join(args.datasets)}")
    print(f"  • Methods  : {', '.join(args.methods)}")
    print(f"  • GPU      : {args.gpu}")
    print(f"  • Logs Dir : {logs_dir}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Plan of Execution:")
        for idx, (m, d, met) in enumerate(plan, 1):
            print(f"  [{idx:02d}/{total_runs}] model={m:<8} dataset={d:<12} method={met:<8}")
        return

    results = []
    success_count = 0
    fail_count = 0

    suite_start = time.time()

    for idx, (model, dataset, method) in enumerate(plan, 1):
        run_name = f"{model}_{dataset}_{method}"
        run_log = logs_dir / f"{run_name}.log"
        run_out = runs_dir / run_name

        print(f"\n{'─'*80}")
        print(f"▶️  [{idx:02d}/{total_runs}] Starting: {model.upper()} | {dataset} | {method.upper()} (1 epoch)")
        print(f"    Log: {run_log}")
        print(f"{'─'*80}")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "train.py"),
            "--model", model,
            "--dataset", dataset,
            "--method", method,
            "--epochs", str(args.epochs),
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--output_dir", str(run_out),
        ]

        run_start = time.time()
        status = "SUCCESS"
        exit_code = 0

        with open(run_log, "w", encoding="utf-8") as lf:
            # Write run metadata header
            lf.write(f"# FairRSFM Benchmark Run {idx}/{total_runs}\n")
            lf.write(f"# Model: {model} | Dataset: {dataset} | Method: {method}\n")
            lf.write(f"# Command: {' '.join(cmd)}\n")
            lf.write(f"# Start Time: {datetime.now().isoformat()}\n\n")
            lf.flush()

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                )
                exit_code = proc.returncode
                if exit_code != 0:
                    status = "FAILED"
            except Exception as e:
                lf.write(f"\nExecution Exception: {e}\n")
                status = "FAILED"
                exit_code = -1

        duration = time.time() - run_start

        if status == "SUCCESS":
            success_count += 1
            print(f"✅ [{idx:02d}/{total_runs}] COMPLETED {run_name} in {duration:.1f}s")
        else:
            fail_count += 1
            print(f"❌ [{idx:02d}/{total_runs}] FAILED {run_name} (exit code: {exit_code}) in {duration:.1f}s")

        # Parse metrics
        extracted = extract_metrics_from_log(run_log)

        run_summary = {
            "index": idx,
            "model": model,
            "dataset": dataset,
            "method": method,
            "status": status,
            "exit_code": exit_code,
            "duration_sec": duration,
            "log_file": str(run_log),
            "metrics": extracted,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(run_summary)

        # Update JSON and Markdown summaries
        summary_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_markdown_summary(summary_md_path, results, total_runs)

    total_time = time.time() - suite_start
    print("\n" + "=" * 80)
    print(f"🏁 Benchmark Suite Completed in {total_time/60:.2f} minutes")
    print(f"  • Total Runs : {total_runs}")
    print(f"  • Successful : {success_count}")
    print(f"  • Failed     : {fail_count}")
    print(f"  • JSON Summary: {summary_json_path}")
    print(f"  • Markdown    : {summary_md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
