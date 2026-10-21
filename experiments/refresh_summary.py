#!/usr/bin/env python3
"""
FairRSFM Benchmark Summary Refresher
====================================
Scans all executed benchmark logs and output JSONs in runs/benchmark_48_epochs/
and updates summary.json and summary.md with complete DOFA metrics.
"""

import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "runs" / "benchmark_48_epochs"
LOGS_DIR = OUT_DIR / "logs"
RUNS_DIR = OUT_DIR / "runs"

MODELS = ["prithvi", "satmae", "dofa"]
DATASETS = ["eurosat", "bigearthnet", "sa_crop", "mmearth"]
METHODS = ["erm", "dbr", "bolp", "groupdro"]


def parse_metrics_from_log_and_json(log_path: Path, run_out: Path = None) -> dict:
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

        # Check for [JSON] Results updated in <path>
        val_json_path = None
        if run_out and (run_out / "val_results.json").is_file():
            val_json_path = run_out / "val_results.json"
        else:
            json_match = re.search(r"\[JSON\]\s+Results updated in\s+([^\s\n]+val_results\.json)", content)
            if json_match:
                cand = Path(json_match.group(1).strip())
                if cand.is_file():
                    val_json_path = cand

        if val_json_path and val_json_path.is_file():
            try:
                data = json.loads(val_json_path.read_text(encoding="utf-8"))
                last = data[-1] if isinstance(data, list) else data
                metrics["overall_performance"] = (
                    last.get("overall_macro_f1")
                    or last.get("overall_mIoU")
                    or last.get("overall_miou")
                    or last.get("overall_f1")
                )
                metrics["worst_group"] = (
                    last.get("worst_case_macro_f1")
                    or last.get("worst_case_mIoU")
                    or last.get("worst_case_miou")
                    or last.get("worst_f1")
                )
                metrics["best_group"] = (
                    last.get("best_case_macro_f1")
                    or last.get("best_case_mIoU")
                    or last.get("best_case_miou")
                    or last.get("best_f1")
                )
                metrics["group_mean"] = (
                    last.get("group_mean_macro_f1")
                    or last.get("group_mean_mIoU")
                    or last.get("group_mean_miou")
                    or last.get("mean_f1")
                )
                metrics["per"] = (
                    last.get("performance_equity_ratio")
                    or last.get("per_ratio")
                    or last.get("per")
                )
                metrics["gini"] = (
                    last.get("gini_coefficient")
                    or last.get("gini")
                )
                metrics["nfr"] = (
                    last.get("normalized_fairness_range")
                    or last.get("nfr")
                )
                metrics["eq_odds"] = (
                    last.get("equalized_odds_disparity")
                    or last.get("equalized_odds")
                )
                metrics["demographic_parity"] = (
                    last.get("demographic_parity_disparity")
                    or last.get("demographic_parity_difference")
                    or last.get("demographic_parity")
                    or last.get("dpm")
                )
                return metrics
            except Exception:
                pass

        # Fallback regex
        lines = content.splitlines()
        for line in reversed(lines):
            l = line.strip()
            if metrics["overall_performance"] is None:
                m1 = re.search(r"Overall Macro F1\s*[:]?\s*([0-9\.]+)", l)
                m2 = re.search(r"Overall mIoU\s*[:]?\s*([0-9\.]+)", l)
                if m1: metrics["overall_performance"] = float(m1.group(1))
                elif m2: metrics["overall_performance"] = float(m2.group(1))

            if metrics["worst_group"] is None:
                mw = re.search(r"Worst-[Cc]ase(?:\s+Group)?(?:\s+Macro F1)?\s*[:]?\s*([0-9\.]+)", l)
                if mw: metrics["worst_group"] = float(mw.group(1))

            if metrics["best_group"] is None:
                mb = re.search(r"Best-[Cc]ase(?:\s+Group)?(?:\s+Macro F1)?\s*[:]?\s*([0-9\.]+)", l)
                if mb: metrics["best_group"] = float(mb.group(1))

            if metrics["per"] is None:
                mp = re.search(r"Performance Equity Ratio[^\d]*([0-9\.]+)", l)
                if mp: metrics["per"] = float(mp.group(1))

            if metrics["gini"] is None:
                mg = re.search(r"Gini Coefficient[^\d]*([0-9\.]+)", l)
                if mg: metrics["gini"] = float(mg.group(1))

            if metrics["eq_odds"] is None:
                me = re.search(r"Equalized Odds Disparity[^\d]*([0-9\.]+)", l)
                if me: metrics["eq_odds"] = float(me.group(1))

            if metrics["demographic_parity"] is None:
                md = re.search(r"(?:Demographic Parity|DPM)[^\d]*([0-9\.]+)", l)
                if md: metrics["demographic_parity"] = float(md.group(1))

    except Exception:
        pass

    return metrics


def refresh():
    plan = []
    for model in MODELS:
        for dataset in DATASETS:
            for method in METHODS:
                plan.append((model, dataset, method))

    total_count = len(plan)
    results = []

    for idx, (model, dataset, method) in enumerate(plan, 1):
        run_name = f"{model}_{dataset}_{method}"
        log_file = LOGS_DIR / f"{run_name}.log"
        run_out = RUNS_DIR / run_name

        if not log_file.exists():
            continue

        content = log_file.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            continue

        # Check status
        if "[Done] Training complete" in content or "Best Macro F1 Score" in content or "Best Worst F1 Score" in content or "Best mIoU Checkpoint" in content or "val_results.json" in content:
            status = "SUCCESS"
            exit_code = 0
        elif "Execution Exception" in content or "Error:" in content or "Traceback (most recent call last):" in content:
            status = "FAILED"
            exit_code = 1
        elif "Epoch 0:" in content:
            status = "IN_PROGRESS"
            exit_code = None
        else:
            status = "RUNNING"
            exit_code = None

        metrics = parse_metrics_from_log_and_json(log_file, run_out)

        # Estimate duration if available in summary.json
        duration_sec = 0.0
        results.append({
            "index": idx,
            "model": model,
            "dataset": dataset,
            "method": method,
            "status": status,
            "exit_code": exit_code,
            "duration_sec": duration_sec,
            "log_file": str(log_file),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        })

    # Save summary.json
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Generate Markdown summary
    completed = [r for r in results if r["status"] == "SUCCESS"]
    in_prog = [r for r in results if r["status"] in ("IN_PROGRESS", "RUNNING")]

    lines = [
        "# FairRSFM 48-Run Benchmark Summary (1 Full Epoch)",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Completed Runs:** {len(completed)} / {total_count}  ",
        f"**Currently Running:** {len(in_prog)}  ",
        "",
        "| # | Model | Dataset | Method | Status | Overall Perf (F1/mIoU) | Worst Group | PER | Gini | Eq. Odds | Log |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        m = r["metrics"]
        perf_val = m.get("overall_performance")
        if perf_val is not None:
            perf_str = f"**{perf_val:.4f}**" if perf_val <= 1.0 else f"**{perf_val:.2f}%**"
        else:
            perf_str = "-"
        worst_str = f"{m['worst_group']:.4f}" if m.get("worst_group") is not None else "-"
        per_str = f"{m['per']:.4f}" if m.get("per") is not None else "-"
        gini_str = f"{m['gini']:.4f}" if m.get("gini") is not None else "-"
        eq_str = f"{m['eq_odds']:.4f}" if m.get("eq_odds") is not None else "-"

        if r["status"] == "SUCCESS":
            status_icon = "✅ SUCCESS"
        elif r["status"] in ("IN_PROGRESS", "RUNNING"):
            status_icon = "🔄 RUNNING"
        else:
            status_icon = "❌ FAILED"

        log_name = Path(r["log_file"]).name

        lines.append(
            f"| {r['index']} | {r['model'].upper()} | {r['dataset']} | {r['method'].upper()} | "
            f"{status_icon} | {perf_str} | {worst_str} | {per_str} | "
            f"{gini_str} | {eq_str} | `{log_name}` |"
        )

    md_content = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(md_content, encoding="utf-8")
    (OUT_DIR / "benchmark_summary.md").write_text(md_content, encoding="utf-8")
    (REPO_ROOT / "BENCHMARK_RESULTS.md").write_text(md_content, encoding="utf-8")
    print(f"Refreshed summary: {len(completed)} completed runs recorded.")


if __name__ == "__main__":
    refresh()
