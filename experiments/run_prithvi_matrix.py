#!/usr/bin/env python3
"""Run an isolated Prithvi dataset/method/seed matrix with bounded concurrency."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("eurosat", "bigearthnet", "sa_crop", "mmearth")
METHODS = ("erm", "dbr", "bolp", "groupdro")


def training_script(dataset: str, method: str) -> Path:
    family = "classification" if dataset in ("eurosat", "bigearthnet") else "segmentation"
    dataset_dir = "mmearth_20k" if dataset == "mmearth" else dataset
    return ROOT / "experiments" / "scripts" / "prithvi" / family / dataset_dir / f"{dataset}_{method}.py"


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_job(job: dict, gpu: str) -> tuple[str, int]:
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(ROOT / "experiments" / "train.py"),
        "--model", "prithvi", "--dataset", job["dataset"],
        "--method", job["method"], "--seed", str(job["seed"]),
        "--epochs", str(job["epochs"]), "--gpu", gpu,
        "--output_dir", str(out_dir),
    ]
    env = os.environ.copy()
    if job.get("smoke_test"):
        command.append("--smoke-test")
    env["PYTHONUNBUFFERED"] = "1"
    (out_dir / "command.json").write_text(
        json.dumps({"argv": command, "gpu": gpu, "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n"
    )
    with (out_dir / "launcher.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out_dir / "status.json").write_text(
        json.dumps({"returncode": completed.returncode, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n"
    )
    return str(out_dir), completed.returncode


def run_worker(worker_index: int, gpu: str, jobs: list[dict], worker_count: int) -> list[str]:
    failures = []
    for job in jobs[worker_index::worker_count]:
        try:
            out_dir, code = run_job(job, gpu)
        except Exception as exc:
            out_dir, code = job["output_dir"], 1
            print(f"FAILED {out_dir}: {exc}", flush=True)
        print(f"{'DONE' if code == 0 else 'FAILED'} {out_dir} (exit {code})", flush=True)
        if code:
            failures.append(out_dir)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["erm", "bolp"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", nargs="+", required=True,
                        help="One worker per entry; repeat an ID only when sharing GPU memory is safe")
    parser.add_argument("--run-id", required=True, help="Unique name under runs/prithvi_matrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Bounded real-model checks; not benchmark results")
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs must be positive; methods and seeds must be unique")
    if not args.run_id.replace("-", "").replace("_", "").isalnum() or args.run_id in (".", ".."):
        parser.error("run-id must contain only letters, digits, hyphens, or underscores")

    run_root = ROOT / "runs" / "prithvi_matrix" / args.run_id
    jobs = [
        {"dataset": dataset, "method": method, "seed": seed, "epochs": args.epochs,
         "smoke_test": args.smoke_test,
         "output_dir": str(run_root / dataset / method / f"seed_{seed}")}
        for seed in args.seeds for dataset in args.datasets for method in args.methods
    ]
    if run_root.exists():
        parser.error(f"run directory already exists: {run_root}")
    sources = [Path(__file__), ROOT / "experiments" / "train.py"] + [
        training_script(dataset, method) for dataset in args.datasets for method in args.methods
    ]
    if args.smoke_test:
        sources.append(ROOT / "experiments" / "smoke_prithvi.py")
    missing_sources = [str(path) for path in sources if not path.is_file()]
    if missing_sources:
        parser.error(f"training source is missing: {missing_sources}")
    print(f"{len(jobs)} jobs; {len(args.gpus)} simultaneous worker(s); output: {run_root}")
    for index, job in enumerate(jobs):
        print(f"GPU {args.gpus[index % len(args.gpus)]}: {job['dataset']} {job['method']} seed {job['seed']}")
    if args.dry_run:
        return 0

    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "methods": args.methods, "datasets": args.datasets,
        "seeds": args.seeds, "epochs": args.epochs, "gpus": args.gpus,
        "smoke_test": args.smoke_test,
        "jobs": jobs,
    }, indent=2) + "\n")
    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_worker, index, gpu, jobs, len(args.gpus))
                   for index, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            failures.extend(future.result())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
