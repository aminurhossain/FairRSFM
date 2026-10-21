#!/bin/bash
set -e

# Resolve repository root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p "$REPO_DIR/runs/benchmark_48_epochs/logs"

# Ensure environment variables are loaded
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:$PYTHONPATH"
export PROJ_DATA="/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
export PROJ_LIB="/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
export GEO_BENCH_DIR="/data1/aminur/dataset/geobench"
export MMEARTH_DATA_DIR="/data1/aminur/om/MMearth20k/hardcoded"

echo "=============================================================================="
echo "🚀 Starting FairRSFM 48-Run Benchmark (1 Full Epoch per Run)"
echo "   Working Directory : $REPO_DIR"
echo "   Start Time        : $(date)"
echo "=============================================================================="

/home/aminur/anaconda3/bin/python3 -u "$REPO_DIR/experiments/run_48_benchmarks.py" --epochs 1 --gpu 0 2>&1 | tee "$REPO_DIR/runs/benchmark_48_epochs/orchestrator.log"

echo "=============================================================================="
echo "🏁 All 48 benchmark runs completed at $(date)"
echo "=============================================================================="
