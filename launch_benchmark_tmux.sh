#!/bin/bash
# ==============================================================================
# FairRSFM 48-Run Benchmark Launcher (tmux)
# ==============================================================================

SESSION_NAME="fairrsfm_48"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$REPO_DIR" != "/" ]; do
    if [ -f "$REPO_DIR/pyproject.toml" ]; then
        break
    fi
    REPO_DIR="$(dirname "$REPO_DIR")"
done

mkdir -p "$REPO_DIR/runs/benchmark_48_epochs/logs"

# Check if session already exists
tmux has-session -t "$SESSION_NAME" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "⚠️  Tmux session '$SESSION_NAME' is already running!"
    echo "   To view live progress: tmux attach -t $SESSION_NAME"
    echo "   To kill session:       tmux kill-session -t $SESSION_NAME"
    exit 0
fi

echo "=============================================================================="
echo "🚀 Launching FairRSFM 48-Run Benchmark in tmux session: '$SESSION_NAME'"
echo "   Working Directory : $REPO_DIR"
echo "   Epochs per run    : 1 full epoch"
echo "   Output Logs       : runs/benchmark_48_epochs/logs/"
echo "   Live Summary      : runs/benchmark_48_epochs/summary.md"
echo "=============================================================================="

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_DIR" \
    "bash $REPO_DIR/experiments/run_tmux_worker.sh; echo 'Press Enter to exit'; read"

echo ""
echo "✅ Benchmark suite started successfully in background tmux session!"
echo ""
echo "Helpful tmux commands:"
echo "  • Attach to live training session : tmux attach -t $SESSION_NAME"
echo "  • Detach safely from session      : Press Ctrl+B then release and press D"
echo "  • View live Markdown table        : cat runs/benchmark_48_epochs/summary.md"
echo "  • View live orchestrator log      : tail -f runs/benchmark_48_epochs/orchestrator.log"
echo "  • Stop/kill the session           : tmux kill-session -t $SESSION_NAME"
echo "=============================================================================="
