#!/usr/bin/env bash
###############################################################################
# run_airsim_interactive.sh — Run the AirSim benchmark interactively.
#
# Use this AFTER you've already allocated a node (salloc/srun --pty) and
# AirSim is running. Handles pip install + benchmark + plots in one shot.
#
# Usage (from an interactive node session):
#   bash scripts/run_airsim_interactive.sh
#   bash scripts/run_airsim_interactive.sh --tasks "1 3"
#   bash scripts/run_airsim_interactive.sh --skip-install
#
# Options:
#   --tasks "1 2 3 4 5"  — override which tasks to run
#   --skip-install       — skip pip install (already done this session)
#   --controller NAME    — controller type (default: classical)
###############################################################################

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TASKS="1 2 3 4 5"
CONTROLLER="classical"
SKIP_INSTALL=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"; shift 2 ;;
        --controller)
            CONTROLLER="$2"; shift 2 ;;
        --skip-install)
            SKIP_INSTALL=1; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="$PROJ_DIR/logs/airsim_output/${CONTROLLER}"
CONFIG="$PROJ_DIR/airsim_benchmark/config/benchmark_config.yaml"

echo "============================================"
echo " AirSim Benchmark — Interactive Mode"
echo " Controller: $CONTROLLER"
echo " Tasks:      $TASKS"
echo " Output:     $OUTPUT_DIR"
echo "============================================"

# ── Install dependencies (into user site-packages) ────────────────────────────

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    echo ""
    echo "[1/4] Installing dependencies..."
    # numpy and msgpack-rpc-python must be installed first —
    # airsim's setup.py imports them at install time
    pip install --quiet --user 'numpy>=1.24.0' 'msgpack-rpc-python'
    pip install --quiet --user \
        'airsim>=1.8.1' \
        'matplotlib>=3.7.0' \
        'pyyaml>=6.0' \
        'opencv-python>=4.8.0'
    echo "      Done."
else
    echo ""
    echo "[1/4] Skipping install (--skip-install)"
fi

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

# ── Deploy settings.json ──────────────────────────────────────────────────────

echo ""
echo "[2/4] Deploying settings.json..."
mkdir -p "$HOME/Documents/AirSim"
cp "$PROJ_DIR/airsim_benchmark/config/settings.json" "$HOME/Documents/AirSim/settings.json"
echo "      → ~/Documents/AirSim/settings.json"

# ── Run benchmark ─────────────────────────────────────────────────────────────

echo ""
echo "[3/4] Running benchmark..."
mkdir -p "$OUTPUT_DIR"

TASK_ARGS=""
if [[ "$TASKS" != "1 2 3 4 5" ]]; then
    TASK_ARGS="--tasks $TASKS"
fi

python3 -m airsim_benchmark.scripts.run_benchmark \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    --controller "$CONTROLLER" \
    $TASK_ARGS \
    --verbose

# ── Generate plots ────────────────────────────────────────────────────────────

TRAJ_DIR="$OUTPUT_DIR/trajectories"
PLOT_DIR="$OUTPUT_DIR/plots"

echo ""
echo "[4/4] Generating plots..."
if ls "$TRAJ_DIR"/task_*_trajectory.csv 1>/dev/null 2>&1; then
    python3 -m airsim_benchmark.scripts.plot_flights \
        "$TRAJ_DIR" \
        --output "$PLOT_DIR" \
        --config "$CONFIG"
    echo "      Plots: $PLOT_DIR/"
else
    echo "      No trajectories found — skipping"
fi

echo ""
echo "============================================"
echo " Done. Results: $OUTPUT_DIR/"
echo "============================================"
