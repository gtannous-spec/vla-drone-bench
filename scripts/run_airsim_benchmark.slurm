#!/usr/bin/env bash
###############################################################################
# run_airsim_benchmark.slurm — Slurm batch job for AirSim Milestone 1 benchmark.
#
# Runs the 5-task classical waypoint benchmark on AirSim Neighborhood env.
# Handles: dependency installation, AirSim launch (headless), benchmark
# execution, metrics computation, and flight-path plot generation.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_airsim_benchmark.slurm
#   cd ~/vla-proj && sbatch --export=TASKS="1 3 5" scripts/run_airsim_benchmark.slurm
#
# Environment variables (optional overrides via --export):
#   TASKS          — space-separated task IDs (default: "1 2 3 4 5")
#   CONTROLLER     — controller type (default: "classical")
#   AIRSIM_BINARY  — path to AirSim binary (default: auto-detect)
#   SKIP_AIRSIM    — set to "1" if AirSim is already running externally
#
# Output: logs go to ~/vla-proj/logs/airsim_<jobid>.out
###############################################################################

#SBATCH --job-name=vla-airsim
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/airsim_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/airsim_%j.out

set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
TASKS="${TASKS:-1 2 3 4 5}"
CONTROLLER="${CONTROLLER:-classical}"
SKIP_AIRSIM="${SKIP_AIRSIM:-1}"
AIRSIM_BINARY="${AIRSIM_BINARY:-$HOME/AirSim/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh}"
RECORD_FRAMES="${RECORD_FRAMES:-1}"
RECORD_FPS="${RECORD_FPS:-5}"
AIRSIM_SETTINGS_SRC="$PROJ_DIR/airsim_benchmark/config/settings.json"
AIRSIM_SETTINGS_DST="$HOME/Documents/AirSim/settings.json"

OUTPUT_DIR="$PROJ_DIR/logs/airsim_output/${CONTROLLER}"
TRAJ_DIR="$OUTPUT_DIR/trajectories"
PLOT_DIR="$OUTPUT_DIR/plots"
CONFIG="$PROJ_DIR/airsim_benchmark/config/benchmark_config.yaml"

mkdir -p "$PROJ_DIR/logs" "$OUTPUT_DIR" "$TRAJ_DIR" "$PLOT_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────

echo "============================================"
echo " VLA Navigation Benchmark — AirSim Milestone 1"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Controller: $CONTROLLER"
echo " Tasks:      $TASKS"
echo " Output:     $OUTPUT_DIR"
echo " Start:      $(date)"
echo "============================================"

# ── Cleanup trap ──────────────────────────────────────────────────────────────

AIRSIM_PID=""
XVFB_PID=""

cleanup() {
    echo ""
    echo "[cleanup] Stopping processes..."
    [[ -n "$AIRSIM_PID" ]] && kill "$AIRSIM_PID" 2>/dev/null || true
    [[ -n "$XVFB_PID" ]] && kill "$XVFB_PID" 2>/dev/null || true
    sleep 2
    [[ -n "$AIRSIM_PID" ]] && kill -9 "$AIRSIM_PID" 2>/dev/null || true
    [[ -n "$XVFB_PID" ]] && kill -9 "$XVFB_PID" 2>/dev/null || true
    pkill -9 -f "Neighborhood" 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT

# ── Step 1: Install Python dependencies ──────────────────────────────────────

echo ""
echo "[1/5] Installing Python dependencies..."
# numpy and msgpack-rpc-python must be installed first —
# airsim's setup.py imports numpy and msgpackrpc at install time
pip install --quiet --user 'numpy>=1.24.0' 'msgpack-rpc-python' 2>&1 | tail -3
pip install --quiet --user \
    'airsim>=1.8.1' \
    'matplotlib>=3.7.0' \
    'pyyaml>=6.0' \
    'opencv-python>=4.8.0' \
    2>&1 | tail -5
echo "      Dependencies installed."

# Ensure user site-packages is on PATH
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

# ── Step 2: Deploy AirSim settings.json ──────────────────────────────────────

echo ""
echo "[2/5] Deploying AirSim settings.json..."
mkdir -p "$(dirname "$AIRSIM_SETTINGS_DST")"
cp "$AIRSIM_SETTINGS_SRC" "$AIRSIM_SETTINGS_DST"
echo "      → $AIRSIM_SETTINGS_DST"

# ── Step 3: Start AirSim (headless) ──────────────────────────────────────────

if [[ "$SKIP_AIRSIM" == "1" ]]; then
    echo ""
    echo "[3/5] Skipping AirSim launch (SKIP_AIRSIM=1, assuming already running)"
else
    echo ""
    echo "[3/5] Starting AirSim Neighborhood (headless)..."

    if [[ ! -f "$AIRSIM_BINARY" ]]; then
        echo "      ERROR: AirSim binary not found at $AIRSIM_BINARY"
        echo "      Set AIRSIM_BINARY or SKIP_AIRSIM=1 if running externally."
        exit 1
    fi

    # Virtual framebuffer for headless GPU rendering
    Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
    XVFB_PID=$!
    export DISPLAY=:99
    sleep 2

    # Launch AirSim with Vulkan rendering (headless)
    "$AIRSIM_BINARY" -RenderOffScreen -WINDOWED &
    AIRSIM_PID=$!
    echo "      AirSim PID: $AIRSIM_PID"
    echo "      Waiting for AirSim to initialize (30s)..."
    sleep 30

    if ! kill -0 "$AIRSIM_PID" 2>/dev/null; then
        echo "      ERROR: AirSim process died during startup."
        exit 1
    fi
    echo "      AirSim running."
fi

# ── Step 4: Run the benchmark ─────────────────────────────────────────────────

echo ""
echo "[4/5] Running benchmark (controller=$CONTROLLER, tasks=$TASKS)..."
echo "============================================"

# Build task arguments
TASK_ARGS=""
if [[ "$TASKS" != "1 2 3 4 5" ]]; then
    TASK_ARGS="--tasks $TASKS"
fi

RECORD_ARGS=""
if [[ "$RECORD_FRAMES" == "1" ]]; then
    RECORD_ARGS="--record --record-fps $RECORD_FPS"
fi

python3 -m airsim_benchmark.scripts.run_benchmark \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    --controller "$CONTROLLER" \
    $TASK_ARGS \
    $RECORD_ARGS \
    --verbose
BENCHMARK_EXIT=$?

echo ""
if [[ $BENCHMARK_EXIT -eq 0 ]]; then
    echo "  Benchmark: ALL TASKS PASSED"
else
    echo "  Benchmark: SOME TASKS FAILED (exit code $BENCHMARK_EXIT)"
fi

# ── Step 5: Generate plots ────────────────────────────────────────────────────

echo ""
echo "[5/5] Generating flight-path plots..."

if ls "$TRAJ_DIR"/task_*_trajectory.csv 1>/dev/null 2>&1; then
    python3 -m airsim_benchmark.scripts.plot_flights \
        "$TRAJ_DIR" \
        --output "$PLOT_DIR" \
        --config "$CONFIG" \
        && echo "      Plots saved to $PLOT_DIR" \
        || echo "      WARNING: Plot generation failed"
else
    echo "      No trajectory CSVs found — skipping plots"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo " BENCHMARK COMPLETE"
echo " Controller: $CONTROLLER"
echo " Tasks:      $TASKS"
echo " Output:     $OUTPUT_DIR"
echo " Metrics:    $OUTPUT_DIR/metrics.json"
echo " Plots:      $PLOT_DIR/"
echo " End:        $(date)"
echo "============================================"

exit $BENCHMARK_EXIT
