#!/usr/bin/env bash
###############################################################################
# run_airsim_vla.slurm — Slurm batch job for AirSim VLA Milestone 2 benchmark.
#
# Runs the VLA-guided drone navigation benchmark using OpenVLA-7B as a
# directional oracle. Requires A100 GPU (40GB) for model inference.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_airsim_vla.slurm
#   cd ~/vla-proj && sbatch --export=TASKS="1 3" scripts/run_airsim_vla.slurm
#
# Environment variables:
#   TASKS          — space-separated task IDs (default: "1 2 3 4 5")
#   MODEL_PATH     — path to OpenVLA weights (default: ~/models/openvla-7b)
#   WAYPOINT_SCALE — VLA delta scaling factor (default: 15.0)
#   SKIP_AIRSIM    — set to "1" if AirSim is already running externally
#   RECORD_FRAMES  — "1" to record camera frames (default: 1)
#
# Output: logs go to ~/vla-proj/logs/vla_<jobid>.out
###############################################################################

#SBATCH --job-name=vla-m2
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/vla_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/vla_%j.out

set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
TASKS="${TASKS:-1 2 3 4 5}"
CONTROLLER="${CONTROLLER:-vlm}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/openvla-7b}"
VLM_MODEL="${VLM_MODEL:-OpenGVLab/InternVL2-8B}"
WAYPOINT_SCALE="${WAYPOINT_SCALE:-15.0}"
SKIP_AIRSIM="${SKIP_AIRSIM:-1}"
RECORD_FRAMES="${RECORD_FRAMES:-1}"
RECORD_FPS="${RECORD_FPS:-5}"
AIRSIM_BINARY="${AIRSIM_BINARY:-$HOME/AirSim/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh}"
AIRSIM_SETTINGS_SRC="$PROJ_DIR/airsim_benchmark/config/settings.json"
AIRSIM_SETTINGS_DST="$HOME/Documents/AirSim/settings.json"

OUTPUT_DIR="$PROJ_DIR/logs/airsim_output/$CONTROLLER"
TRAJ_DIR="$OUTPUT_DIR/trajectories"
PLOT_DIR="$OUTPUT_DIR/plots"
CONFIG="$PROJ_DIR/airsim_benchmark/config/benchmark_config.yaml"

mkdir -p "$PROJ_DIR/logs" "$OUTPUT_DIR" "$TRAJ_DIR" "$PLOT_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────

echo "============================================"
echo " Navigation Benchmark — Milestone 2"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo " Controller: $CONTROLLER"
if [[ "$CONTROLLER" == "vlm" ]]; then
    echo " VLM:        $VLM_MODEL"
else
    echo " Model:      $MODEL_PATH"
fi
echo " Scale:      $WAYPOINT_SCALE"
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
    pkill -9 -f "AirSimNH" 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT

# ── Step 1: Install Python dependencies ──────────────────────────────────────

echo ""
echo "[1/6] Installing Python dependencies..."
# Upgrade pip first to avoid old resolver bugs (use python3 -m pip afterward)
python3 -m pip install --quiet --user --upgrade pip 2>&1 | tail -1
PIP="python3 -m pip"

# Filter function to suppress Ubuntu's non-PEP440 version parsing noise
_pip_filter() { grep -v "Error parsing dependencies of" || true; }

# Core deps (prefer binary wheels to avoid source-build failures)
$PIP install --quiet --user --prefer-binary \
    'numpy>=1.24.0' 'msgpack-rpc-python' \
    2>&1 | _pip_filter | tail -3
$PIP install --quiet --user --prefer-binary \
    'airsim>=1.8.1' \
    'matplotlib>=3.7.0' \
    'pyyaml>=6.0' \
    'opencv-python>=4.8.0' \
    2>&1 | _pip_filter | tail -3

# Detect CUDA driver version to pick correct PyTorch wheel
CUDA_DRIVER_VER=$(python3 -c "
import ctypes, ctypes.util
try:
    lib = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
    v = ctypes.c_int()
    lib.cuDriverGetVersion(ctypes.byref(v))
    print(v.value)
except:
    print(0)
" 2>/dev/null)
echo "      CUDA driver version code: $CUDA_DRIVER_VER"

if [[ "$CUDA_DRIVER_VER" -ge 12010 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    TORCH_PKG="torch torchvision"
    echo "      Installing PyTorch 2.x (CUDA 12.1 wheels)..."
elif [[ "$CUDA_DRIVER_VER" -ge 11030 ]]; then
    # Driver 470+ (CUDA 11.3+) supports cu118 runtime via forward compat
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_PKG="torch torchvision"
    echo "      Installing PyTorch 2.x (CUDA 11.8 wheels — driver CUDA $((CUDA_DRIVER_VER/1000)).$((CUDA_DRIVER_VER%1000/10)))..."
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_PKG="torch torchvision"
    echo "      WARNING: No compatible CUDA driver found (code=$CUDA_DRIVER_VER). Installing CPU-only PyTorch (inference will be SLOW)..."
fi

$PIP install --quiet --user \
    $TORCH_PKG \
    --index-url "$TORCH_INDEX" \
    2>&1 | _pip_filter | tail -5

# Transformers + VLM deps
echo "      Installing transformers + VLM deps..."
$PIP install --quiet --user --prefer-binary \
    'transformers>=4.40.0' \
    'tokenizers>=0.19.0' \
    'timm>=0.9.10,<1.0.0' \
    'Pillow>=10.0.0' \
    'accelerate>=0.25.0' \
    'sentencepiece' \
    'huggingface_hub>=0.20.0' \
    'einops' \
    2>&1 | _pip_filter | tail -5

# Verify torch is importable with GPU
python3 -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'      PyTorch {torch.__version__}, CUDA available: {cuda_ok}')
if cuda_ok:
    print(f'      GPU: {torch.cuda.get_device_name(0)}')
else:
    print('      WARNING: CUDA not available — VLA inference will be VERY slow on CPU')
" || { echo "      ERROR: PyTorch installation failed!"; exit 1; }

python3 -c "import transformers; print(f'      Transformers {transformers.__version__}')" \
    || { echo "      ERROR: transformers installation failed!"; exit 1; }
echo "      Dependencies installed."

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

# ── Step 2: Check model weights ──────────────────────────────────────────────

echo ""
echo "[2/6] Checking model weights..."
if [[ "$CONTROLLER" == "vlm" ]]; then
    # VLM models are downloaded by HuggingFace's cache on first load.
    # Pre-download here to avoid timeout during benchmark.
    echo "      VLM model: $VLM_MODEL"
    python3 -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('$VLM_MODEL', resume_download=True)
print(f'      Model cached at: {path}')
" || {
        echo "      WARNING: VLM model pre-download failed — will retry at inference time"
    }
elif [[ -d "$MODEL_PATH" ]] && [[ -f "$MODEL_PATH/config.json" ]]; then
    echo "      VLA model found at $MODEL_PATH"
else
    echo "      VLA model not found at $MODEL_PATH — downloading..."
    python3 -m airsim_benchmark.scripts.download_model \
        --model-id openvla/openvla-7b \
        --output "$MODEL_PATH"
    if [[ $? -ne 0 ]]; then
        echo "      ERROR: Model download failed."
        exit 1
    fi
fi

# ── Step 3: Deploy AirSim settings.json ──────────────────────────────────────

echo ""
echo "[3/6] Deploying AirSim settings.json..."
mkdir -p "$(dirname "$AIRSIM_SETTINGS_DST")"
cp "$AIRSIM_SETTINGS_SRC" "$AIRSIM_SETTINGS_DST"
echo "      → $AIRSIM_SETTINGS_DST"

# ── Step 4: Start AirSim (headless) ──────────────────────────────────────────

# Suppress audio/VR noise regardless of whether we launch AirSim here
export SDL_AUDIODRIVER=dummy
# Disable OpenXR by pointing to a minimal valid-but-empty manifest
_XR_DUMMY="$PROJ_DIR/logs/.openxr_stub.json"
echo '{"file_format_version":"1.0.0","runtime":{"library_path":"/dev/null"}}' > "$_XR_DUMMY"
export XR_RUNTIME_JSON="$_XR_DUMMY"

if [[ "$SKIP_AIRSIM" == "1" ]]; then
    echo ""
    echo "[4/6] Skipping AirSim launch (SKIP_AIRSIM=1)"
else
    echo ""
    echo "[4/6] Starting AirSim Neighborhood (headless)..."

    if [[ ! -f "$AIRSIM_BINARY" ]]; then
        echo "      ERROR: AirSim binary not found at $AIRSIM_BINARY"
        exit 1
    fi

    Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
    XVFB_PID=$!
    export DISPLAY=:99

    sleep 2

    "$AIRSIM_BINARY" -RenderOffScreen -WINDOWED -nosound &
    AIRSIM_PID=$!
    echo "      AirSim PID: $AIRSIM_PID — waiting 30s..."
    sleep 30

    if ! kill -0 "$AIRSIM_PID" 2>/dev/null; then
        echo "      ERROR: AirSim process died during startup."
        exit 1
    fi
    echo "      AirSim running."
fi

# ── Step 5: Run VLA benchmark ─────────────────────────────────────────────────

echo ""
echo "[5/6] Running $CONTROLLER benchmark (tasks=$TASKS)..."
echo "============================================"

TASK_ARGS=""
if [[ "$TASKS" != "1 2 3 4 5" ]]; then
    TASK_ARGS="--tasks $TASKS"
fi

RECORD_ARGS=""
if [[ "$RECORD_FRAMES" == "1" ]]; then
    RECORD_ARGS="--record --record-fps $RECORD_FPS"
fi

CONTROLLER_ARGS="--controller $CONTROLLER"
if [[ "$CONTROLLER" == "vlm" ]]; then
    CONTROLLER_ARGS="$CONTROLLER_ARGS --vlm-model $VLM_MODEL"
elif [[ "$CONTROLLER" == "vla" ]]; then
    CONTROLLER_ARGS="$CONTROLLER_ARGS --model-path $MODEL_PATH --waypoint-scale $WAYPOINT_SCALE"
fi

python3 -m airsim_benchmark.scripts.run_benchmark \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    $CONTROLLER_ARGS \
    $TASK_ARGS \
    $RECORD_ARGS \
    --verbose
BENCHMARK_EXIT=$?

echo ""
if [[ $BENCHMARK_EXIT -eq 0 ]]; then
    echo "  Benchmark ($CONTROLLER): ALL TASKS PASSED"
else
    echo "  Benchmark ($CONTROLLER): SOME TASKS FAILED (exit code $BENCHMARK_EXIT)"
fi

# ── Step 6: Generate plots ────────────────────────────────────────────────────

echo ""
echo "[6/6] Generating flight-path plots..."

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
if [[ "$CONTROLLER" == "vlm" ]]; then
    echo " VLM:        $VLM_MODEL"
else
    echo " Model:      $MODEL_PATH"
fi
echo " Tasks:      $TASKS"
echo " Output:     $OUTPUT_DIR"
echo " Metrics:    $OUTPUT_DIR/metrics.json"
echo " Plots:      $PLOT_DIR/"
echo " End:        $(date)"
echo "============================================"

exit $BENCHMARK_EXIT
