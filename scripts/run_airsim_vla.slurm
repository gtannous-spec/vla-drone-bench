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
echo "      Python: $(python3 --version 2>&1)"

# Python 3.12+ enforces PEP 668 (externally managed environment).
# Allow --user installs on HPC nodes where we don't control the system Python.
export PIP_BREAK_SYSTEM_PACKAGES=1

# Bootstrap pip if not available (dlc2 nodes lack pip and ensurepip)
if ! python3 -m pip --version &>/dev/null; then
    echo "      pip not found — bootstrapping via get-pip.py..."
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user --quiet 2>&1 | tail -3
    export PATH="$HOME/.local/bin:$PATH"
fi
python3 -m pip install --quiet --user --upgrade pip 2>&1 | tail -1
PIP="python3 -m pip"

# Filter function to suppress Ubuntu's non-PEP440 version parsing noise
_pip_filter() { grep -v "Error parsing dependencies of" || true; }

# Core deps (prefer binary wheels to avoid source-build failures)
$PIP install --quiet --user --prefer-binary \
    'numpy>=1.24.0' \
    'matplotlib>=3.7.0' \
    'pyyaml>=6.0' \
    'opencv-python>=4.8.0' \
    'msgpack' \
    'Pillow>=10.0.0' \
    2>&1 | _pip_filter | tail -3

# airsim + msgpack-rpc-python have build deps incompatible with Python 3.12:
#   - airsim's setup.py imports itself during build → fails
#   - msgpack-rpc-python requires tornado<5 which uses collections.MutableMapping (removed in 3.10)
# Strategy: install runtime deps, download source tarballs directly from PyPI
# (bypassing pip's build system), extract the pure-Python packages, and copy
# them into user site-packages.
$PIP install --quiet --user 'setuptools' 'tornado>=6.0' 'msgpack>=1.0' 'nest_asyncio' \
    2>&1 | _pip_filter | tail -1

_AIRSIM_TMP=$(mktemp -d)
_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
mkdir -p "$_SITE"
echo "      Installing airsim (Python 3.12 direct-extract workaround)..."

# Download source tarballs directly from PyPI (no build step)
_download_pypi_sdist() {
    local pkg_name="$1" dest_dir="$2"
    local url
    url=$(python3 -c "
import urllib.request, json
data = json.loads(urllib.request.urlopen('https://pypi.org/pypi/$pkg_name/json').read())
sdists = [u['url'] for u in data['urls'] if u['packagetype'] == 'sdist']
if sdists: print(sdists[0])
else: print('')
" 2>/dev/null)
    if [[ -n "$url" ]]; then
        curl -sSL "$url" -o "$dest_dir/$pkg_name.tar.gz" 2>/dev/null
        return $?
    fi
    return 1
}

_download_pypi_sdist "airsim" "$_AIRSIM_TMP" \
    && echo "      Downloaded airsim source" \
    || echo "      WARNING: airsim download failed"
_download_pypi_sdist "msgpack-rpc-python" "$_AIRSIM_TMP" \
    && echo "      Downloaded msgpack-rpc-python source" \
    || echo "      WARNING: msgpack-rpc-python download failed"

# Extract tarballs and copy pure-Python package dirs to site-packages
for _tarball in "$_AIRSIM_TMP"/*.tar.gz; do
    [ -f "$_tarball" ] && tar -xzf "$_tarball" -C "$_AIRSIM_TMP" 2>/dev/null
done
_installed=0
for _pkg_dir in "$_AIRSIM_TMP"/*/airsim "$_AIRSIM_TMP"/*/msgpackrpc; do
    if [ -d "$_pkg_dir" ]; then
        cp -r "$_pkg_dir" "$_SITE/"
        echo "      Extracted $(basename "$_pkg_dir") → site-packages"
        _installed=$((_installed + 1))
    fi
done
rm -rf "$_AIRSIM_TMP"

if [[ $_installed -lt 2 ]]; then
    echo "      WARNING: Only $_installed/2 packages extracted, trying pip fallback..."
    $PIP install --quiet --user --no-build-isolation --no-deps 'airsim' \
        2>&1 | _pip_filter | tail -2
fi

# Patch msgpackrpc for Tornado 6+ compatibility (tornado.platform.auto removed,
# IOLoop.instance() deprecated, etc.)
echo "      Patching msgpackrpc for Tornado 6 compatibility..."
python3 << 'PATCH_SCRIPT'
import pathlib, site

sp = pathlib.Path(site.getusersitepackages()) / "msgpackrpc"
if not sp.exists():
    print("      WARNING: msgpackrpc not found in site-packages, skipping patch")
    raise SystemExit(0)

# ── address.py ──────────────────────────────────────────────────────────
# tornado.platform.auto removed in tornado 6; set_close_exec is a no-op
# on Python 3+ (CLOEXEC is set by default).
patches = {
    "address.py": [
        ("from tornado.platform.auto import set_close_exec",
         "def set_close_exec(fd): pass"),
    ],
}

# ── loop.py ─────────────────────────────────────────────────────────────
# IOLoop.instance() → IOLoop.current()  (deprecated in tornado 5)
# PeriodicCallback 3rd arg changed from io_loop to jitter (float) in tornado 6
patches["loop.py"] = [
    ("IOLoop.instance()", "IOLoop.current()"),
    ("ioloop.PeriodicCallback(callback, callback_time, self._ioloop)",
     "ioloop.PeriodicCallback(callback, callback_time)"),
    ("loop or ioloop.IOLoop()", "loop or ioloop.IOLoop.current()"),
    # Remove make_current() — deprecated in tornado 6
    ("self._ioloop.make_current()\n        self._periodic_callback = None",
     "self._periodic_callback = None"),
]

# loop.py also needs nest_asyncio at the top to allow reentrant event loops
# (msgpackrpc calls loop.start() inside callbacks, which is not allowed by asyncio)
loop_path = sp / "loop.py"
if loop_path.exists():
    lt = loop_path.read_text()
    if "nest_asyncio" not in lt:
        lt = "import nest_asyncio\nnest_asyncio.apply()\n\n" + lt
        loop_path.write_text(lt)
        print("      Injected nest_asyncio into loop.py")

# ── server.py ───────────────────────────────────────────────────────────
patches["server.py"] = [
    ("IOLoop.instance()", "IOLoop.current()"),
]

# ── transport/tcp.py ────────────────────────────────────────────────────
# This file has the most breakage:
#   1. IOStream() no longer accepts io_loop kwarg
#   2. TCPServer.__init__() no longer accepts io_loop kwarg
#   3. msgpack.Packer/Unpacker encoding param removed in msgpack 2.x
patches["transport/tcp.py"] = [
    ("from tornado.platform.auto import set_close_exec",
     "def set_close_exec(fd): pass"),
    ("IOLoop.instance()", "IOLoop.current()"),
    # Client: IOStream(sock, io_loop=...) → IOStream(sock)
    ("IOStream(self._address.socket(), io_loop=self._session._loop._ioloop)",
     "IOStream(self._address.socket())"),
    # Server: TCPServer.__init__(self, io_loop=io_loop) → TCPServer.__init__(self)
    ("tcpserver.TCPServer.__init__(self, io_loop=io_loop)",
     "tcpserver.TCPServer.__init__(self)"),
    # Server constructor: remove io_loop parameter
    ("def __init__(self, transport, io_loop=None, encodings=None):",
     "def __init__(self, transport, encodings=None):"),
    # Server instantiation: remove io_loop kwarg
    ("MessagePackServer(self, io_loop=self._server._loop._ioloop, encodings=self._encodings)",
     "MessagePackServer(self, encodings=self._encodings)"),
]

patched = 0
for relpath, replacements in patches.items():
    fpath = sp / relpath
    if not fpath.exists():
        continue
    text = fpath.read_text()
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        fpath.write_text(text)
        patched += 1
        print(f"      Patched {relpath}")

# Handle msgpack 2.x encoding parameter removal (Packer/Unpacker)
tcp_path = sp / "transport" / "tcp.py"
if tcp_path.exists():
    text = tcp_path.read_text()
    if "encoding=encodings" in text:
        # Replace Packer(encoding=...) and Unpacker(encoding=...) with
        # version-compatible alternatives
        text = text.replace(
            "self._packer = msgpack.Packer(encoding=encodings[0], default=lambda x: x.to_msgpack())",
            "try:\n"
            "            self._packer = msgpack.Packer(encoding=encodings[0], default=lambda x: x.to_msgpack())\n"
            "        except TypeError:\n"
            "            self._packer = msgpack.Packer(default=lambda x: x.to_msgpack())"
        )
        text = text.replace(
            "self._unpacker = msgpack.Unpacker(encoding=encodings[1])",
            "try:\n"
            "            self._unpacker = msgpack.Unpacker(encoding=encodings[1])\n"
            "        except TypeError:\n"
            "            self._unpacker = msgpack.Unpacker(raw=False)"
        )
        tcp_path.write_text(text)
        print("      Patched transport/tcp.py (msgpack compat)")

print(f"      {patched} file(s) patched for tornado 6 compat")
PATCH_SCRIPT

# Verify airsim is importable
python3 -c "import airsim; print(f'      airsim {airsim.__version__} OK')" 2>/dev/null \
    || python3 -c "import airsim; print('      airsim OK')" \
    || { echo "      ERROR: airsim installation failed!"; exit 1; }

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
    'transformers>=4.40.0,<5.0' \
    'tokenizers>=0.19.0' \
    'timm>=0.9.10,<1.0.0' \
    'Pillow>=10.0.0' \
    'accelerate>=0.25.0' \
    'sentencepiece' \
    'protobuf>=3.20.0' \
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
    echo "      VLM model: $VLM_MODEL"
    # Check if model is already cached; if so, skip download entirely
    _VLM_CACHED=$(python3 -c "
import pathlib
cache_dir = pathlib.Path.home() / '.cache/huggingface/hub'
model_slug = '$VLM_MODEL'.replace('/', '--')
model_dir = cache_dir / f'models--{model_slug}'
snapshots = list(model_dir.glob('snapshots/*/config.json')) if model_dir.exists() else []
if snapshots:
    print(snapshots[0].parent)
else:
    print('')
" 2>/dev/null)

    if [[ -n "$_VLM_CACHED" ]]; then
        echo "      Model already cached at: $_VLM_CACHED"
    else
        echo "      Model not cached — downloading..."
        python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$VLM_MODEL')
print(f'      Model cached at: {path}')
" || {
            echo "      WARNING: VLM model pre-download failed — will retry at inference time"
        }
    fi
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
