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
echo "      Python: $(python3 --version 2>&1)"

# Python 3.12+ enforces PEP 668 (externally managed environment)
export PIP_BREAK_SYSTEM_PACKAGES=1

# Bootstrap pip if not available (dlc2 nodes may lack pip)
if ! python3 -m pip --version &>/dev/null; then
    echo "      pip not found — bootstrapping via get-pip.py..."
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user --quiet 2>&1 | tail -3
    export PATH="$HOME/.local/bin:$PATH"
fi
python3 -m pip install --quiet --user --upgrade pip 2>&1 | tail -1
PIP="python3 -m pip"
_pip_filter() { grep -v "Error parsing dependencies of" || true; }

$PIP install --quiet --user --prefer-binary \
    'numpy>=1.24.0' \
    'matplotlib>=3.7.0' \
    'pyyaml>=6.0' \
    'opencv-python>=4.8.0' \
    'msgpack>=1.0' \
    'Pillow>=10.0.0' \
    2>&1 | _pip_filter | tail -3

# airsim + msgpack-rpc-python can't be pip-installed on Python 3.12+
# Download source tarballs from PyPI, extract pure-Python packages directly
$PIP install --quiet --user 'setuptools' 'tornado>=6.0' 'nest_asyncio' \
    2>&1 | _pip_filter | tail -1

_AIRSIM_TMP=$(mktemp -d)
_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
mkdir -p "$_SITE"
echo "      Installing airsim (direct-extract for Python 3.12)..."

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

# Patch msgpackrpc for Tornado 6+ compatibility
echo "      Patching msgpackrpc for Tornado 6 compatibility..."
python3 << 'PATCH_SCRIPT'
import pathlib, site

sp = pathlib.Path(site.getusersitepackages()) / "msgpackrpc"
if not sp.exists():
    print("      WARNING: msgpackrpc not found, skipping patch")
    raise SystemExit(0)

patches = {
    "address.py": [
        ("from tornado.platform.auto import set_close_exec",
         "def set_close_exec(fd): pass"),
    ],
    "loop.py": [
        ("IOLoop.instance()", "IOLoop.current()"),
        ("ioloop.PeriodicCallback(callback, callback_time, self._ioloop)",
         "ioloop.PeriodicCallback(callback, callback_time)"),
        ("loop or ioloop.IOLoop()", "loop or ioloop.IOLoop.current()"),
        ("self._ioloop.make_current()\n        self._periodic_callback = None",
         "self._periodic_callback = None"),
    ],
    "server.py": [
        ("IOLoop.instance()", "IOLoop.current()"),
    ],
    "transport/tcp.py": [
        ("from tornado.platform.auto import set_close_exec",
         "def set_close_exec(fd): pass"),
        ("IOLoop.instance()", "IOLoop.current()"),
        ("IOStream(self._address.socket(), io_loop=self._session._loop._ioloop)",
         "IOStream(self._address.socket())"),
        ("tcpserver.TCPServer.__init__(self, io_loop=io_loop)",
         "tcpserver.TCPServer.__init__(self)"),
        ("def __init__(self, transport, io_loop=None, encodings=None):",
         "def __init__(self, transport, encodings=None):"),
        ("MessagePackServer(self, io_loop=self._server._loop._ioloop, encodings=self._encodings)",
         "MessagePackServer(self, encodings=self._encodings)"),
    ],
}

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

# Inject nest_asyncio into loop.py for reentrant event loop support
loop_path = sp / "loop.py"
if loop_path.exists():
    lt = loop_path.read_text()
    if "nest_asyncio" not in lt:
        lt = "import nest_asyncio\nnest_asyncio.apply()\n\n" + lt
        loop_path.write_text(lt)
        print("      Injected nest_asyncio into loop.py")

tcp_path = sp / "transport" / "tcp.py"
if tcp_path.exists():
    text = tcp_path.read_text()
    if "encoding=encodings" in text:
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

python3 -c "import airsim; print('      airsim OK')" \
    || { echo "      ERROR: airsim installation failed!"; exit 1; }
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
