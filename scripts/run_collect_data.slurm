#!/usr/bin/env bash
###############################################################################
# run_collect_data.slurm — Collect LoRA training data from AirSim.
#
# Launches AirSim headless, then runs collect_trajectories.py with the
# oracle controller to generate keyframe triplets + OpenFly-compatible
# action labels.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_collect_data.slurm
#   cd ~/vla-proj && sbatch --export=ALL,EPISODES=24,OUTPUT_DIR=$PWD/data/lora_training_v4_smoke scripts/run_collect_data.slurm
#
# Output: data/lora_training_v4/manifest.jsonl + per-episode image dirs
###############################################################################

#SBATCH --job-name=collect-data
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/collect_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/collect_%j.out

set -uo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
EPISODES="${EPISODES:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/data/lora_training_v4}"
AIRSIM_BINARY="${AIRSIM_BINARY:-$HOME/AirSim/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh}"
AIRSIM_SETTINGS_SRC="$PROJ_DIR/airsim_benchmark/config/settings.json"
AIRSIM_SETTINGS_DST="$HOME/Documents/AirSim/settings.json"

mkdir -p "$PROJ_DIR/logs" "$OUTPUT_DIR"

echo "============================================"
echo " LoRA Data Collection"
echo " Node:          $(hostname)"
echo " Episodes:      $EPISODES"
echo " Output:        $OUTPUT_DIR"
echo " Start:         $(date)"
echo "============================================"

AIRSIM_PID=""
XVFB_PID=""
cleanup() {
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

# ── Dependencies ──────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing dependencies..."
export PIP_BREAK_SYSTEM_PACKAGES=1
export PATH="$HOME/.local/bin:$PATH"

if ! python3 -m pip --version &>/dev/null; then
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user --quiet 2>&1 | tail -3
fi
PIP="python3 -m pip"
_pip_filter() { grep -v "Error parsing dependencies of" || true; }

$PIP install --quiet --user --prefer-binary \
    'numpy>=1.24.0' 'pyyaml>=6.0' 'Pillow>=10.0.0' 'opencv-python>=4.8.0' \
    2>&1 | _pip_filter | tail -3

# airsim (same workaround as the benchmark slurm)
_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
python3 -c "import airsim" 2>/dev/null || {
    echo "      airsim not found — installing via extract workaround..."
    $PIP install --quiet --user 'setuptools' 'tornado>=6.0' 'msgpack>=1.0' 'nest_asyncio' \
        2>&1 | _pip_filter | tail -1
    _TMP=$(mktemp -d)
    _download_pypi() {
        local url
        url=$(python3 -c "
import urllib.request, json
data = json.loads(urllib.request.urlopen('https://pypi.org/pypi/$1/json').read())
sdists = [u['url'] for u in data['urls'] if u['packagetype'] == 'sdist']
print(sdists[0] if sdists else '')
" 2>/dev/null)
        [[ -n "$url" ]] && curl -sSL "$url" -o "$_TMP/$1.tar.gz"
    }
    _download_pypi airsim; _download_pypi msgpack-rpc-python
    for t in "$_TMP"/*.tar.gz; do [ -f "$t" ] && tar -xzf "$t" -C "$_TMP" 2>/dev/null; done
    for p in "$_TMP"/*/airsim "$_TMP"/*/msgpackrpc; do
        [ -d "$p" ] && cp -r "$p" "$_SITE/" && echo "      Extracted $(basename "$p")"
    done
    rm -rf "$_TMP"
}
python3 -c "import airsim; print(f'      airsim OK')" || { echo "ERROR: airsim"; exit 1; }

export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

# ── AirSim settings ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Deploying AirSim settings..."
mkdir -p "$(dirname "$AIRSIM_SETTINGS_DST")"
cp "$AIRSIM_SETTINGS_SRC" "$AIRSIM_SETTINGS_DST"

# ── Launch AirSim ─────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Starting AirSim..."
export SDL_AUDIODRIVER=dummy
_XR="$PROJ_DIR/logs/.openxr_stub.json"
echo '{"file_format_version":"1.0.0","runtime":{"library_path":"/dev/null"}}' > "$_XR"
export XR_RUNTIME_JSON="$_XR"

Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

"$AIRSIM_BINARY" -RenderOffScreen -WINDOWED -nosound &
AIRSIM_PID=$!
echo "      AirSim PID: $AIRSIM_PID — waiting 30s..."
sleep 30

if ! kill -0 "$AIRSIM_PID" 2>/dev/null; then
    echo "ERROR: AirSim died during startup."
    exit 1
fi
echo "      AirSim running."

# ── Collect data ──────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Collecting trajectories..."
echo "============================================"

python3 -m airsim_benchmark.scripts.collect_trajectories \
    --output "$OUTPUT_DIR" \
    --episodes "$EPISODES"
status=$?

echo ""
echo "============================================"
echo " COLLECTION COMPLETE"
echo " Output:   $OUTPUT_DIR"
echo " Manifest: $OUTPUT_DIR/manifest.jsonl"
echo " End:      $(date)"
echo "============================================"
exit "$status"
