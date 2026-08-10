#!/usr/bin/env bash
###############################################################################
# run_survey_landmarks.slurm — Start AirSim NH, dump scene object poses.
#
# Login nodes have no AirSim RPC. This job launches Neighborhood headless,
# then writes data/airsim_objects.txt for landmarks.yaml correction.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_survey_landmarks.slurm
###############################################################################

#SBATCH --job-name=survey-lm
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/survey_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/survey_%j.out

set -uo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
OUT_FILE="${OUT_FILE:-$PROJ_DIR/data/airsim_objects.txt}"
AIRSIM_BINARY="${AIRSIM_BINARY:-$HOME/AirSim/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh}"
AIRSIM_SETTINGS_SRC="$PROJ_DIR/airsim_benchmark/config/settings.json"
AIRSIM_SETTINGS_DST="$HOME/Documents/AirSim/settings.json"

mkdir -p "$PROJ_DIR/logs" "$(dirname "$OUT_FILE")"

echo "============================================"
echo " AirSim landmark survey"
echo " Node:   $(hostname)"
echo " Output: $OUT_FILE"
echo " Start:  $(date)"
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

export PIP_BREAK_SYSTEM_PACKAGES=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

python3 -c "import airsim" 2>/dev/null || {
    echo "ERROR: python3 -c 'import airsim' failed. Use the same env as collection."
    exit 1
}

mkdir -p "$(dirname "$AIRSIM_SETTINGS_DST")"
cp "$AIRSIM_SETTINGS_SRC" "$AIRSIM_SETTINGS_DST"

export SDL_AUDIODRIVER=dummy
_XR="$PROJ_DIR/logs/.openxr_stub.json"
echo '{"file_format_version":"1.0.0","runtime":{"library_path":"/dev/null"}}' > "$_XR"
export XR_RUNTIME_JSON="$_XR"

Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

if [[ ! -f "$AIRSIM_BINARY" ]]; then
    echo "ERROR: AirSim binary not found at $AIRSIM_BINARY"
    exit 1
fi

"$AIRSIM_BINARY" -RenderOffScreen -WINDOWED -nosound &
AIRSIM_PID=$!
echo "AirSim PID: $AIRSIM_PID — waiting 30s..."
sleep 30

if ! kill -0 "$AIRSIM_PID" 2>/dev/null; then
    echo "ERROR: AirSim died during startup."
    exit 1
fi

python3 -m airsim_benchmark.scripts.survey_airsim_objects --out "$OUT_FILE"
status=$?

echo "============================================"
echo " SURVEY COMPLETE  status=$status"
echo " Output: $OUT_FILE"
echo " End:    $(date)"
echo "============================================"
exit "$status"
