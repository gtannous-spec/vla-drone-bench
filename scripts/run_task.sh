#!/usr/bin/env bash
###############################################################################
# run_task.sh — Run a single benchmark task INSIDE the container.
#
# Starts: Xvfb → Gazebo (headless) → PX4 SITL → XRCE-DDS Agent → planner
# Monitors the planner and shuts everything down when the mission completes.
#
# Usage:  run_task.sh [TASK_ID]
#         TASK_ID defaults to 1 (range: 1-10)
###############################################################################

set -uo pipefail
# Note: -e is intentionally omitted — background processes exiting would
# otherwise kill the script before cleanup can run.

TASK_ID="${1:-1}"
PLANNER_MODE="${PLANNER_MODE:-classical}"

export PX4_DIR="${PX4_DIR:-/opt/PX4-Autopilot}"
export GZ_DISTRO=harmonic

echo "============================================"
echo " VLA Benchmark — Task $TASK_ID  [${PLANNER_MODE}]"
echo " PX4:  $PX4_DIR"
echo " Time: $(date)"
echo "============================================"

# Source ROS 2 + workspace (setup.bash uses unset variables)
set +u
source /opt/ros/humble/setup.bash
set -u

# Rebuild vla_navigation from the mounted project dir (picks up latest code)
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ -d "$PROJ_DIR/src/vla_navigation" ]]; then
    echo "[0/5] Rebuilding vla_navigation from mounted source..."
    rm -rf /opt/ros2_ws/src/vla_navigation
    cp -r "$PROJ_DIR/src/vla_navigation" /opt/ros2_ws/src/vla_navigation
    cd /opt/ros2_ws
    set +u
    colcon build --packages-select vla_navigation 2>&1 | tail -3
    set -u
fi

set +u
source /opt/ros2_ws/install/setup.bash
set -u

WORLD_FILE="$(ros2 pkg prefix vla_navigation)/share/vla_navigation/worlds/urban_city.sdf"
CONFIG_FILE="$(ros2 pkg prefix vla_navigation)/share/vla_navigation/config/benchmark_config.yaml"

# PX4 model resources must be visible to Gazebo
export GZ_SIM_RESOURCE_PATH="${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds:${GZ_SIM_RESOURCE_PATH:-}"

# Tell PX4 which model to spawn (without starting its own Gazebo)
export PX4_SIM_MODEL=gz_x500_mono_cam
export PX4_GZ_MODEL_POSE="0,0,0,0,0,0"

cleanup() {
    echo ""
    echo "[cleanup] Stopping all processes..."
    kill "$PX4_PID" 2>/dev/null || true
    kill "$GZ_PID"  2>/dev/null || true
    kill "$DDS_PID"  2>/dev/null || true
    kill "$XVFB_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$PX4_PID" 2>/dev/null || true
    kill -9 "$GZ_PID"  2>/dev/null || true
    kill -9 "$DDS_PID" 2>/dev/null || true
    kill -9 "$XVFB_PID" 2>/dev/null || true
    # Kill any orphan Gazebo / PX4 processes this session may have leaked
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ruby.*gz" 2>/dev/null || true
    pkill -9 -f "px4" 2>/dev/null || true
    pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
    sleep 1
    wait 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT

# ── 1. Virtual framebuffer (headless rendering for Gazebo sensors) ──────

echo "[1/5] Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1280x720x24 &>/dev/null &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

# ── 2. Gazebo Harmonic (server only, no GUI) ───────────────────────────

echo "[2/5] Starting Gazebo server (headless)..."
gz sim -r -s "$WORLD_FILE" &
GZ_PID=$!
sleep 8

# ── 3. PX4 SITL — run the binary directly (don't use 'make') ──────────

echo "[3/5] Starting PX4 SITL (x500_mono_cam)..."
cd "${PX4_DIR}/build/px4_sitl_default"
# Wipe stale state from previous runs so PX4 starts cleanly
rm -f dataman parameters.bson parameters_backup.bson 2>/dev/null || true
rm -rf log/ .ros/ 2>/dev/null || true
./bin/px4 -d -s etc/init.d-posix/rcS -w . &
PX4_PID=$!
sleep 15

# ── 4. Micro-XRCE-DDS Agent ──────────────────────────────────────────

echo "[4/5] Starting Micro-XRCE-DDS Agent..."
MicroXRCEAgent udp4 -p 8888 &
DDS_PID=$!
sleep 5

# ── 5. Classical planner ─────────────────────────────────────────────

MISSION_TIMEOUT="${MISSION_TIMEOUT:-120}"
TRAJ_DIR="${PROJ_DIR}/logs/trajectories/${PLANNER_MODE}"
mkdir -p "$TRAJ_DIR"

echo "[5/5] Starting planner [${PLANNER_MODE}] — Task $TASK_ID (timeout ${MISSION_TIMEOUT}s)"
echo "      Waiting for mission completion..."
echo "============================================"

set +u
if [[ "$PLANNER_MODE" == "classical" ]]; then
    timeout "$MISSION_TIMEOUT" \
        ros2 run vla_navigation classical_planner \
            --ros-args \
            -p task_id:="$TASK_ID" \
            -p config_file:="$CONFIG_FILE" \
            -p output_dir:="$TRAJ_DIR"
    PLANNER_EXIT=$?
else
    timeout "$MISSION_TIMEOUT" \
        ros2 run vla_navigation vla_planner \
            --ros-args \
            -p task_id:="$TASK_ID" \
            -p config_file:="$CONFIG_FILE" \
            -p output_dir:="$TRAJ_DIR" \
            -p mode:="$PLANNER_MODE"
    PLANNER_EXIT=$?
fi
set -u

if [[ $PLANNER_EXIT -eq 124 ]]; then
    echo "[TIMEOUT] Task $TASK_ID exceeded ${MISSION_TIMEOUT}s — killed"
fi

echo ""
echo "============================================"
if [[ $PLANNER_EXIT -eq 0 ]]; then
    echo " RESULT: Task $TASK_ID [$PLANNER_MODE] PASSED"
else
    echo " RESULT: Task $TASK_ID [$PLANNER_MODE] FAILED (exit code $PLANNER_EXIT)"
fi
echo "============================================"

exit $PLANNER_EXIT
