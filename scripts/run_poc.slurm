#!/usr/bin/env bash
###############################################################################
# run_poc.slurm — Slurm batch job to run the Milestone 1 PoC.
#
# Runs tasks 1-5 sequentially inside the VLA container on a DGX node.
# Each task: arm → takeoff → navigate to goal → land → disarm.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_poc.slurm
#   cd ~/vla-proj && sbatch --export=TASKS="1 3" scripts/run_poc.slurm
#
# Output: logs go to ~/vla-proj/logs/poc_<jobid>.out
###############################################################################

#SBATCH --job-name=vla-poc
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/poc_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/poc_%j.out

set -uo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
SQSH_FILE="$HOME/vla_sim.sqsh"
TASKS="${TASKS:-1 2 3 4 5}"

mkdir -p "$PROJ_DIR/logs"

echo "============================================"
echo " VLA Navigation Benchmark — Milestone 1 PoC"
echo " Node:      $(hostname)"
echo " GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Container: $SQSH_FILE"
echo " Tasks:     $TASKS"
echo " Start:     $(date)"
echo "============================================"

if [[ ! -f "$SQSH_FILE" ]]; then
    echo "ERROR: Container image not found at $SQSH_FILE"
    echo "       Run 'bash scripts/build_container.sh' first."
    exit 1
fi

PASSED=0
FAILED=0
RESULTS=""

for TASK_ID in $TASKS; do
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Running Task $TASK_ID                            ║"
    echo "╚══════════════════════════════════════════╝"

    srun \
        --container-image="$SQSH_FILE" \
        --container-mounts="$PROJ_DIR:$PROJ_DIR" \
        --container-remap-root \
        --export=ALL,PROJ_DIR="$PROJ_DIR",PLANNER_MODE="classical" \
        bash "$PROJ_DIR/scripts/run_task.sh" "$TASK_ID" \
        && TASK_RESULT="PASS" \
        || TASK_RESULT="FAIL"

    RESULTS="${RESULTS}  Task $TASK_ID: $TASK_RESULT\n"

    if [[ "$TASK_RESULT" == "PASS" ]]; then
        ((PASSED++)) || true
    else
        ((FAILED++)) || true
    fi

    sleep 15
done

echo ""
echo "============================================"
echo " MILESTONE 1 PoC SUMMARY"
echo "============================================"
echo -e "$RESULTS"
echo "  Passed: $PASSED / $(echo $TASKS | wc -w)"
echo "  Failed: $FAILED / $(echo $TASKS | wc -w)"
echo ""
echo "  End: $(date)"
echo "============================================"

# ── Generate flight-path plots from trajectory CSVs ──────────────────
TRAJ_DIR="$PROJ_DIR/logs/trajectories/classical"
PLOT_DIR="$PROJ_DIR/logs/plots"
CONFIG="$PROJ_DIR/src/vla_navigation/config/benchmark_config.yaml"

if ls "$TRAJ_DIR"/task_*_trajectory.csv 1>/dev/null 2>&1; then
    echo ""
    echo "Generating flight-path plots..."
    srun \
        --container-image="$SQSH_FILE" \
        --container-mounts="$PROJ_DIR:$PROJ_DIR" \
        --container-remap-root \
        bash -c "rm -rf /usr/lib/python3/dist-packages/matplotlib* /usr/lib/python3/dist-packages/mpl_toolkits; \
                 pip install -q 'matplotlib>=3.8' pyyaml 2>/dev/null; \
                 python3 '$PROJ_DIR/scripts/plot_flights.py' \
                     '$TRAJ_DIR' --output '$PLOT_DIR' --config '$CONFIG'" \
        && echo "Plots saved to $PLOT_DIR" \
        || echo "WARNING: Plot generation failed"

    echo ""
    echo "Evaluating classical-planner trajectories..."
    srun \
        --container-image="$SQSH_FILE" \
        --container-mounts="$PROJ_DIR:$PROJ_DIR" \
        --container-remap-root \
        bash -c "pip install -q pyyaml numpy 2>/dev/null; \
                 python3 '$PROJ_DIR/scripts/evaluate.py' \
                     '$TRAJ_DIR' --config '$CONFIG'" \
        && echo "Metrics saved to $TRAJ_DIR/metrics.json" \
        || echo "WARNING: Evaluation failed"
else
    echo "No trajectory CSVs found — skipping plots"
fi

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
