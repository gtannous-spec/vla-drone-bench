#!/usr/bin/env bash
###############################################################################
# run_ablation.slurm — Full three-way ablation study.
#
# Runs all 10 tasks under three planner modes:
#   1. classical  — coordinate-based waypoint controller (no VLA)
#   2. vla_only   — OpenVLA with camera + language instruction only
#   3. vla_gt     — OpenVLA with camera + language + ground-truth coordinates
#
# After each mode, evaluate.py computes metrics.  After all modes,
# report_gen.py generates comparison tables and plots.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_ablation.slurm
#
# Overrides (env vars):
#   MODES="classical vla_only"   — run a subset of modes
#   TASKS="1 2 3"                — run a subset of tasks
###############################################################################

#SBATCH --job-name=vla-ablation
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/ablation_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/ablation_%j.out

set -uo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
SQSH_FILE="$HOME/vla_sim.sqsh"
MODES="${MODES:-classical vla_only vla_gt}"
TASKS="${TASKS:-1 2 3 4 5 6 7 8 9 10}"
CONFIG="$PROJ_DIR/src/vla_navigation/config/benchmark_config.yaml"
VLA_WEIGHTS="${VLA_WEIGHTS:-$HOME/models/openvla-7b}"

mkdir -p "$PROJ_DIR/logs"

echo "╔══════════════════════════════════════════════════╗"
echo "║  VLA Navigation Benchmark — Ablation Study       ║"
echo "╚══════════════════════════════════════════════════╝"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Container:  $SQSH_FILE"
echo " VLA Weights: $VLA_WEIGHTS"
echo " Modes:      $MODES"
echo " Tasks:      $TASKS"
echo " Start:      $(date)"
echo ""

if [[ ! -f "$SQSH_FILE" ]]; then
    echo "ERROR: Container image not found at $SQSH_FILE"
    echo "       Run 'bash scripts/build_container.sh' first."
    exit 1
fi

# ── Run each mode ────────────────────────────────────────────────────────

OVERALL_RESULTS=""

for MODE in $MODES; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  MODE: $MODE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    PASSED=0
    FAILED=0
    MODE_RESULTS=""

    for TASK_ID in $TASKS; do
        echo ""
        echo "  ── Task $TASK_ID [$MODE] ──"

        MOUNTS="$PROJ_DIR:$PROJ_DIR"
        EXTRA_EXPORTS="PROJ_DIR=$PROJ_DIR,PLANNER_MODE=$MODE"

        # Mount VLA weights for non-classical modes
        if [[ "$MODE" != "classical" && -d "$VLA_WEIGHTS" ]]; then
            MOUNTS="$MOUNTS,$VLA_WEIGHTS:$VLA_WEIGHTS"
            EXTRA_EXPORTS="$EXTRA_EXPORTS,VLA_MODEL_PATH=$VLA_WEIGHTS"
        fi

        srun \
            --container-image="$SQSH_FILE" \
            --container-mounts="$MOUNTS" \
            --container-remap-root \
            --export=ALL,$EXTRA_EXPORTS \
            bash "$PROJ_DIR/scripts/run_task.sh" "$TASK_ID" \
            && TASK_RESULT="PASS" \
            || TASK_RESULT="FAIL"

        MODE_RESULTS="${MODE_RESULTS}    Task $TASK_ID: $TASK_RESULT\n"

        if [[ "$TASK_RESULT" == "PASS" ]]; then
            ((PASSED++)) || true
        else
            ((FAILED++)) || true
        fi

        sleep 15
    done

    OVERALL_RESULTS="${OVERALL_RESULTS}\n  [$MODE]\n$MODE_RESULTS"
    OVERALL_RESULTS="${OVERALL_RESULTS}    Passed: $PASSED / $(echo $TASKS | wc -w)\n"

    # ── Evaluate this mode ───────────────────────────────────────────────
    TRAJ_DIR="$PROJ_DIR/logs/trajectories/$MODE"
    BASELINE_FLAG=""
    if [[ "$MODE" != "classical" ]]; then
        CLASSICAL_DIR="$PROJ_DIR/logs/trajectories/classical"
        if [[ -d "$CLASSICAL_DIR" ]]; then
            BASELINE_FLAG="--baseline $CLASSICAL_DIR"
        fi
    fi

    if ls "$TRAJ_DIR"/task_*_trajectory.csv 1>/dev/null 2>&1; then
        echo ""
        echo "  Evaluating $MODE trajectories..."
        srun \
            --container-image="$SQSH_FILE" \
            --container-mounts="$PROJ_DIR:$PROJ_DIR" \
            --container-remap-root \
            bash -c "pip install -q pyyaml numpy 2>/dev/null; \
                     python3 '$PROJ_DIR/scripts/evaluate.py' \
                         '$TRAJ_DIR' \
                         --config '$CONFIG' \
                         $BASELINE_FLAG" \
            && echo "  Metrics saved for $MODE" \
            || echo "  WARNING: Evaluation failed for $MODE"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ABLATION STUDY SUMMARY                          ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "$OVERALL_RESULTS"
echo ""
echo "  End: $(date)"

# ── Generate comparison report ───────────────────────────────────────────

echo ""
echo "Generating comparison report..."
srun \
    --container-image="$SQSH_FILE" \
    --container-mounts="$PROJ_DIR:$PROJ_DIR" \
    --container-remap-root \
    bash -c "rm -rf /usr/lib/python3/dist-packages/matplotlib* /usr/lib/python3/dist-packages/mpl_toolkits; \
             pip install -q 'matplotlib>=3.8' pyyaml numpy 2>/dev/null; \
             python3 '$PROJ_DIR/scripts/report_gen.py' \
                 '$PROJ_DIR/logs/trajectories' \
                 --output '$PROJ_DIR/logs/report' \
                 --config '$CONFIG'" \
    && echo "Report saved to $PROJ_DIR/logs/report/" \
    || echo "WARNING: Report generation failed"
