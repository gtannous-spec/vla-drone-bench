#!/usr/bin/env bash
###############################################################################
# resume_build.sh — Resume container build from step 5 (PX4 Autopilot).
#
# Steps 1-4 (base packages, ROS 2, Gazebo, XRCE-DDS) already completed.
# This script runs steps 5-7 inside the saved container and produces the
# final vla_sim.sqsh.
#
# Usage:   bash scripts/resume_build.sh
###############################################################################

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SQSH_FILE="$HOME/vla_sim.sqsh"

echo "============================================"
echo " VLA Benchmark — Resume Build (steps 5-7)"
echo " Project:   $PROJ_DIR"
echo " Output:    $SQSH_FILE"
echo "============================================"

# Remove the failed/partial sqsh so --container-save can write a fresh one
rm -f "$SQSH_FILE"

echo "[*] Submitting resume build job to Slurm..."
echo "    This will take 15-30 minutes (PX4 clone + compile)."
echo ""

srun \
    --partition=dlc \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=32G \
    --time=02:00:00 \
    --job-name=vla-resume \
    --container-image=ubuntu:22.04 \
    --container-save="$SQSH_FILE" \
    --container-remap-root \
    --container-mounts="$PROJ_DIR:$PROJ_DIR" \
    bash "$PROJ_DIR/scripts/setup_inside_container.sh" "$PROJ_DIR"

echo ""
echo "Container saved to: $SQSH_FILE"
echo "Size: $(du -h "$SQSH_FILE" | cut -f1)"
echo ""
echo "Next step:  sbatch scripts/run_poc.slurm"
