#!/usr/bin/env bash
###############################################################################
# build_container.sh — Build the VLA simulation container on the DGX cluster.
#
# This script uses Pyxis (--container-image) to pull ubuntu:22.04 and runs
# a setup script inside it, saving the result as a .sqsh image.
#
# Usage:   bash scripts/build_container.sh
# Result:  ~/vla_sim.sqsh  (the reusable container image)
###############################################################################

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SQSH_FILE="$HOME/vla_sim.sqsh"
SETUP_SCRIPT="$PROJ_DIR/scripts/setup_inside_container.sh"

echo "============================================"
echo " VLA Benchmark — Container Build"
echo " Project:   $PROJ_DIR"
echo " Output:    $SQSH_FILE"
echo "============================================"

if [[ -f "$SQSH_FILE" ]]; then
    echo "[!] $SQSH_FILE already exists."
    echo "    Delete it first if you want to rebuild:  rm $SQSH_FILE"
    exit 1
fi

echo "[1/2] Submitting container build job to Slurm..."
echo "      This will take 20-40 minutes (downloads + compiles PX4)."
echo ""

srun \
    --partition=dlc \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=32G \
    --time=02:00:00 \
    --job-name=vla-build \
    --container-image=ubuntu:22.04 \
    --container-save="$SQSH_FILE" \
    --container-remap-root \
    --container-mounts="$PROJ_DIR:$PROJ_DIR" \
    bash "$SETUP_SCRIPT" "$PROJ_DIR"

echo ""
echo "[2/2] Container saved to: $SQSH_FILE"
echo "      Size: $(du -h "$SQSH_FILE" | cut -f1)"
echo ""
echo "Next step: run the PoC with:"
echo "  sbatch scripts/run_poc.slurm"
