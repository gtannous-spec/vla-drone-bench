#!/usr/bin/env bash
###############################################################################
# download_vla_model.slurm — Slurm job to download OpenVLA-7B model weights.
#
# Downloads the ~14GB model from HuggingFace to ~/models/openvla-7b.
# Run this ONCE before running the VLA benchmark.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/download_vla_model.slurm
#
# Environment variables:
#   MODEL_ID   — HuggingFace model ID (default: openvla/openvla-7b)
#   MODEL_PATH — local destination (default: ~/models/openvla-7b)
#   HF_TOKEN   — HuggingFace API token (optional, for gated models)
###############################################################################

#SBATCH --job-name=vla-download
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/vla_download_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/vla_download_%j.out

set -euo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
MODEL_ID="${MODEL_ID:-openvla/openvla-7b}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/openvla-7b}"

echo "============================================"
echo " OpenVLA Model Download"
echo " Node:       $(hostname)"
echo " Model:      $MODEL_ID"
echo " Dest:       $MODEL_PATH"
echo " Start:      $(date)"
echo "============================================"

# ── Install dependencies ──────────────────────────────────────────────────────

echo ""
echo "[1/3] Installing Python dependencies..."
pip install --quiet --user 'huggingface_hub>=0.20.0' 'pyyaml' 2>&1 | tail -3
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"
echo "      Done."

# ── Download model ────────────────────────────────────────────────────────────

echo ""
echo "[2/3] Downloading model (~14GB, may take 10-30 minutes)..."
mkdir -p "$MODEL_PATH"

python3 -m airsim_benchmark.scripts.download_model \
    --model-id "$MODEL_ID" \
    --output "$MODEL_PATH"

DOWNLOAD_EXIT=$?

# ── Verify ────────────────────────────────────────────────────────────────────

echo ""
echo "[3/3] Verifying download..."
if [[ $DOWNLOAD_EXIT -eq 0 ]] && [[ -f "$MODEL_PATH/config.json" ]]; then
    MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    echo "      Model downloaded successfully."
    echo "      Size: $MODEL_SIZE"
    echo "      Path: $MODEL_PATH"
else
    echo "      ERROR: Download failed or incomplete."
    exit 1
fi

echo ""
echo "============================================"
echo " DOWNLOAD COMPLETE"
echo " Model ready at: $MODEL_PATH"
echo " End: $(date)"
echo ""
echo " Next step — run the VLA benchmark:"
echo "   sbatch --export=SKIP_AIRSIM=0 scripts/run_airsim_vla.slurm"
echo "============================================"
