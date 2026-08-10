#!/bin/bash
###############################################################################
# run_finetune_dino.slurm — Fine-tune GroundingDINO with LoRA on aerial data.
#
# Loads the base grounding-dino-base model, applies LoRA on the vision
# backbone, and trains on annotated aerial frames from AirSim.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_finetune_dino.slurm
#   cd ~/vla-proj && sbatch --export=ALL,EPOCHS=100,LR=5e-5 scripts/run_finetune_dino.slurm
###############################################################################

#SBATCH --job-name=dino-lora
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/dino_ft_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/dino_ft_%j.out

set -euo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/dino_finetune}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/models/dino-aerial-lora}"
MODEL_ID="${MODEL_ID:-IDEA-Research/grounding-dino-base}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"

mkdir -p "$PROJ_DIR/logs" "$OUTPUT_DIR"

echo "============================================"
echo " GroundingDINO LoRA Fine-Tuning"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Data:       $DATA_DIR"
echo " Output:     $OUTPUT_DIR"
echo " Model:      $MODEL_ID"
echo " Epochs:     $EPOCHS"
echo " LR:         $LR"
echo " LoRA rank:  $LORA_RANK"
echo " Start:      $(date)"
echo "============================================"

export PIP_BREAK_SYSTEM_PACKAGES=1
export PATH="$HOME/.local/bin:$PATH"

PIP="python3 -m pip"
_pip_filter() { grep -v "Error parsing dependencies of" || true; }

echo ""
echo "[1/2] Installing dependencies..."
$PIP install --quiet --user --prefer-binary \
    'peft>=0.11.0' 'scipy>=1.10.0' \
    2>&1 | _pip_filter | tail -3

export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

echo ""
echo "[2/2] Starting training..."
echo "============================================"

python3 -m airsim_benchmark.training.finetune_dino \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-id "$MODEL_ID" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --save-merged
status=$?

echo ""
echo "============================================"
echo " FINE-TUNING COMPLETE"
echo " Output:  $OUTPUT_DIR"
echo " End:     $(date)"
echo "============================================"
exit "$status"
