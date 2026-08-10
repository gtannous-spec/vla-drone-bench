#!/bin/bash
#SBATCH --job-name=det-audit
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/audit_%j.out
#SBATCH --error=logs/audit_%j.out

# Detection audit — test GroundingDINO on saved AirSim frames
# No AirSim needed, just GPU for the detection model
#
# Usage:
#   sbatch scripts/run_audit_detection.slurm
#   sbatch --export=ALL,FRAMES_DIR=path/to/frames scripts/run_audit_detection.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"

FRAMES_DIR="${FRAMES_DIR:-logs/airsim_output/detection_bias0.0/frames/mission_16}"
MAX_FRAMES="${MAX_FRAMES:-40}"
OUTPUT="${OUTPUT:-data/detection_audit.csv}"
MODEL="${MODEL:-IDEA-Research/grounding-dino-base}"
LORA_PATH="${LORA_PATH:-}"

echo "============================================"
echo " DETECTION AUDIT"
echo " Frames:     $FRAMES_DIR"
echo " Max frames: $MAX_FRAMES"
echo " Model:      $MODEL"
echo " LoRA:       ${LORA_PATH:-none}"
echo " Output:     $OUTPUT"
echo " Start:      $(date)"
echo "============================================"

LORA_ARG=""
if [[ -n "$LORA_PATH" ]]; then
    LORA_ARG="--lora-path $LORA_PATH"
fi

python3 -m airsim_benchmark.scripts.audit_detection \
    --frames-dir "$FRAMES_DIR" \
    --max-frames "$MAX_FRAMES" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    $LORA_ARG

echo ""
echo "============================================"
echo " AUDIT COMPLETE"
echo " Output: $OUTPUT"
echo " End:    $(date)"
echo "============================================"
