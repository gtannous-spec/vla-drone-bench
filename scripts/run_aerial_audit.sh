#!/bin/bash
# Quick aerial prompt engineering audit — no Slurm needed, runs on current GPU
# Tests alternative phrases for the 7 objects that failed at 0% detection

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"

FRAMES_DIR="logs/airsim_output/detection_bias0.0/frames/mission_16"
OUTPUT="data/aerial_prompt_audit.csv"

echo "============================================"
echo " Aerial Prompt Engineering Audit"
echo " Testing alternative phrases for failed objects"
echo " Start: $(date)"
echo "============================================"

python3 -m airsim_benchmark.scripts.audit_detection \
    --frames-dir "$FRAMES_DIR" \
    --max-frames 40 \
    --output "$OUTPUT" \
    --queries \
        "rooftop" \
        "roof" \
        "roof from above" \
        "house roof" \
        "flat roof" \
        "building top" \
        "house top view" \
        "building" \
        "residential building" \
        "tree" \
        "green tree" \
        "tree canopy" \
        "foliage" \
        "green bush" \
        "vegetation" \
        "plant" \
        "fence" \
        "wooden fence" \
        "barrier" \
        "railing" \
        "picket fence" \
        "intersection" \
        "road crossing" \
        "crossroad" \
        "road junction" \
        "traffic intersection" \
        "garage" \
        "garage door" \
        "carport" \
        "car shelter" \
        "structure" \
        "small building" \
        "shed" \
        "outdoor structure" \
        "swimming pool" \
        "pool" \
        "water pool" \
        "blue pool" \
        "mailbox" \
        "post box" \
        "letter box"

echo ""
echo "============================================"
echo " AUDIT COMPLETE: $OUTPUT"
echo " End: $(date)"
echo "============================================"
