#!/bin/bash
# Download all required weights for LLaMA-UAV
# Run once on a node with internet access

set -euo pipefail

MODELS_DIR="${HOME}/models/llama-uav"
mkdir -p "$MODELS_DIR"

echo "[1/4] Downloading Vicuna-7B-v1.5 base model..."
if [ ! -d "$MODELS_DIR/vicuna-7b-v1.5" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lmsys/vicuna-7b-v1.5',
                  local_dir='${MODELS_DIR}/vicuna-7b-v1.5',
                  local_dir_use_symlinks=False)
print('Done.')
"
else
    echo "  Already exists, skipping."
fi

echo "[2/4] Downloading EVA-ViT-G weights..."
if [ ! -f "$MODELS_DIR/eva_vit_g.pth" ]; then
    wget -q --show-progress -O "$MODELS_DIR/eva_vit_g.pth" \
        "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
else
    echo "  Already exists, skipping."
fi

echo "[3/4] Downloading Q-Former (InstructBLIP) weights..."
if [ ! -f "$MODELS_DIR/instruct_blip_vicuna7b_trimmed.pth" ]; then
    wget -q --show-progress -O "$MODELS_DIR/instruct_blip_vicuna7b_trimmed.pth" \
        "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"
else
    echo "  Already exists, skipping."
fi

echo "[4/4] Downloading LLaMA-UAV adapter (LoRA + projector)..."
if [ ! -d "$MODELS_DIR/llama-uav-7b" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('wangxiangyu0814/llama-uav-7b',
                  local_dir='${MODELS_DIR}/llama-uav-7b',
                  local_dir_use_symlinks=False)
print('Done.')
"
else
    echo "  Already exists, skipping."
fi

echo ""
echo "All weights downloaded to: $MODELS_DIR"
ls -lh "$MODELS_DIR"
