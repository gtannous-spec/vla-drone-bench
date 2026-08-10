#!/usr/bin/env bash
###############################################################################
# run_train_lora.slurm — LoRA fine-tune OpenFly-Agent-7B on AirSim data.
#
# Expects data at DATA_DIR/manifest.jsonl (produced by run_collect_data.slurm).
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_train_lora.slurm
#   cd ~/vla-proj && sbatch --export=EPOCHS=10,LR=1e-4 scripts/run_train_lora.slurm
###############################################################################

#SBATCH --job-name=lora-train
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/lora_train_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/lora_train_%j.out

set -uo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/lora_training}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/data/lora_checkpoints}"
MODEL="${MODEL:-IPEC-COMMUNITY/openfly-agent-7b}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
UNFREEZE_VISION_LAYERS="${UNFREEZE_VISION_LAYERS:-0}"
FREEZE_PROJECTOR="${FREEZE_PROJECTOR:-0}"
UNFREEZE_LM_HEAD="${UNFREEZE_LM_HEAD:-0}"
N_BINS="${N_BINS:-16}"

mkdir -p "$PROJ_DIR/logs" "$OUTPUT_DIR"

echo "============================================"
echo " LoRA Training — OpenFly on AirSim"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo " Model:      $MODEL"
echo " Data:       $DATA_DIR"
echo " Output:     $OUTPUT_DIR"
echo " Epochs:     $EPOCHS"
echo " Batch:      $BATCH_SIZE (accum=$GRAD_ACCUM)"
echo " LR:         $LR"
echo " LoRA rank:  $RANK, alpha=$ALPHA"
echo " Vision layers unfrozen: $UNFREEZE_VISION_LAYERS"
echo " Freeze projector: $FREEZE_PROJECTOR"
echo " Unfreeze lm_head: $UNFREEZE_LM_HEAD"
echo " Action bins: $N_BINS"
echo " Start:      $(date)"
echo "============================================"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo ""
echo "[1/2] Installing dependencies..."
export PIP_BREAK_SYSTEM_PACKAGES=1
export PATH="$HOME/.local/bin:$PATH"

if ! python3 -m pip --version &>/dev/null; then
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user --quiet 2>&1 | tail -3
fi
PIP="python3 -m pip"
_pip_filter() { grep -v "Error parsing dependencies of" || true; }

# Detect CUDA and install PyTorch
CUDA_DRIVER_VER=$(python3 -c "
import ctypes, ctypes.util
try:
    lib = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
    v = ctypes.c_int()
    lib.cuDriverGetVersion(ctypes.byref(v))
    print(v.value)
except:
    print(0)
" 2>/dev/null)

if [[ "$CUDA_DRIVER_VER" -ge 12010 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi

$PIP install --quiet --user --upgrade \
    'torch>=2.6' 'torchvision>=0.21' \
    --index-url "$TORCH_INDEX" \
    2>&1 | _pip_filter | tail -3

$PIP install --quiet --user --prefer-binary \
    'transformers>=4.47.0,<5.0' \
    'tokenizers>=0.19.0' \
    'timm>=0.9.10,<1.0.0' \
    'Pillow>=10.0.0' \
    'accelerate>=0.25.0' \
    'huggingface_hub>=0.20.0' \
    'peft>=0.12.0' \
    'numpy>=1.24.0' \
    2>&1 | _pip_filter | tail -5

python3 -c "
import torch, transformers, peft
print(f'      PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'      Transformers {transformers.__version__}')
print(f'      PEFT {peft.__version__}')
" || { echo "ERROR: dependency check failed"; exit 1; }

export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

# ── Verify data ───────────────────────────────────────────────────────────────
MANIFEST="$DATA_DIR/manifest.jsonl"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found at $MANIFEST"
    echo "       Run data collection first: sbatch scripts/run_collect_data.slurm"
    exit 1
fi
N_SAMPLES=$(wc -l < "$MANIFEST")
echo "      Training data: $N_SAMPLES samples in $MANIFEST"

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Starting LoRA training..."
echo "============================================"

EXTRA_ARGS=""
if [[ "$UNFREEZE_VISION_LAYERS" -gt 0 ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --unfreeze-vision-layers $UNFREEZE_VISION_LAYERS"
fi
if [[ "$FREEZE_PROJECTOR" == "1" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --freeze-projector"
fi
if [[ "$UNFREEZE_LM_HEAD" == "1" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --unfreeze-lm-head"
fi

python3 -m airsim_benchmark.training.train_lora \
    --data "$MANIFEST" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --n-bins "$N_BINS" \
    $EXTRA_ARGS

echo ""
echo "============================================"
echo " TRAINING COMPLETE"
echo " Checkpoints: $OUTPUT_DIR"
echo " End:         $(date)"
echo "============================================"
