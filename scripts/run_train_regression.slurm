#!/usr/bin/env bash
###############################################################################
# run_train_regression.slurm — Regression-head fine-tune OpenFly on AirSim.
#
# Instead of autoregressive token prediction, this trains a small MLP head
# on top of the LLM's last hidden state to directly output 8 continuous
# action values.  Uses MSE loss instead of cross-entropy.
#
# Usage:
#   cd ~/vla-proj && sbatch scripts/run_train_regression.slurm
#   cd ~/vla-proj && sbatch --export=EPOCHS=10,LR=2e-4 scripts/run_train_regression.slurm
###############################################################################

#SBATCH --job-name=reg-train
#SBATCH --partition=dlc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/users/ogal/gtannous/vla-proj/logs/reg_train_%j.out
#SBATCH --error=/users/ogal/gtannous/vla-proj/logs/reg_train_%j.out

set -uo pipefail

PROJ_DIR="${SLURM_SUBMIT_DIR:-/users/ogal/gtannous/vla-proj}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/lora_training_v4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/data/regression_checkpoints}"
MODEL="${MODEL:-IPEC-COMMUNITY/openfly-agent-7b}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
HEAD_TYPE="${HEAD_TYPE:-film}"
FILTER_MODE="${FILTER_MODE:-agreement_only}"

mkdir -p "$PROJ_DIR/logs" "$OUTPUT_DIR"

echo "============================================"
echo " Regression Head Training — OpenFly on AirSim"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo " Model:      $MODEL"
echo " Data:       $DATA_DIR"
echo " Output:     $OUTPUT_DIR"
echo " Epochs:     $EPOCHS"
echo " Batch:      $BATCH_SIZE (accum=$GRAD_ACCUM)"
echo " LR:         $LR"
echo " LoRA rank:  $RANK, alpha=$ALPHA"
echo " Head type:  $HEAD_TYPE"
echo " Filter:     $FILTER_MODE"
echo " Mode:       REGRESSION (MSE loss, continuous output)"
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
    exit 1
fi
N_SAMPLES=$(wc -l < "$MANIFEST")
echo "      Training data: $N_SAMPLES samples in $MANIFEST"

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Starting regression training..."
echo "============================================"

python3 -m airsim_benchmark.training.train_regression \
    --data "$MANIFEST" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --head-type "$HEAD_TYPE" \
    --filter-mode "$FILTER_MODE"

echo ""
echo "============================================"
echo " TRAINING COMPLETE"
echo " Checkpoints: $OUTPUT_DIR"
echo " End:         $(date)"
echo "============================================"
