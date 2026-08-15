# Experiment Log — VLA Drone Navigation

**Date**: August 14, 2026  
**Project**: Vision-Language-Action (VLA) Drone Navigation in AirSim  
**Period covered**: August 11–14, 2026

---

## Overview

This document records every experiment we ran, what we changed between runs, what we observed, and what we learned. The goal is to adapt the OpenFly-Agent-7B model (trained on a different simulator) to navigate drones in Microsoft AirSim's Neighborhood environment.

---

## Datasets Used

### Dataset v1: Discrete Actions (data/lora_training)

- **Collection method**: Oracle controller flying waypoint-based paths, recording one of 10 discrete action classes (stop, forward slow/medium/fast, turn left, turn right, go up, go down, move left, move right)
- **Samples**: ~2,482
- **Action diversity**: Only **7 unique action patterns** across all samples
- **Problem**: Actions were one-hot encoded (e.g., `[0, 0, 15, 0, 0, 0, 0, 0]` for "turn left"), meaning most of the 8 dimensions were exactly zero. The model learned to output zeros (token 31999) for everything because that was the majority class
- **Used by**: Training runs v1, v2, v3, v4

### Dataset v2: Continuous Actions (data/lora_training_v2)

- **Collection method**: Same oracle controller, but now recording the **actual displacement** the drone achieved at each step — measuring position before and after movement to get true continuous values
- **Samples**: 3,533
- **Action diversity**: ~2,900 unique action patterns — a massive improvement over v1
- **Fix applied**: Rewrote `oracle_controller.py` and `collect_trajectories.py` to compute continuous 8-dimensional action vectors from real displacement instead of mapping to discrete classes
- **Used by**: Training runs v5, v6, v7, v8

### Dataset v3: Continuous Actions, 10x Scale (data/lora_training_v3)

- **Collection method**: Same continuous-action approach as v2
- **Configuration**: 100 task runs x 25 mission runs = 2,500 episodes
- **Expected samples**: ~25,000–30,000
- **Status as of Aug 14**: Still collecting (~544 episodes completed, estimated ~33 hours remaining)
- **Purpose**: Provide enough training data for the model to learn diverse, input-dependent actions instead of memorizing a single dominant pattern

---

## Training Infrastructure

All training runs use the same core pipeline:

- **Hardware**: Single NVIDIA A100-SXM4-40GB GPU on DGX cluster (Slurm job scheduler)
- **Framework**: PyTorch 2.6, HuggingFace Transformers 4.57, PEFT 0.20
- **Training script**: `airsim_benchmark/training/train_lora.py`
- **Method**: LoRA (Low-Rank Adaptation) applied to the LLM's attention layers (q_proj, k_proj, v_proj, o_proj)
- **Batch size**: Effective batch of 32 (batch_size=1, gradient accumulation=32) — forced to batch_size=1 because the vision backbone cannot handle batched image tensors
- **Optimizer**: AdamW with differential learning rates (different rates for LoRA adapters, projector, and vision layers)
- **Early stopping**: Patience of 3 epochs on validation loss
- **Loss function**: Cross-entropy loss computed only on the 8 action token positions (instruction tokens are masked)

---

## Experiment Timeline

### Phase 1: Initial Diagnosis (Aug 11)

**Goal**: Understand why the base OpenFly model produces constant output in AirSim.

**What we did**:
- Ran the base model on AirSim mission 1 (navigate to intersection)
- Added diagnostic logging to capture raw logits and token probabilities
- Discovered the model outputs the same action every single step regardless of camera input

**Findings**:
- The model's top logits were for regular text tokens (Chinese characters), not action tokens (31744–31999)
- Root cause identified as **domain gap**: the model was trained on OpenFly-Platform visuals and cannot interpret AirSim images
- A tokenizer mismatch was also found (tokenizer had 32,001 entries but lm_head expected 32,064) — fixed by extending the tokenizer with placeholder tokens

**Fixes implemented**:
- Fix A: Extended tokenizer to match lm_head embedding size
- Fix B: Added fallback direct-logit inference path for diagnostics
- Fix C: Added action logit diagnostic logging
- Fix D: Added image triplet history buffer management

### Phase 2: LoRA Infrastructure Build (Aug 11–12)

**Goal**: Build the complete LoRA fine-tuning pipeline from scratch.

**What we built**:
1. `oracle_controller.py` — Waypoint-following controller that generates diverse flight trajectories
2. `collect_trajectories.py` — Data collection script that records image-action pairs during oracle flights
3. `airsim_dataset.py` — PyTorch Dataset that loads collected data and prepares it for training
4. `train_lora.py` — LoRA training loop with validation, early stopping, and diagnostics
5. `run_collect_data.slurm` — Slurm job script for data collection
6. `run_train_lora.slurm` — Slurm job script for LoRA training

### Phase 3: Training Runs v1–v4 (Aug 12–13) — Discrete Action Data

All four runs used **Dataset v1** (discrete actions, 2,482 samples, 7 unique patterns).

#### Run v1 (Slurm job 253049)

| Setting | Value |
|---------|-------|
| Data | v1 (discrete, 2,482 samples) |
| LR | 2e-4 |
| Epochs | 5 |
| Rank | 16 |
| Projector | Frozen |
| Vision backbone | Frozen |
| lm_head | Frozen |

- **Result**: Crashed with a batching error (`RuntimeError: split_with_sizes expects split_sizes to sum exactly to 3`)
- **Fix**: Forced batch_size=1 and adjusted gradient accumulation

#### Run v2 (Slurm job 253099)

| Setting | Value |
|---------|-------|
| Data | v1 (discrete, 2,482 samples) |
| LR | 5e-4 |
| Epochs | 10 |
| Rank | 32 |
| Projector | Frozen |
| Vision backbone | Frozen |
| lm_head | Frozen |
| Best val loss | **0.149** |

- **Result**: Very low validation loss — appeared to learn well
- **AirSim inference**: Model collapsed to a different constant action than the base model, but still constant. It learned the single most common action in the training data (the "majority class") rather than responding to visual input
- **Diagnosis**: With only 7 unique action patterns and most token positions being 31999 (zero value), a val loss of 0.15 is achievable by simply predicting 31999 everywhere

#### Run v3 (Slurm job 253121)

| Setting | Value |
|---------|-------|
| Data | v1 (discrete, 2,482 samples) |
| LR | 2e-4 |
| Epochs | 10 |
| Rank | 16 |
| Projector | **Unfrozen** |
| Vision backbone | Frozen |
| lm_head | Frozen |
| Best val loss | **0.160** |

- **Change from v2**: Unfroze the projector (the vision-to-language translator) based on the hypothesis that the LLM couldn't receive useful visual information through a frozen projector
- **Result**: Similar val loss, same constant-output behavior during inference
- **Diagnosis**: The data was too degenerate for any architectural change to help

#### Run v4 (Slurm job 253122)

| Setting | Value |
|---------|-------|
| Data | v1 (discrete, 2,482 samples) |
| LR | 2e-4 |
| Epochs | 10 |
| Rank | 16 |
| Projector | **Unfrozen** |
| Vision backbone | **Unfrozen** (last 4 layers) |
| lm_head | Frozen |
| Best val loss | **0.158** |

- **Change from v3**: Also unfroze the last 4 layers of the vision backbone to let it adapt to AirSim visuals
- **Result**: Negligible difference from v3
- **Diagnosis**: Confirmed the problem was the training data, not the model capacity

### Phase 4: Data Fix — Continuous Actions (Aug 13)

**Key insight**: The training data had only 7 unique action patterns because the oracle controller was mapping movements to discrete classes. We rewrote the data collection to record the actual displacement as continuous values.

**Changes made**:
1. `oracle_controller.py` — Changed `get_action()` to compute 8D continuous vectors from planned displacement instead of mapping to one of 10 discrete actions
2. `collect_trajectories.py` — Added `compute_continuous_action()` that measures position before/after movement and computes the true displacement vector

**Result**: Dataset v2 had ~2,900 unique action patterns across 3,533 samples (vs. 7 patterns in v1).

### Phase 5: Training Runs v5–v7 (Aug 13) — Continuous Data, 256 Bins

#### Run v5 (Slurm job 253257)

| Setting | Value |
|---------|-------|
| Data | v2 (continuous, 3,533 samples) |
| LR | 2e-4 |
| Epochs | 10 |
| Rank | 16 |
| Bins | 256 |
| Projector | Unfrozen |
| Vision backbone | Frozen |
| lm_head | Frozen |
| Best val loss | **1.110** |

- **Result**: Much higher val loss than v2-v4, which makes sense — the task is now genuinely harder (continuous actions vs. 7-class classification). The model was producing non-action text tokens (278, 18403, etc.) during inference instead of action tokens in the 31744–31999 range
- **Fix applied**: Added `ActionTokenLogitsProcessor` to constrain generation to only produce valid action tokens
- **AirSim inference**: With the logit processor, all outputs collapsed to token 31999 (the lowest bin value). The model learned to distinguish only 3–5 tokens out of 256 — the bin resolution was too fine for the amount of training data

#### Run v6 (Slurm job 253383)

| Setting | Value |
|---------|-------|
| Data | v2 (continuous, 3,533 samples) |
| LR | 2e-4 |
| Epochs | 10 |
| Rank | 16 |
| Bins | 256 |
| Projector | Unfrozen |
| Vision backbone | Frozen |
| lm_head | **Unfrozen** |
| Best val loss | **1.083** |

- **Change from v5**: Unfroze the lm_head (the final layer that maps hidden states to token probabilities) based on the hypothesis that a frozen lm_head prevented the model from learning to differentiate action tokens
- **Result**: Slightly lower val loss (1.083 vs 1.110). Diagnostics showed per-dimension losses were still very high for all non-stop dimensions (4.6–3.9). Predictions still collapsed to all-31999
- **Diagnosis**: 256 bins is still too many classes for the model to learn from 3,533 samples, even with unfrozen lm_head

#### Run v7 (Slurm job 253400)

| Setting | Value |
|---------|-------|
| Data | v2 (continuous, 3,533 samples) |
| LR | 2e-4 |
| Epochs | 10 |
| Rank | 16 |
| Bins | 256 |
| Projector | Unfrozen |
| Vision backbone | Frozen |
| lm_head | **Unfrozen** |
| Best val loss | **1.085** |

- **Change from v6**: None — identical configuration, run as a parallel reproducibility check
- **Result**: Nearly identical to v6, confirming the results are stable and reproducible

### Phase 6: Training Run v8 (Aug 14) — 16 Bins

#### Run v8 (Slurm job 253586)

| Setting | Value |
|---------|-------|
| Data | v2 (continuous, 3,533 samples) |
| LR | 2e-4 |
| Epochs | 10 |
| Rank | 16 |
| Bins | **16** |
| Projector | Unfrozen |
| Vision backbone | Frozen |
| lm_head | Unfrozen |
| Best val loss | **0.444** |

- **Change from v6/v7**: Reduced the number of action bins from 256 to 16, making it a 16-class classification per dimension instead of 256-class
- **Result**: Val loss dropped significantly (0.444 vs 1.083), confirming the classification is much easier with fewer bins
- **Positive signal**: The top-5 predicted tokens shifted from old 256-bin tokens to proper 16-bin tokens — the model learned the correct vocabulary
- **Remaining problem**: Despite better loss, predictions still collapse to all-31999. The logit gap between 31999 and the next-best token is ~4 points, suggesting the model needs more training data to close this gap

---

## Ablation Summary

### What We Varied and What We Learned

| Experiment | Variable Changed | Result | Conclusion |
|---|---|---|---|
| v2 vs v3 | Projector: frozen → unfrozen | Val loss 0.149 → 0.160 (similar) | No effect with degenerate data |
| v3 vs v4 | Vision backbone: frozen → last 4 layers unfrozen | Val loss 0.160 → 0.158 (similar) | No effect with degenerate data |
| v2 vs v5 | Data: discrete → continuous | Val loss 0.149 → 1.110 (higher) | The task became genuinely harder; low loss on discrete data was misleading |
| v5 vs v6 | lm_head: frozen → unfrozen | Val loss 1.110 → 1.083 (slightly better) | Modest improvement, not transformative with 256 bins |
| v6 vs v8 | Bins: 256 → 16 | Val loss 1.083 → 0.444 (much better) | Fewer bins dramatically reduces classification difficulty |
| All v1–v4 | Any architecture change | All ~0.15 val loss, all constant output | **Data quality is more important than model capacity** |

### Key Findings

1. **Data quality dominates**: No architectural change (projector, vision layers, lm_head) could overcome degenerate training data. The single most impactful change was switching from 7 discrete action patterns to 2,900 continuous action patterns.

2. **Low loss can be misleading**: A val loss of 0.15 on discrete data sounds great, but the model simply memorized "output 31999 for most positions" — which is the trivially correct answer for data where 80%+ of token positions are 31999.

3. **The constant-output problem persists across all runs**: In every single training run (v1–v8), the model's argmax prediction is all-31999 for every validation sample. The model learns some probability mass for correct tokens (reflected in the loss going down), but never enough to overtake 31999 as the top prediction.

4. **Unfreezing lm_head is necessary but not sufficient**: With a frozen lm_head, the model cannot redirect its output from language tokens to action tokens. But unfreezing it alone doesn't solve the collapse — more data is needed.

5. **Fewer bins helps, but doesn't solve the fundamental problem**: 16-bin classification (val loss 0.44) is easier than 256-bin (val loss 1.08), but the model still collapses. The dominant token (31999) is too entrenched.

---

## Current Status (Aug 14, 2026)

### What Is Running
- **Data collection v3** (job 253585): Collecting ~25,000–30,000 continuous-action samples from 2,500 episodes. ~22% complete, estimated to finish Saturday morning.

### What Is Planned
- **Training v9**: Will use Dataset v3 (~10x more data than v2), with lm_head unfrozen, projector unfrozen, backbone frozen, 256 bins, rank 16, LR 2e-4, 10 epochs
- **AirSim inference**: Run the v9 checkpoint on missions 1 and 16 to evaluate whether 10x more data closes the logit gap

### Open Questions
1. Will 10x more data be enough to break the 31999 dominance, or is there a fundamental limit to LoRA's capacity for this task?
2. Should we try higher LoRA rank (32 or 64) to give the model more adaptation capacity?
3. Would a completely different decoding approach (regression head instead of autoregressive token prediction) avoid the binning problem entirely?
