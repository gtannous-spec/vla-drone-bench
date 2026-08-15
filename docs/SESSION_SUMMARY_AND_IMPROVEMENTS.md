# VLA-Based Urban Drone Navigation — Session Summary & Improvement Plan

## Project Overview

**Thesis Topic:** Vision-Language-Action (VLA) based urban drone navigation  
**Goal:** Build a reproducible urban navigation benchmark where a drone navigates purely from camera images and chained language instructions (GPS-free), demonstrating VLA models as high-level decision modules for autonomous flight.

**Simulation Platform:** Microsoft AirSim (Neighborhood environment)  
**Hardware:** NVIDIA A100-SXM4-40GB (Slurm cluster, node `dgx06`)  
**Python:** 3.12.3  
**Primary Model:** IPEC-COMMUNITY/openfly-agent-7b (OpenFly-Agent)

---

## Infrastructure — What We Built

### 1. Modular Codebase (`airsim_benchmark/`)

```
airsim_benchmark/
├── config/
│   ├── benchmark_config.yaml    # Tasks + 10 GPS-free multi-leg missions
│   └── settings.json            # AirSim camera/vehicle config
├── controllers/
│   ├── base_controller.py       # Abstract controller interface
│   ├── classical_controller.py  # Waypoint-based baseline (Milestone 1)
│   ├── openfly_controller.py    # OpenFly VLN model (primary VLA controller)
│   └── llamauav_controller.py   # LLaMA-UAV alternative (tested, poor results)
├── core/
│   ├── airsim_client.py         # AirSim API wrapper (Python 3.12 compatible)
│   ├── drone_fsm.py             # Finite State Machine: IDLE→TAKEOFF→NAVIGATE→LAND→DONE
│   ├── frame_recorder.py        # Camera frame capture → MP4 video
│   └── telemetry.py             # Background telemetry logging thread
├── runner/
│   └── benchmark_runner.py      # Orchestrates tasks & multi-leg missions
├── scripts/
│   └── run_benchmark.py         # CLI entry point (--mode task|mission)
└── visualization/
    └── plot_trajectories.py     # 3D trajectory plots
```

### 2. Benchmark Modes

| Mode | Description | CLI |
|------|-------------|-----|
| **Task** | Single-goal navigation (5 tasks with explicit goal coordinates) | `--mode task` |
| **Mission** | Multi-leg GPS-free navigation (10 missions, chained instructions) | `--mode mission` |

### 3. Mission Architecture (GPS-Free)

Each mission has:
- A **global description** (e.g., "Follow the main road heading north")
- A **start position** (the only coordinate provided)
- Multiple **legs**, each with:
  - A natural-language **instruction** (e.g., "Turn right toward the white house")
  - A **max_hops** budget (hop count determines leg completion)
- **No goal coordinates** — the drone must navigate purely from vision + language

### 4. Drone FSM

```
IDLE → TAKEOFF → NAVIGATE → LAND → DONE
                     ↕
              (per-leg reset in mission mode)
```

In mission mode, the FSM re-enters NAVIGATE for each new leg without re-executing TAKEOFF/LAND between legs.

### 5. Evaluation Metrics

- **Path length** (total meters flown)
- **Heading smoothness** (heading change variance across hops)
- **Action variance** (diversity of model outputs — low = degenerate)
- **Collision count**
- **Mission completion** (all legs finished within timeout)
- **Model-guided ratio** (hops using model vs. fallback)

### 6. Slurm Integration

- `scripts/run_airsim_vla.slurm` — handles Python 3.12 compatibility patches, model download, AirSim startup
- Environment variables: `CONTROLLER`, `GOAL_BIAS`, `RUN_MODE`, `MISSION_IDS`, `WAYPOINT_SCALE`
- Handles: airsim pip install workaround, Tornado 6 monkey-patching, nest_asyncio, msgpack fixes

---

## Model — OpenFly-Agent-7B

### Architecture

```
3 Camera Images (keyframes) + Language Instruction
    ↓
DINOv2 Vision Backbone (SigLIP fused)
    ↓
LLaMA-2-7B Language Model
    ↓
Action Token Generation (8 tokens)
    ↓
Discretized Action Delta → Unnormalize → [stop, dist, heading1, heading2, pitch, 0, 0, 0]
```

### Model Loading Pipeline

1. Download weights from HuggingFace (`IPEC-COMMUNITY/openfly-agent-7b`)
2. Copy custom architecture files from OpenFly-Platform repo (`configuration_prismatic.py`, `modeling_prismatic.py`, `processing_prismatic.py`)
3. Inject `auto_map` into `config.json` and `preprocessor_config.json`
4. Load with `trust_remote_code=True`, `low_cpu_mem_usage=False`, `attn_implementation="eager"`

### Current Inference Pipeline

1. Capture camera frame → maintain 3-image keyframe buffer
2. Process through `PrismaticProcessor` → `input_ids` + `pixel_values`
3. Call `model.predict_action()` → 8-dimensional normalized action
4. Extract dims [1:4] as (distance, heading1, heading2) direction delta
5. Scale by `waypoint_scale` (15.0m) → target waypoint
6. Send `moveToPosition()` to AirSim

---

## Diagnostic Run Results

### Run vla_252901 (pre-fix baseline)
- Controller: OpenFly, Goal Bias: 0.0, Mode: mission
- Every inference produced identical tokens: `[31999 x8]`
- Constant action, drone flew a straight line regardless of input

### Run vla_252927 (post-fix, with Fixes A-D applied)
- Tokenizer extended to 32064 (Fix A confirmed)
- Keyframes confirmed DIVERSE on hops 2, 3, 5 (Fix D confirmed)
- `generate()` STILL collapsed to `[31999 x8]` despite tokenizer extension
- Fix B fallback activated: direct logit extraction produced varied tokens

---

## Confirmed Root Cause — Domain Gap (Not Tokenizer)

### What We Proved

**The initial hypothesis (tokenizer masking) was wrong.** Extending the tokenizer from 32001 to 32064 had no effect on `generate()` output. The definitive evidence came from Fix B's direct logit extraction:

```
[FIX-B] top-5 logit IDs: [31744, 31999, 31914, 31872, 2]
                  values: [ 6.812, 6.750, 0.562, -3.172, -7.750]
[LOGIT-DIAG] global argmax=31744, in action range: False
```

### Key Findings

1. **Action tokens ARE 31744-31999** (the last 256 LLaMA vocab tokens), NOT 32001-32063. Confirmed from OpenFly's `action_tokenizer.py`: `token_id = vocab_size - np.digitize(action, bins)` where `vocab_size = 32000`. IDs 32001-32063 are just padding rows from `pad_to_multiple_of=64`.

2. **The model outputs only STOP and FULL FORWARD.** The direct logit fallback consistently produces:
   - `[31744, 31999, 31999, 31999, 31999, 31999, 31872, 31872]` → `[stop=1.0, fwd=0, yaw=0, ...]` (STOP)
   - `[31999, 31744, 31999, 31999, 31872, 31872, 31872, 31872]` → `[stop=0, fwd=5.0, yaw=0, ...]` (FULL FORWARD)

3. **The model's logit distribution is extremely peaked** — only 2-3 tokens have non-negligible probability. The top two (31744 at 6.812, 31999 at 6.750) are nearly tied; everything else is far below.

4. **Vision inputs are valid and diverse.** Keyframe diversity logging confirms different images per hop. The vision backbone loads correctly.

### Root Cause

**Pure domain gap.** The model was trained on OpenFly-Platform's 3D synthetic environments and has never seen AirSim Neighborhood's visual appearance. It receives valid images but cannot interpret them, defaulting to the two most extreme actions in its vocabulary. The tokenizer was never the bottleneck — the model simply doesn't produce meaningful action distributions for this visual domain.

### OpenFly Action Space (Corrected Understanding)

| Action | ID | 8-D Vector | Token (dim 0) |
|--------|-----|-----------|---------------|
| Stop | 0 | `[1,0,0,0,0,0,0,0]` | 31745 |
| Forward x1 | 1 | `[0,3,0,0,0,0,0,0]` | 31999 (dim0), 31847 (dim1) |
| Turn left 30deg | 2 | `[0,0,15,0,0,0,0,0]` | 31745 (dim2) |
| Turn right 30deg | 3 | `[0,0,0,15,0,0,0,0]` | 31745 (dim3) |
| Go up | 4 | `[0,0,0,0,2,0,0,0]` | 31745 (dim4) |
| Go down | 5 | `[0,0,0,0,0,2,0,0]` | (dim5, masked by q99=0) |
| Forward x2 | 8 | `[0,6,0,0,0,0,0,0]` | clips to same as x3 |
| Forward x3 | 9 | `[0,9,0,0,0,0,0,0]` | clips to same as x2 |

Normalization: `q01=[0,0,0,0,0,0,0,0]`, `q99=[1,5,15,15,2,0,0,0]`

---

## Fixes Applied (in openfly_controller.py)

| Fix | Status | Result |
|-----|--------|--------|
| **A: Extend tokenizer** | DONE | Tokenizer extended 32001 → 32064, but did not fix collapse |
| **B: Direct logit extraction** | DONE | Fallback produces varied tokens; proved issue is domain gap |
| **C: Token-ID diagnostics** | DONE | Logs raw token IDs and collapse detection per inference |
| **D: Keyframe diversity** | DONE | Confirmed images are DIVERSE on hops 2, 3, 5 |
| **E: LoRA fine-tuning** | IN PROGRESS | Full pipeline built, data collection running |

---

## LoRA Fine-Tuning Pipeline (Fix E)

### Architecture

Data collection (oracle controller in AirSim) → keyframe triplets + action labels
→ PyTorch Dataset → peft LoRA on LLM q/k/v/o_proj (vision frozen)
→ adapter checkpoint → merged into base model at inference time

### New Components Built

| File | Purpose |
|------|---------|
| `controllers/oracle_controller.py` | Waypoint oracle: intermediate waypoints, heading-based action classification |
| `scripts/collect_trajectories.py` | Data collection: 3-image triplets, 8-D action vectors, tasks + missions |
| `training/airsim_dataset.py` | PyTorch Dataset: loads triplets, tokenises actions, builds labels |
| `training/train_lora.py` | peft LoRA training: AdamW, cosine schedule, action-only loss |
| `scripts/run_collect_data.slurm` | Slurm job for data collection |
| `scripts/run_train_lora.slurm` | Slurm job for LoRA training |

### LoRA Configuration

- Target modules: `q_proj, k_proj, v_proj, o_proj` (LLM attention)
- Rank: 16, Alpha: 32, Dropout: 0.05
- **Round 1-2 (v1/v2):** Frozen vision backbone + projector (LoRA-only, 33M params)
- **Round 3a (v3):** Projector unfrozen + LoRA (88M params, differential LR)
- **Round 3b (v4):** Projector unfrozen + last 4 vision layers + LoRA (162M params)
- Memory: ~20-30 GB on A100 40GB (gradient checkpointing enabled for Round 3)

### Inference Integration

`openfly_controller.py` accepts `lora_path` parameter. After loading the base model, LoRA weights are merged via `PeftModel.from_pretrained()` + `merge_and_unload()` — zero inference overhead.

Slurm: `LORA_PATH` env var in `run_airsim_vla.slurm` passes through to the controller.

---

## Summary of Completed Milestones

| Milestone | Status | Key Achievement |
|-----------|--------|-----------------|
| **M1: Classical Baseline** | DONE | 5/5 tasks pass with waypoint controller |
| **M2: VLA Integration** | DONE (60% SR with goal_bias) | OpenFly model loaded, inference running, pipeline complete |
| **M3: GPS-Free Missions** | INFRA DONE | 10 missions defined, multi-leg runner works |
| **M3a: Generation Collapse Fix** | DONE | Root cause identified as domain gap (not tokenizer); diagnostics + fallback implemented |
| **M3b: LoRA Pipeline** | IN PROGRESS | Full collection + training + inference pipeline built; data collection running |

---

## Training Data Quality Fix (Critical)

### Problem Identified

LoRA Rounds 1-2 achieved low validation loss (~0.15) but produced no behavioral
improvement.  Root cause: the training data was **degenerate**.

- Only **7 unique action patterns** across 2482 samples
- Each action was a **one-hot discrete vector** (e.g. `[0, 0, 15, 0, 0, 0, 0, 0]`)
- Token distribution: **51.6% token 31999**, 37.5% token 31872, 10.8% token 31744
- Only **4 unique tokens** across all 19,856 token positions
- The model minimized loss by learning the marginal distribution, not visual conditioning

### Fix Applied

1. **`oracle_controller.py`** — now computes **continuous** action vectors from the
   planned displacement (forward distance, yaw angle, altitude change) instead of
   mapping to one-hot discrete patterns via `OPENFLY_ACTION_MAP`.

2. **`collect_trajectories.py`** — measures **actual post-movement displacement**
   after each hop and encodes it as a continuous 8-D vector.  This captures
   multi-dimensional actions (e.g. "forward 4.1m + turn left 8.5° + climb 0.5m"
   in a single sample) and naturally reflects physics/collision effects.

3. **Diversity stats** — collection now logs unique-action counts and per-dimension
   ranges so data quality is visible immediately.

### Expected Impact

| Metric | Old (discrete) | New (continuous) |
|--------|---------------|-----------------|
| Unique action patterns | 7 | ~2000+ (one per sample) |
| Unique individual tokens | 4 | 100+ (full 256-bin range) |
| Multi-dimensional actions | Never | Every sample |
| Token 31999 frequency | 51.6% | ~15-20% (estimated) |

---

## Next Steps

1. **Re-collect training data** with the fixed continuous data collector
2. **Re-run LoRA training** with projector unfreezing (Round 3 config) on the new data
3. Evaluate LoRA-adapted model on missions — expect vision-responsive diverse actions
4. If navigation quality is good → run full 10-mission benchmark, collect metrics
5. Add longer missions (6-8 legs, 1800s timeout) for thesis long-duration navigation goal
6. Iterate: adjust waypoint_scale, collect more data from failure cases, retrain
7. Verify normalization alignment: ensure `VLN_Q99` in `airsim_dataset.py` matches the model's internal `get_action_stats()` values
