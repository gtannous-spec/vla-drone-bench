# VLA Urban Drone Navigation

A system that navigates a drone using **natural language instructions** and **camera vision** — no GPS, no pre-programmed waypoints.

Give it an instruction like *"Fly to the red car, then land on the rooftop"* and the drone carries it out by looking through its camera and finding the objects mentioned in the text.

## How It Works

```
                        "Fly to the red car, then land on the rooftop"
                                          │
                          ┌───────────────┴───────────────┐
                          │     INSTRUCTION PARSER         │
                          │  Breaks text into steps:       │
                          │   1. navigate(red car)         │
                          │   2. land(rooftop, near=car)   │
                          └───────────────┬───────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │            FOR EACH HOP (~2 seconds)       │
                    │                                            │
                    │   ┌──────────────┐   ┌──────────────┐     │
                    │   │ GroundingDINO│   │  OpenFly-7B  │     │
                    │   │  (234M)      │   │  (7.6B)      │     │
                    │   │              │   │              │     │
                    │   │ "WHERE is    │   │ "HOW to fly  │     │
                    │   │  the target?"│   │  smoothly?"  │     │
                    │   └──────┬───────┘   └──────┬───────┘     │
                    │          │                   │             │
                    │          └─────────┬─────────┘             │
                    │                    │                       │
                    │              BLEND: direction              │
                    │              from DINO + flight            │
                    │              dynamics from OpenFly         │
                    │                    │                       │
                    │                    ▼                       │
                    │          Move drone 5m toward target       │
                    │          Take new camera frame             │
                    │          Repeat                            │
                    └───────────────────────────────────────────┘
```

### The Two AI Models

| Model | Role | What it does |
|-------|------|-------------|
| **GroundingDINO** (234M params) | The Eyes | Finds objects in camera images from text descriptions |
| **OpenFly-Agent-7B** (7.6B params) | The Pilot | Trained on 100K drone flights, provides flight dynamics |

GroundingDINO tells the drone **where** to go. OpenFly tells it **how** to fly there.

### Navigation Modes

| Mode | When | Behavior |
|------|------|----------|
| **TRACK** | Target visible in camera | Fly toward it |
| **COAST** | Target lost briefly (1-3 hops) | Continue on last known heading |
| **SEARCH** | Target not found | Rotate 360° to scan the area |

### Landing Approach

```
Drone ----→ ----→ ----↘ ----↘ ----↓ LAND
 cruise altitude     descending    vertical
 (10m above)         gradually     drop onto
                     as target     the target
                     gets closer
```

## Mission Results

| Mission | Instruction | Legs | Path | Time | Collisions |
|---------|------------|------|------|------|------------|
| 16 | Fly to red car → Land on rooftop | 2/2 | 166.9m | 83.4s | 0 |
| 17 | Land on rooftop near red car | 1/1 | 102.3m | 63.5s | 0 |
| 19 | Fly to car → Circle → Return | 3/3 | 123.6m | 67.6s | 0 |

All missions GPS-free. The drone navigates using only camera + text instruction.

## Project Structure

```
vla-proj/
├── airsim_benchmark/
│   ├── controllers/
│   │   ├── hybrid_controller.py       # Main pipeline: DINO + OpenFly
│   │   ├── augmented_controller.py    # Detection-only (no OpenFly)
│   │   ├── detection_controller.py    # Single-object TRACK/COAST/SEARCH
│   │   └── openfly_controller.py      # OpenFly-only (no detection)
│   ├── core/
│   │   ├── detection_inference.py     # GroundingDINO wrapper
│   │   ├── instruction_parser.py      # Text → subtasks
│   │   ├── subtask_fsm.py            # Step sequencing
│   │   ├── target_phrase.py           # Aerial-friendly query alternatives
│   │   ├── drone_fsm.py              # Flight state machine
│   │   └── airsim_client.py          # Simulator connection
│   ├── config/
│   │   └── benchmark_config.yaml      # Mission definitions
│   └── scripts/
│       ├── run_benchmark.py           # CLI entry point
│       └── audit_detection.py         # Detection rate testing
├── scripts/
│   ├── run_airsim_vla.slurm          # Run missions on cluster
│   └── run_aerial_audit.sh           # Test detection alternatives
├── docs/
│   ├── PIPELINE_HANDBOOK_2026-08-24.md  # Detailed pipeline explanation
│   └── GROUNDING_DINO_HANDBOOK_2026-08-23.md  # Detection model details
└── logs/airsim_output/                # Flight videos and metrics
```

## Quick Start

**Prerequisites:** AirSim Neighborhood binary, Python 3.10+, NVIDIA GPU (A100 recommended)

```bash
# Run the main mission (red car + rooftop landing)
sbatch --export=ALL,CONTROLLER=hybrid,RUN_MODE=mission,MISSION_IDS=16,\
GOAL_BIAS=0.0,SKIP_AIRSIM=0,RECORD_FRAMES=1 scripts/run_airsim_vla.slurm
```

Flight videos saved to `logs/airsim_output/hybrid_bias0.0/frames/mission_16/flight.mp4`

## Aerial Detection — Solving the Domain Gap

GroundingDINO was trained on ground-level internet photos. From a drone at 10-20m altitude, objects look completely different. We solved this with **aerial-friendly alternative queries**:

| Object | Original query (0% detection) | Aerial alternative | Detection rate |
|--------|------------------------------|-------------------|---------------|
| Rooftop | "rooftop" | "house top view" | **97.5%** |
| Tree | "tree" | "tree canopy" | **90.0%** |
| Garage | "garage" | "car shelter" | **92.5%** |
| Intersection | "intersection" | "road crossing" | **60.0%** |
| Mailbox | "mailbox" | "post box" | **72.5%** |

Zero training required — just smarter prompts. The system automatically tries all alternatives in a single detection pass.

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA A100-SXM4-40GB |
| CPU | 8 cores per job |
| RAM | 48 GB |
| Simulator | AirSim (Unreal Engine) |
| Cluster | Slurm-managed DGX nodes |

## License

MIT
