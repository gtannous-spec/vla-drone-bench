# VLA Urban Drone Navigation Benchmark

A research benchmark for evaluating **Vision-Language-Action (VLA)** models on urban drone navigation tasks using natural language instructions.

The project compares three controller paradigms — classical waypoint, OpenVLA, and deep reinforcement learning — across identical language-instructed missions in simulated urban environments.

## Milestones

| # | Controller | Simulator | Status |
|---|-----------|-----------|--------|
| M1 | Classical Waypoint | AirSim (Neighborhood) | Complete |
| M2 | OpenVLA (7B) | ROS 2 Humble + Gazebo Harmonic + PX4 SITL | In Progress |
| M3 | DRL Policy | TBD | Planned |

## Project Structure

```
vla-proj/
├── airsim_benchmark/           # M1: AirSim benchmark framework
│   ├── config/                 # Task definitions (YAML) + AirSim settings
│   ├── controllers/            # Controller implementations
│   │   ├── base_controller.py  # Abstract controller interface
│   │   ├── classical_controller.py
│   │   ├── vla_controller.py   # OpenVLA stub
│   │   └── drl_controller.py   # DRL stub
│   ├── core/                   # AirSim client, FSM, telemetry, frame recorder
│   ├── runner/                 # BenchmarkRunner orchestration
│   ├── scripts/                # Entry points (run_benchmark.py, plot_flights.py)
│   └── requirements.txt
├── src/vla_navigation/         # M2: ROS 2 package
│   ├── vla_navigation/         # ROS 2 nodes (classical_planner, vla_planner)
│   ├── launch/                 # Launch files
│   ├── worlds/                 # Gazebo SDF worlds
│   └── config/                 # Benchmark config for ROS 2 track
├── scripts/                    # Slurm jobs, container build, evaluation
├── Dockerfile                  # M2 container (ROS 2 + Gazebo + PX4 + PyTorch)
└── logs/                       # Generated output (gitignored)
```

## Benchmark Tasks

Five language-instructed navigation tasks in the AirSim Neighborhood environment:

| ID | Instruction | Distance |
|----|-------------|----------|
| 1 | Fly to the red car parked near the cul-de-sac | 54.4 m |
| 2 | Navigate to the rooftop of the two-story house on the left | 86.0 m |
| 3 | Go around the block and land in the backyard behind the white house | 73.3 m |
| 4 | Fly low over the main road heading north toward the intersection | 120.1 m |
| 5 | Inspect the mailbox at the end of the driveway on the right | 58.9 m |

## Evaluation Metrics

- **SR** — Success Rate (fraction of tasks where final distance < 1.5 m)
- **FDG** — Final Distance to Goal (metres)
- **NPL** — Normalized Path Length (actual / straight-line, 1.0 = optimal)
- **Collisions** — Number of collision events
- **Constraint Violations** — Altitude/geofence breaches

## Getting Started

### M1 — AirSim Benchmark

**Prerequisites:**
- AirSim v1.8.1 Linux binary ([AirSimNH](https://github.com/microsoft/AirSim/releases) environment)
- Python 3.8+
- GPU node with Xvfb for headless rendering

**Install dependencies:**

```bash
cd airsim_benchmark
pip install numpy msgpack-rpc-python
pip install -r requirements.txt
```

**Run the benchmark:**

```bash
python -m airsim_benchmark.scripts.run_benchmark --controller classical
```

**On a Slurm cluster:**

```bash
sbatch scripts/run_airsim_benchmark.slurm
```

### M2 — ROS 2 / Gazebo / PX4 / OpenVLA

**Build the container:**

```bash
docker build -t vla-sim:m2 .
```

**Run with OpenVLA weights mounted:**

```bash
docker run --gpus all \
  -v $HOME/models/openvla-7b:/models/openvla-7b \
  vla-sim:m2
```

On a Slurm cluster with Pyxis, see `scripts/run_poc.slurm`.

## Architecture

All controllers implement a common interface (`BaseController`) with:
- `reset()` — initialize for a new task
- `get_action()` — compute next action from state/observation
- `is_goal_reached()` — check task completion

The drone operates through a 5-state FSM: **IDLE -> TAKEOFF -> NAVIGATE -> LAND -> DONE**.

## Hardware (Tested On)

- NVIDIA DGX cluster (A100-SXM4-40GB GPUs)
- 8 CPU cores, 32 GB RAM per job
- Shared NFS storage

## Baseline Results (M1 Classical Controller)

| Metric | Value |
|--------|-------|
| Success Rate | 100% (5/5) |
| Mean Final Distance | 0.466 m |
| Mean Normalized Path Length | 1.177 |
| Mean Time to Goal | 28.45 s |

Full results: [`logs/airsim_output/classical/BASELINE_REPORT.md`](logs/airsim_output/classical/BASELINE_REPORT.md)

## License

MIT
