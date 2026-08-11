#!/usr/bin/env python3
"""
collect_trajectories.py — Collect demonstration trajectories for M3 LoRA fine-tuning.

Runs the classical controller on all tasks, and at each navigation hop saves:
  - Camera image (RGB, 640x480)
  - Language instruction
  - Action taken (target waypoint delta)
  - Drone state (position, velocity, orientation)
  - Goal coordinates
  - Task metadata

Output format: JSONL file + image directory, compatible with OpenVLA/OpenFly
fine-tuning pipeline.

Usage:
    python -m airsim_benchmark.scripts.collect_trajectories \
        --config airsim_benchmark/config/benchmark_config.yaml \
        --output ./data/trajectories \
        --runs 5

This generates 500+ image-instruction-action triples from successful
classical navigation runs across all tasks.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.controllers.classical_controller import ClassicalWaypointController
from airsim_benchmark.controllers.base_controller import DroneState
from airsim_benchmark.core.airsim_client import AirSimClient
from airsim_benchmark.core.drone_fsm import DroneFSM

import yaml

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def compute_action_delta(current_pos, target_pos):
    """Compute the normalized action delta from current to target position."""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dz = target_pos[2] - current_pos[2]
    return [dx, dy, dz]


def collect_single_task(client, task_cfg, output_dir, run_id, config):
    """Run one task with classical controller, collecting data at each hop."""
    task_id = task_cfg["id"]
    instruction = task_cfg["instruction"]
    goal = task_cfg["goal"]
    start = task_cfg["start"]

    img_dir = output_dir / "images" / f"run{run_id}_task{task_id}"
    img_dir.mkdir(parents=True, exist_ok=True)

    controller = ClassicalWaypointController(
        arrival_tolerance=config.get("arrival_tolerance", 1.5),
        nav_speed=config.get("nav_speed", 5.0),
    )
    controller.reset(task_cfg)

    # Teleport to start
    sx, sy, sz = start
    client.teleport_to(sx, sy, sz)
    time.sleep(0.5)

    # Takeoff
    takeoff_alt = config.get("takeoff_altitude", -10.0)
    client.takeoff(altitude=takeoff_alt)

    samples = []
    hop = 0
    max_hops = 30
    mission_timeout = config.get("mission_timeout", 120.0)
    t_start = time.time()

    while hop < max_hops and (time.time() - t_start) < mission_timeout:
        hop += 1

        # Get current state
        pos = client.get_position()
        vel = client.get_velocity()
        ori = client.get_orientation()
        image = client.get_camera_image()

        if image is None:
            continue

        state = DroneState(position=pos, velocity=vel, orientation=ori, image=image)

        # Check if goal reached
        if controller.is_goal_reached(state):
            logger.info(f"  Task {task_id} run {run_id}: goal reached at hop {hop}")
            break

        # Get action from classical controller
        action = controller.get_action(state)
        target = action.target_position

        # Compute action delta (what the model should learn to predict)
        action_delta = compute_action_delta(pos, target)

        # Save image
        img_filename = f"hop_{hop:04d}.npy"
        img_path = img_dir / img_filename
        np.save(str(img_path), image)

        # Record sample
        sample = {
            "task_id": task_id,
            "run_id": run_id,
            "hop": hop,
            "instruction": instruction,
            "image_path": str(img_path),
            "position": list(pos),
            "velocity": list(vel),
            "orientation": list(ori),
            "goal": goal,
            "target_waypoint": list(target),
            "action_delta": action_delta,
            "distance_to_goal": math.sqrt(
                (pos[0] - goal[0])**2 + (pos[1] - goal[1])**2 + (pos[2] - goal[2])**2
            ),
            "timestamp": time.time(),
        }
        samples.append(sample)

        # Execute the action
        client.move_to(
            target[0], target[1], target[2],
            velocity=action.velocity,
            timeout_sec=10.0,
        )

    # Land
    try:
        client.land()
    except Exception:
        pass

    success = controller.is_goal_reached(state) if hop > 0 else False
    logger.info(
        f"  Task {task_id} run {run_id}: {'SUCCESS' if success else 'TIMEOUT'} "
        f"— {len(samples)} samples collected"
    )

    return samples, success


def main():
    parser = argparse.ArgumentParser(
        description="Collect demonstration trajectories for LoRA fine-tuning"
    )
    parser.add_argument(
        "--config", "-c",
        default=str(Path(__file__).parent.parent / "config" / "benchmark_config.yaml"),
        help="Path to benchmark_config.yaml",
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/trajectories",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int, default=5,
        help="Number of runs per task (default: 5)",
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+", type=int, default=None,
        help="Task IDs to collect (default: all)",
    )
    args = parser.parse_args()

    setup_logging()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tasks = config.get("tasks", [])
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in args.tasks]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "demonstrations.jsonl"

    logger.info(f"Collecting trajectories: {len(tasks)} tasks x {args.runs} runs")
    logger.info(f"Output: {output_dir}")

    client = AirSimClient()
    client.connect()

    all_samples = []
    total_success = 0
    total_runs = 0

    try:
        for run_id in range(1, args.runs + 1):
            for task_cfg in tasks:
                total_runs += 1
                try:
                    client.reset()
                    samples, success = collect_single_task(
                        client, task_cfg, output_dir, run_id, config
                    )
                    all_samples.extend(samples)
                    if success:
                        total_success += 1
                except Exception as e:
                    logger.error(f"  Run {run_id} task {task_cfg['id']} failed: {e}")
                    continue
    finally:
        client.disconnect()

    # Write JSONL
    with open(jsonl_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  COLLECTION COMPLETE")
    logger.info(f"  Total samples: {len(all_samples)}")
    logger.info(f"  Successful runs: {total_success}/{total_runs}")
    logger.info(f"  JSONL: {jsonl_path}")
    logger.info(f"  Images: {output_dir / 'images'}")
    logger.info(f"{'='*60}")

    # Write summary metadata
    meta = {
        "total_samples": len(all_samples),
        "total_runs": total_runs,
        "successful_runs": total_success,
        "tasks": [t["id"] for t in tasks],
        "runs_per_task": args.runs,
        "config": str(config_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
