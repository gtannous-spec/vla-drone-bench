#!/usr/bin/env python3
"""
thesis_eval_matrix.py — Generate evaluation matrix commands for thesis.

5 controllers × 3 missions × 3 weather = 45 runs.
Prints sbatch commands (dry-run by default, does NOT submit jobs).

Usage:
    python -m airsim_benchmark.scripts.thesis_eval_matrix --dry-run
"""

import argparse
import os
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONTROLLERS = ["openfly", "openfly", "detection", "openfly", "classical"]
CONTROLLER_LABELS = ["base_openfly", "openfly_v12", "detection", "openfly_film", "oracle"]
MISSIONS = [16, 17, 18]
WEATHER = ["clear", "fog", "rain"]


def generate_eval_commands(args):
    commands = []
    for (ctrl, label), mission, weather in product(
        zip(CONTROLLERS, CONTROLLER_LABELS), MISSIONS, WEATHER
    ):
        run_name = f"{label}_m{mission}_{weather}"
        output_dir = os.path.join(args.output_dir, run_name)

        cmd_parts = [
            f"SKIP_AIRSIM=0",
            f"CONTROLLER={ctrl}",
            f"GOAL_BIAS=0.0",
            f"RUN_MODE=mission",
            f"MISSION_IDS={mission}",
            f"WEATHER={weather}",
        ]

        if label == "openfly_v12" and args.regression_head_path:
            cmd_parts.append(f"REGRESSION_HEAD_PATH={args.regression_head_path}")
        elif label == "openfly_film" and args.film_head_path:
            cmd_parts.append(f"REGRESSION_HEAD_PATH={args.film_head_path}")
        elif label == "base_openfly":
            pass

        env_str = ",".join(cmd_parts)
        output_flag = f"OUTPUT_DIR={output_dir}"
        cmd = f"sbatch --export=ALL,{env_str},{output_flag} scripts/run_airsim_vla.slurm"
        commands.append((run_name, cmd))

    return commands


def main():
    parser = argparse.ArgumentParser(description="Thesis evaluation matrix generator")
    parser.add_argument("--output-dir", default="logs/eval_matrix")
    parser.add_argument("--regression-head-path", default="", help="Path to v12 regression checkpoint")
    parser.add_argument("--film-head-path", default="", help="Path to FiLM head checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    commands = generate_eval_commands(args)

    print(f"\n{'='*60}")
    print(f"  Thesis Evaluation Matrix: {len(commands)} runs")
    print(f"{'='*60}\n")

    for name, cmd in commands:
        print(f"  {name}:")
        print(f"    {cmd}\n")

    if not args.dry_run:
        print("To submit, remove --dry-run. Commands above can also be copied individually.")


if __name__ == "__main__":
    main()
