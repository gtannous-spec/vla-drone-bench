#!/usr/bin/env python3
"""
run_benchmark.py — CLI entry point for the AirSim navigation benchmark.

Usage:
    python -m airsim_benchmark.scripts.run_benchmark \
        --config airsim_benchmark/config/benchmark_config.yaml \
        --output ./output \
        --controller classical \
        --tasks 1 2 3 4 5

Prerequisites:
    - AirSim simulator running with the Neighborhood environment
    - settings.json from airsim_benchmark/config/ placed in ~/Documents/AirSim/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.controllers.classical_controller import ClassicalWaypointController
from airsim_benchmark.controllers.vla_controller import VLAHybridController
from airsim_benchmark.controllers.vlm_controller import VLMController
from airsim_benchmark.runner.benchmark_runner import BenchmarkRunner


CONTROLLERS = {
    "classical": ClassicalWaypointController,
    "vla": VLAHybridController,
    "vlm": VLMController,
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="AirSim Urban Drone Navigation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default=str(Path(__file__).parent.parent / "config" / "benchmark_config.yaml"),
        help="Path to benchmark_config.yaml",
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory for trajectories and metrics",
    )
    parser.add_argument(
        "--controller",
        choices=list(CONTROLLERS.keys()),
        default="classical",
        help="Controller type to use (default: classical)",
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        type=int,
        default=None,
        help="Task IDs to run (default: all tasks in config)",
    )
    parser.add_argument(
        "--arrival-tolerance",
        type=float,
        default=None,
        help="Override arrival tolerance from config (metres)",
    )
    parser.add_argument(
        "--nav-speed",
        type=float,
        default=None,
        help="Override navigation speed from config (m/s)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record camera frames during flight (saves frames + mp4 per task)",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=5.0,
        help="Frame capture rate when recording (default: 5 fps)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to VLA model weights (required for --controller vla)",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="HuggingFace model ID or local path for VLM (default: from config)",
    )
    parser.add_argument(
        "--waypoint-scale",
        type=float,
        default=None,
        help="VLA waypoint scale factor (default: from config or 15.0)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("benchmark")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Instantiate controller
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    arrival_tol = args.arrival_tolerance or cfg.get("arrival_tolerance", 1.5)
    nav_speed = args.nav_speed or cfg.get("nav_speed", 5.0)

    if args.controller == "vlm":
        vlm_model = args.vlm_model or cfg.get("vlm_model", "OpenGVLab/InternVL2-8B")
        controller = VLMController(
            model_path=vlm_model,
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
            waypoint_scale=cfg.get("vlm_waypoint_scale", 15.0),
            confidence_threshold=cfg.get("vlm_confidence_threshold", 0.3),
            max_hops=cfg.get("vlm_max_hops", 40),
        )
    elif args.controller == "vla":
        model_path = args.model_path or os.path.expanduser("~/models/openvla-7b")
        if not Path(model_path).exists():
            logger.error(f"VLA model not found at: {model_path}")
            logger.error("Download with: python -m airsim_benchmark.scripts.download_model")
            sys.exit(1)
        waypoint_scale = args.waypoint_scale or cfg.get("vla_waypoint_scale", 15.0)
        controller = VLAHybridController(
            model_path=model_path,
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
            waypoint_scale=waypoint_scale,
            convergence_threshold=cfg.get("vla_convergence_threshold", 0.005),
            max_hops=cfg.get("vla_max_hops", 50),
            min_hops_before_convergence=cfg.get("vla_min_hops_before_convergence", 8),
            goal_bias=cfg.get("vla_goal_bias", 0.3),
            fallback_to_coords=cfg.get("vla_fallback_to_coords", True),
        )
    else:
        ControllerClass = CONTROLLERS[args.controller]
        controller = ControllerClass(
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
        )

    logger.info(f"Controller: {args.controller}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Tasks: {args.tasks or 'all'}")
    if args.record:
        logger.info(f"Recording: {args.record_fps} fps")

    # Run benchmark
    runner = BenchmarkRunner(
        config_path=str(config_path),
        controller=controller,
        output_dir=args.output,
        task_ids=args.tasks,
        record_frames=args.record,
        record_fps=args.record_fps,
    )

    metrics = runner.run()

    # Return exit code based on success rate
    sr = metrics.get("aggregate", {}).get("success_rate", 0.0)
    sys.exit(0 if sr == 1.0 else 1)


if __name__ == "__main__":
    main()
