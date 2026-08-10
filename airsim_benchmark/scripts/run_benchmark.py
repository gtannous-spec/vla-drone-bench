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

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

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
from airsim_benchmark.controllers.openfly_controller import OpenFlyController
from airsim_benchmark.controllers.llamauav_controller import LLaMAUAVController
from airsim_benchmark.controllers.drl_controller import DRLController
from airsim_benchmark.controllers.twostage_controller import TwoStageController
from airsim_benchmark.controllers.detection_controller import DetectionController
from airsim_benchmark.controllers.augmented_controller import AugmentedController
from airsim_benchmark.controllers.hybrid_controller import HybridController
from airsim_benchmark.runner.benchmark_runner import BenchmarkRunner


CONTROLLERS = {
    "classical": ClassicalWaypointController,
    "vla": VLAHybridController,
    "vlm": VLMController,
    "openfly": OpenFlyController,
    "llamauav": LLaMAUAVController,
    "drl": DRLController,
    "twostage": TwoStageController,
    "detection": DetectionController,
    "augmented": AugmentedController,
    "hybrid": HybridController,
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
        "--detection-model",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model ID for GroundingDINO (default: grounding-dino-base)",
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Path to LoRA adapter checkpoint for OpenFly controller",
    )
    parser.add_argument(
        "--regression-head-path",
        default=None,
        help="Path to regression head checkpoint dir (contains regression_head.pt)",
    )
    parser.add_argument(
        "--waypoint-scale",
        type=float,
        default=None,
        help="VLA waypoint scale factor (default: from config or 15.0)",
    )
    parser.add_argument(
        "--goal-bias",
        type=float,
        default=None,
        help="Goal bias for VLA/VLN controllers (0.0=pure model, 1.0=pure goal)",
    )
    parser.add_argument(
        "--mode",
        choices=["task", "mission"],
        default="task",
        help="Run mode: 'task' for single-goal tasks, 'mission' for multi-leg GPS-free missions",
    )
    parser.add_argument(
        "--missions",
        nargs="+",
        type=int,
        default=None,
        help="Mission IDs to run in mission mode (default: all)",
    )
    parser.add_argument("--weather", choices=["clear", "fog", "rain"], default="clear",
                        help="Weather condition for evaluation (default: clear)")
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

    goal_bias = args.goal_bias if args.goal_bias is not None else cfg.get("vla_goal_bias", 0.6)

    if args.controller == "openfly":
        model_path = args.model_path or cfg.get(
            "openfly_model", "IPEC-COMMUNITY/openfly-agent-7b"
        )
        controller = OpenFlyController(
            model_path=model_path,
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
            waypoint_scale=args.waypoint_scale or cfg.get("vla_waypoint_scale", 15.0),
            max_hops=cfg.get("vla_max_hops", 50),
            goal_bias=goal_bias,
            lora_path=args.lora_path or "",
            regression_head_path=args.regression_head_path or "",
        )
    elif args.controller == "vlm":
        vlm_model = args.vlm_model or cfg.get("vlm_model", "OpenGVLab/InternVL2-8B")
        controller = VLMController(
            model_path=vlm_model,
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
            waypoint_scale=cfg.get("vlm_waypoint_scale", 15.0),
            confidence_threshold=cfg.get("vlm_confidence_threshold", 0.3),
            max_hops=cfg.get("vlm_max_hops", 40),
        )
    elif args.controller == "llamauav":
        llama_uav_dir = os.path.expanduser(
            args.model_path or cfg.get("llamauav_model", "~/models/llama-uav/llama-uav-7b")
        )
        controller = LLaMAUAVController(
            model_path=llama_uav_dir,
            model_base=cfg.get("llamauav_base", "~/models/llama-uav/vicuna-7b-v1.5"),
            vision_tower=cfg.get("llamauav_vision_tower", "~/models/llama-uav/eva_vit_g.pth"),
            qformer_path=cfg.get("llamauav_qformer", "~/models/llama-uav/instruct_blip_vicuna7b_trimmed.pth"),
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
            max_hops=cfg.get("vla_max_hops", 50),
            goal_bias=goal_bias,
        )
    elif args.controller == "twostage":
        vlm_model = args.vlm_model or cfg.get("vlm_model", "OpenGVLab/InternVL2-8B")
        controller = TwoStageController(
            model_path=vlm_model,
            nav_speed=nav_speed,
            hop_distance=cfg.get("vlm_waypoint_scale", 5.0),
            max_hops=cfg.get("vlm_max_hops", 50),
        )
    elif args.controller == "detection":
        controller = DetectionController(
            model_id=args.detection_model,
            nav_speed=nav_speed,
            hop_distance=5.0,
            max_hops=cfg.get("vlm_max_hops", 50),
        )
    elif args.controller == "augmented":
        controller = AugmentedController(
            model_id=args.detection_model,
            nav_speed=nav_speed,
            hop_distance=5.0,
            max_hops=cfg.get("vlm_max_hops", 50),
        )
    elif args.controller == "hybrid":
        controller = HybridController(
            detection_model_id=args.detection_model,
            openfly_model_path=args.model_path or cfg.get(
                "openfly_model", "IPEC-COMMUNITY/openfly-agent-7b",
            ),
            nav_speed=nav_speed,
            hop_distance=5.0,
            max_hops=cfg.get("vlm_max_hops", 50),
        )
    elif args.controller == "drl":
        policy_path = args.model_path or ""
        controller = DRLController(
            policy_path=policy_path,
            arrival_tolerance=arrival_tol,
            nav_speed=nav_speed,
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
            goal_bias=goal_bias,
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
    logger.info(f"Mode: {args.mode}")
    if args.mode == "task":
        logger.info(f"Tasks: {args.tasks or 'all'}")
    else:
        logger.info(f"Missions: {args.missions or 'all'}")
    if args.record:
        logger.info(f"Recording: {args.record_fps} fps")

    runner = BenchmarkRunner(
        config_path=str(config_path),
        controller=controller,
        output_dir=args.output,
        task_ids=args.tasks,
        record_frames=args.record,
        record_fps=args.record_fps,
    )
    runner._weather = args.weather

    if args.mode == "mission":
        metrics = runner.run_missions(mission_ids=args.missions)
        run_type = "benchmark_mission"
    else:
        metrics = runner.run()
        run_type = "benchmark_task"

    # Auto-register experiment
    try:
        from airsim_benchmark.experiments import ExperimentRegistry
        registry = ExperimentRegistry()
        model_name = ""
        if args.controller == "openfly":
            model_name = args.model_path or "IPEC-COMMUNITY/openfly-agent-7b"
        elif args.controller == "llamauav":
            model_name = args.model_path or "TravelUAV/LLaMA-UAV"
        elif args.controller in ("vlm", "twostage"):
            model_name = args.vlm_model or "OpenGVLab/InternVL2-8B"
        elif args.controller in ("detection", "augmented"):
            model_name = args.detection_model or "IDEA-Research/grounding-dino-base"
        elif args.controller == "vla":
            model_name = args.model_path or "openvla/openvla-7b"
        elif args.controller == "classical":
            model_name = "classical-waypoint"

        exp_id = registry.register_benchmark(
            run_type=run_type,
            controller=args.controller,
            model=model_name,
            metrics=metrics,
            output_dir=args.output,
            goal_bias=args.goal_bias,
            waypoint_scale=args.waypoint_scale,
            lora_path=args.lora_path,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            gpu=os.environ.get("SLURM_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES")),
            node=os.environ.get("SLURM_NODELIST", os.environ.get("HOSTNAME")),
        )
        logger.info(f"Experiment registered: {exp_id}")
    except Exception as e:
        logger.warning(f"Could not register experiment: {e}")

    if run_type == "benchmark_task":
        sr = metrics.get("aggregate", {}).get("success_rate", 0.0)
        sys.exit(0 if sr == 1.0 else 1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
