#!/usr/bin/env python3
"""
collect_trajectories.py — Collect demonstration trajectories for LoRA fine-tuning.

Runs the intent-driven oracle on ScenarioCatalog episodes, recording at each hop:
  - 3-image keyframe triplet (PNG, matching OpenFly's vision backbone input)
  - 8-D continuous action vector from encode_displacement (includes dim5 descent)
  - Language instruction from ScenarioCatalog.sample_episode (already paraphrased)
  - Intent, landmark_id, in_fov (debug positions; no GPS goal in training records)

v4 features:
  - Landmark/scenario catalogs instead of synthetic mission goals
  - Descent (dim5) and planned-stop labels
  - End-of-run collection quality gates
  - Weather/lighting variation per episode
  - Incremental manifest writing (data survives job termination)

Output: ``data/lora_training_v4/manifest.jsonl`` + per-episode image directories.

Usage:
    python -m airsim_benchmark.scripts.collect_trajectories \\
        --output ./data/lora_training_v4 \\
        --episodes 200
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.collection.landmarks import LandmarkCatalog
from airsim_benchmark.collection.quality import evaluate_collection_gates
from airsim_benchmark.collection.scenarios import ScenarioCatalog
from airsim_benchmark.controllers.oracle_controller import (
    OracleController,
    quat_to_yaw_deg,
)
from airsim_benchmark.controllers.base_controller import DroneState
from airsim_benchmark.core.airsim_client import AirSimClient
from airsim_benchmark.core.move_plan import is_stuck_hop, should_call_airsim_land

logger = logging.getLogger(__name__)

_APPROACH_INTENTS = {"search_then_approach", "land_on_surface"}

WEATHER_PRESETS = [
    {"rain": 0.0, "fog": 0.0, "dust": 0.0},
    {"rain": 0.2, "fog": 0.0, "dust": 0.0},
    {"rain": 0.0, "fog": 0.3, "dust": 0.0},
    {"rain": 0.0, "fog": 0.0, "dust": 0.2},
    {"rain": 0.1, "fog": 0.1, "dust": 0.0},
]

TIME_OF_DAY_HOURS = [8, 10, 12, 14, 16, 18]

_DEFAULT_CONFIG = {
    "mission_timeout": 180,
    "takeoff_altitude": -10.0,
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def save_image_png(img: np.ndarray, path: str) -> None:
    from PIL import Image as PILImage
    PILImage.fromarray(img).save(path)


def compute_continuous_action(
    pos_before: Tuple[float, ...],
    pos_after: Tuple[float, ...],
    yaw_before_deg: float,
    yaw_after_deg: float,
    at_goal: bool = False,
) -> List[float]:
    """Build an 8-D continuous action from actual displacement (dim5 = descent)."""
    from airsim_benchmark.core.action_space import encode_displacement
    return encode_displacement(
        pos_before,
        pos_after,
        yaw_before_deg,
        yaw_after_deg,
        at_goal=at_goal,
    ).tolist()


def randomize_environment(client: AirSimClient, enable_weather: bool = True,
                          enable_time: bool = True) -> dict:
    """Randomize weather and time of day for visual diversity."""
    env_info = {}
    if enable_weather:
        preset = random.choice(WEATHER_PRESETS)
        client.set_weather(**preset)
        env_info["weather"] = preset
    if enable_time:
        hour = random.choice(TIME_OF_DAY_HOURS)
        client.set_time_of_day(hour)
        env_info["time_of_day"] = hour
    return env_info


def collect_episode(
    client: AirSimClient,
    controller: OracleController,
    task_cfg: dict,
    episode_id: int,
    output_dir: Path,
    config: dict,
    detector=None,
) -> Tuple[List[dict], bool]:
    """Run one episode and return (samples, success)."""
    # Scenario sampling already paraphrases; do not paraphrase again.
    instruction = task_cfg["instruction"]
    intent = task_cfg.get("intent", "")
    landmark_id = task_cfg.get("landmark_id")
    start = task_cfg.get("start", [0, 0, -10])

    ep_dir = output_dir / f"episode_{episode_id:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    controller.reset(task_cfg)

    sx, sy, sz = start
    yaw0 = task_cfg.get("start_yaw", 0.0)
    client.teleport_to(sx, sy, sz, yaw_deg=yaw0)
    time.sleep(0.5)

    # Already at cruise z from teleport — do not takeoffAsync (drops through roofs).
    client.hold_altitude(sz)

    keyframes: List[np.ndarray] = []
    samples: List[dict] = []
    hop = 0
    stuck_hops = 0
    max_hops = task_cfg.get("max_hops", 50)
    timeout = task_cfg.get("timeout", config.get("mission_timeout", 180))
    t_start = time.time()
    state: Optional[DroneState] = None

    while hop < max_hops and (time.time() - t_start) < timeout:
        pos = client.get_position()
        vel = client.get_velocity()
        ori = client.get_orientation()
        image = client.get_camera_image()

        if image is None:
            time.sleep(0.2)
            continue

        hop += 1

        state = DroneState(position=pos, velocity=vel, orientation=ori, image=image)

        if controller.is_goal_reached(state):
            logger.info(f"  Episode {episode_id}: goal reached at hop {hop}")
            break

        if hop % 3 == 1 or len(keyframes) == 0:
            keyframes.append(image.copy())

        if len(keyframes) >= 2:
            triplet = [keyframes[-2], keyframes[-1], image]
        elif len(keyframes) == 1:
            triplet = [keyframes[0], image, image]
        else:
            triplet = [image, image, image]

        yaw_before = quat_to_yaw_deg(*ori)
        action_result = controller.get_action(state)
        action_id = controller.last_action_id

        img_paths = []
        for img_idx, img in enumerate(triplet):
            fname = f"step_{hop:04d}_img{img_idx}.png"
            fpath = ep_dir / fname
            save_image_png(img, str(fpath))
            img_paths.append(str(fpath.relative_to(output_dir)))

        tgt = action_result.target_position
        client.move_to(tgt[0], tgt[1], tgt[2],
                       velocity=action_result.velocity, timeout_sec=10.0)

        pos_after = client.get_position()
        ori_after = client.get_orientation()
        yaw_after = quat_to_yaw_deg(*ori_after)

        planned_stop = float(controller.last_action_vec[0]) >= 1.0
        # Planned stop: write the stop vector even if AirSim hover-jittered.
        action_vec = compute_continuous_action(
            pos, pos_after, yaw_before, yaw_after, at_goal=planned_stop,
        )

        detection_meta = {}
        if detector is not None and landmark_id:
            try:
                from airsim_benchmark.core.target_phrase import extract_target
                target_info = extract_target(instruction)
                dets = detector.detect(image, target_info.phrase)
                best = dets[0] if dets else None
                detection_meta = {
                    "in_fov_detected": best is not None and best.score >= 0.25,
                    "detection_score": float(best.score) if best else 0.0,
                    "detection_bbox": list(best.bbox_xyxy) if best else None,
                }
            except Exception as e:
                logger.debug(f"Detection verify failed: {e}")

        sample = {
            "episode": episode_id,
            "step": hop,
            "instruction": instruction,
            "images": img_paths,
            "action": action_vec,
            "action_id": action_id,
            "position": list(pos),
            "heading_deg": round(yaw_before, 2),
            "velocity": list(vel),
            "intent": intent,
            "landmark_id": landmark_id,
            "in_fov": bool(controller.last_in_fov),
        }
        sample["in_fov_geometry"] = sample["in_fov"]
        sample.update(detection_meta)
        samples.append(sample)

        if is_stuck_hop(pos, pos_after, planned_stop):
            stuck_hops += 1
            collided = False
            try:
                collided = bool(client.has_collided())
            except Exception:
                pass
            logger.warning(
                "  Episode %s hop %s stuck (moved <0.2 m)%s",
                episode_id, hop, " collision" if collided else "",
            )
            if stuck_hops >= 3:
                logger.warning(
                    "  Episode %s: aborting after 3 stuck hops", episode_id,
                )
                break
        else:
            stuck_hops = 0

    if intent == "land_ground":
        try:
            z_now = client.get_position()[2]
        except Exception:
            z_now = -10.0
        if should_call_airsim_land(z_now):
            try:
                client.land()
            except Exception:
                pass
        else:
            logger.warning(
                "  Episode %s: skip AirSim land() (still high, z=%.1f)",
                episode_id, z_now,
            )

    success = controller.is_goal_reached(state) if state is not None else False
    logger.info(
        f"  Episode {episode_id}: {'SUCCESS' if success else 'TIMEOUT'} "
        f"— {len(samples)} samples"
    )
    return samples, success


def _task_cfg_from_spec(ep: int, spec) -> dict:
    return {
        "id": ep,
        "instruction": spec.instruction,
        "intent": spec.intent,
        "landmark_id": spec.landmark_id,
        "start": list(spec.start),
        "start_yaw": spec.start_yaw,
        "max_hops": spec.max_hops,
        "landmark_position": spec.landmark_position,
        "landmark_radius": spec.landmark_radius,
        "surface_z": spec.surface_z,
        "target_alt_ned": spec.target_alt_ned,
        "constraints": {
            "min_altitude": spec.min_altitude,
            "max_altitude": spec.max_altitude,
        },
        "timeout": 180,
    }


def _gate_inputs_from_manifest(manifest_path: Path):
    actions: List[List[float]] = []
    instructions: List[str] = []
    in_fov: List[bool] = []
    approach_mask: List[bool] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            actions.append(rec["action"])
            instructions.append(rec.get("instruction", ""))
            in_fov.append(bool(rec.get("in_fov", False)))
            approach_mask.append(rec.get("intent") in _APPROACH_INTENTS)
    if actions:
        actions_np = np.asarray(actions, dtype=float)
    else:
        actions_np = np.zeros((0, 8), dtype=float)
    return actions_np, instructions, in_fov, approach_mask


def main() -> None:
    cfg_dir = Path(__file__).resolve().parent.parent / "config"
    parser = argparse.ArgumentParser(
        description="Collect LoRA-ready trajectories with keyframe triplets"
    )
    parser.add_argument(
        "--config", "-c",
        default=str(cfg_dir / "benchmark_config.yaml"),
        help="Unused (kept for CLI compatibility). Collection is catalog-driven.",
    )
    parser.add_argument("--output", "-o", default="./data/lora_training_v4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--landmarks",
        default=str(cfg_dir / "landmarks.yaml"),
    )
    parser.add_argument(
        "--scenarios",
        default=str(cfg_dir / "training_scenarios.yaml"),
    )
    parser.add_argument("--task-runs", type=int, default=10,
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--mission-runs", type=int, default=5,
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--tasks", nargs="+", type=int, default=None,
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--missions", nargs="+", type=int, default=None,
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--skip-tasks", action="store_true",
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--skip-missions", action="store_true",
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--no-weather", action="store_true",
                        help="Disable weather/lighting randomization")
    parser.add_argument("--no-prompt-diversity", action="store_true",
                        help="Disable instruction paraphrasing in scenario sampling")
    parser.add_argument("--perturb-prob", type=float, default=0.25,
                        help="Probability of injecting perturbations (0=disabled)")
    parser.add_argument("--no-detection-verify", action="store_true",
                        help="Skip GroundingDINO verification during collection")
    parser.add_argument("--detection-model", default="IDEA-Research/grounding-dino-base",
                        help="Detection model for data verification")
    args = parser.parse_args()

    setup_logging()

    config = dict(_DEFAULT_CONFIG)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    enable_weather = not args.no_weather
    enable_prompts = not args.no_prompt_diversity

    landmarks = LandmarkCatalog.load(Path(args.landmarks))
    scenarios = ScenarioCatalog.load(Path(args.scenarios))

    controller = OracleController(
        nav_speed=5.0, hop_distance=15.0,
        perturb_prob=args.perturb_prob,
    )

    client = AirSimClient()
    client.connect()

    detector = None
    if not args.no_detection_verify:
        try:
            from airsim_benchmark.core.detection_inference import ObjectDetector
            detector = ObjectDetector(
                model_id=args.detection_model,
                device="cuda:0",
            )
            logger.info("Detection verifier loaded for data quality checks")
        except Exception as e:
            logger.warning(f"Could not load detection verifier: {e}")

    manifest_file = open(manifest_path, "a")
    total_samples = 0
    total_success = 0
    total_runs = 0

    def _flush_samples(samples: List[dict]) -> None:
        nonlocal total_samples
        for s in samples:
            manifest_file.write(json.dumps(s) + "\n")
        manifest_file.flush()
        total_samples += len(samples)

    try:
        logger.info(
            "Collecting %s catalog episodes (paraphrase=%s weather=%s)",
            args.episodes, enable_prompts, enable_weather,
        )
        for ep in range(args.episodes):
            spec = scenarios.sample_episode(
                landmarks,
                rng_seed=ep,
                paraphrase=enable_prompts,
            )
            task_cfg = _task_cfg_from_spec(ep, spec)
            total_runs += 1
            try:
                client.reset()
                if enable_weather:
                    randomize_environment(client)
                samples, success = collect_episode(
                    client, controller, task_cfg,
                    ep, output_dir, config,
                    detector=detector,
                )
                _flush_samples(samples)
                if success:
                    total_success += 1
            except Exception as e:
                logger.error("  Episode %s failed: %s", ep, e)
    finally:
        manifest_file.close()
        client.disconnect()

    all_actions = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_actions.append(json.loads(line)["action"])

    unique_actions = len({
        tuple(round(v, 2) for v in a) for a in all_actions
    })
    if all_actions:
        actions_np = np.array(all_actions)
        dim_stats = []
        for d in range(8):
            vals = actions_np[:, d]
            nonzero_pct = 100 * np.count_nonzero(vals) / len(vals)
            dim_stats.append(
                f"  dim[{d}]: min={vals.min():.3f}  max={vals.max():.3f}  "
                f"mean={vals.mean():.3f}  nonzero={nonzero_pct:.1f}%"
            )
    else:
        dim_stats = []

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  COLLECTION COMPLETE")
    logger.info(f"  Total episodes: {args.episodes}")
    logger.info(f"  Total samples:  {total_samples}")
    logger.info(f"  Successful:     {total_success}/{total_runs}")
    logger.info(f"  Unique action patterns (rounded): {unique_actions}")
    logger.info(f"  Weather variation: {'enabled' if enable_weather else 'disabled'}")
    logger.info(f"  Prompt diversity:  {'enabled' if enable_prompts else 'disabled'}")
    logger.info(f"  Perturbation prob: {args.perturb_prob}")
    for line in dim_stats:
        logger.info(line)
    logger.info(f"  Manifest:       {manifest_path}")
    logger.info(f"{'=' * 60}")

    meta = {
        "total_samples": total_samples,
        "total_episodes": args.episodes,
        "total_runs": total_runs,
        "successful_runs": total_success,
        "unique_actions": unique_actions,
        "weather_enabled": enable_weather,
        "prompt_diversity_enabled": enable_prompts,
        "perturb_prob": args.perturb_prob,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    actions_np, instructions, in_fov, approach_mask = _gate_inputs_from_manifest(
        manifest_path
    )
    ok, report = evaluate_collection_gates(
        actions_np, instructions, in_fov, approach_mask=approach_mask,
    )
    logger.info("Quality gate report: %s", json.dumps(report, indent=2))
    logger.info("Quality gates ok=%s (episodes=%s)", ok, args.episodes)
    if not ok:
        if args.episodes >= 50:
            logger.error("Quality gates failed with episodes>=50; exiting 1.")
            sys.exit(1)
        logger.warning(
            "Quality gates failed (smoke run, episodes=%s); not exiting.",
            args.episodes,
        )


if __name__ == "__main__":
    main()
