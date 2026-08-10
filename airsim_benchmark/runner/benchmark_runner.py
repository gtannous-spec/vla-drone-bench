"""
benchmark_runner.py — Orchestrates the full benchmark across all tasks.

Reads benchmark_config.yaml, iterates over tasks, runs the FSM with the
selected controller, records trajectories, and computes evaluation metrics.
"""

import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from ..controllers.base_controller import BaseController
from ..core.airsim_client import AirSimClient
from ..core.drone_fsm import DroneFSM, FlightPhase, should_ground_land
from ..core.frame_recorder import FrameRecorder
from ..core.telemetry import TelemetryThread

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs the navigation benchmark end-to-end.

    Args:
        config_path: Path to benchmark_config.yaml.
        controller: An instantiated BaseController.
        output_dir: Directory for trajectory CSVs and metrics.json.
        task_ids: Optional list of task IDs to run (default: all).
    """

    def __init__(
        self,
        config_path: str,
        controller: BaseController,
        output_dir: str = "./output",
        task_ids: Optional[List[int]] = None,
        record_frames: bool = False,
        record_fps: float = 5.0,
    ):
        self._config_path = config_path
        self._controller = controller
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._record_frames = record_frames
        self._record_fps = record_fps

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        all_tasks = self._config.get("tasks", [])
        if task_ids:
            self._tasks = [t for t in all_tasks if t["id"] in task_ids]
        else:
            self._tasks = all_tasks

        self._missions = self._config.get("missions", [])

        self._arrival_tolerance = self._config.get("arrival_tolerance", 1.5)
        self._takeoff_altitude = self._config.get("takeoff_altitude", -5.0)
        self._nav_speed = self._config.get("nav_speed", 5.0)
        self._mission_timeout = self._config.get("mission_timeout", 90.0)
        self._mission_timeout_long = self._config.get("mission_timeout_long", 600.0)
        self._telemetry_rate = self._config.get("telemetry_rate_hz", 10.0)

        self._results: List[dict] = []

    def _set_weather(self, client):
        """Set AirSim weather for evaluation."""
        weather = getattr(self, '_weather', 'clear')
        if weather == 'clear':
            return
        try:
            import airsim
            client.client.simEnableWeather(True)
            if weather == 'fog':
                client.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.6)
            elif weather == 'rain':
                client.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.5)
                client.client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0.3)
            logger.info(f"Weather set to: {weather}")
        except Exception as e:
            logger.warning(f"Could not set weather: {e}")

    def run(self) -> dict:
        """Execute the benchmark: connect, run all tasks, compute metrics.

        Returns:
            Dictionary with 'aggregate' and 'tasks' keys (same as metrics.json).
        """
        client = AirSimClient()
        logger.info("Connecting main AirSim client...")
        client.connect()
        logger.info("Main AirSim client connected successfully")
        self._set_weather(client)

        try:
            for task_cfg in self._tasks:
                result = self._run_task(client, task_cfg)
                self._results.append(result)
        finally:
            client.disconnect()

        metrics = self._compute_all_metrics()
        self._write_metrics(metrics)
        self._print_summary(metrics)
        return metrics

    def run_missions(self, mission_ids: Optional[List[int]] = None) -> dict:
        """Execute multi-leg GPS-free missions.

        Each mission has multiple legs with per-leg instructions. The drone
        navigates using only camera images and language prompts (no GPS coords).
        """
        client = AirSimClient()
        logger.info("Connecting main AirSim client...")
        client.connect()
        logger.info("Main AirSim client connected successfully")
        self._set_weather(client)

        missions = self._missions
        if mission_ids:
            missions = [m for m in missions if m["id"] in mission_ids]

        try:
            for mission_cfg in missions:
                result = self._run_mission(client, mission_cfg)
                self._results.append(result)
        finally:
            client.disconnect()

        metrics = self._compute_mission_metrics()
        self._write_metrics(metrics)
        self._print_mission_summary(metrics)
        return metrics

    def _run_mission(self, client: AirSimClient, mission_cfg: dict) -> dict:
        """Run a multi-leg GPS-free mission."""
        mission_id = mission_cfg["id"]
        legs = mission_cfg.get("legs", [])
        timeout = mission_cfg.get("timeout", self._mission_timeout_long)
        constraints = mission_cfg.get("constraints", {})

        logger.info(f"\n{'='*60}")
        logger.info(f"  Mission {mission_id}: \"{mission_cfg['name']}\"")
        logger.info(f"  Description: {mission_cfg['description']}")
        logger.info(f"  Legs: {len(legs)}, timeout: {timeout}s")
        logger.info(f"{'='*60}")

        sx, sy, sz = mission_cfg["start"]
        client.teleport_to(sx, sy, sz)

        telemetry = TelemetryThread(
            vehicle_name=client.vehicle_name, rate_hz=self._telemetry_rate
        )

        recorder = None
        if self._record_frames:
            frames_dir = self._output_dir / "frames" / f"mission_{mission_id}"
            recorder = FrameRecorder(
                output_dir=str(frames_dir),
                vehicle_name=client.vehicle_name,
                fps=self._record_fps,
            )

        needs_images = (
            hasattr(self._controller, '_vla')
            or hasattr(self._controller, '_vlm')
            or hasattr(self._controller, '_model')
            or hasattr(self._controller, '_detector')
        )

        leg_results = []
        mission_start = time.time()
        all_actions = []

        # Execute legs sequentially, reusing FSM for each leg
        for leg_idx, leg in enumerate(legs):
            elapsed_so_far = time.time() - mission_start
            remaining_time = timeout - elapsed_so_far
            if remaining_time <= 5:
                logger.warning(f"  Mission timeout — skipping remaining legs")
                break

            leg_instruction = leg["instruction"]
            leg_max_hops = leg.get("max_hops", self._config.get("mission_default_hops_per_leg", 25))

            logger.info(f"\n  --- Leg {leg_idx + 1}/{len(legs)}: \"{leg_instruction}\" "
                        f"(max {leg_max_hops} hops, {remaining_time:.0f}s left) ---")

            leg_config = {
                "id": mission_id,
                "instruction": leg_instruction,
                "constraints": constraints,
                "max_hops": leg_max_hops,
            }
            self._controller.reset(leg_config)

            leg_timeout = min(remaining_time, leg_max_hops * 10)

            if leg_idx == 0:
                fsm = DroneFSM(
                    client=client,
                    controller=self._controller,
                    telemetry=telemetry,
                    takeoff_altitude=self._takeoff_altitude,
                    mission_timeout=leg_timeout,
                    nav_speed=self._nav_speed,
                    capture_images=needs_images,
                )
                if recorder:
                    logger.info(f"  Starting FrameRecorder (mission {mission_id})...")
                    recorder.start()
                leg_start = time.time()
                success = fsm.execute(land_at_end=False)
                leg_elapsed = time.time() - leg_start
            else:
                fsm._controller = self._controller
                fsm._mission_timeout = leg_timeout
                fsm._phase = FlightPhase.NAVIGATE
                leg_start = time.time()
                success = fsm._run_navigate_phase()
                leg_elapsed = time.time() - leg_start

            leg_traj = telemetry.get_trajectory()
            leg_results.append({
                "leg": leg_idx + 1,
                "instruction": leg_instruction,
                "hops": self._controller._hop_count if hasattr(self._controller, '_hop_count') else 0,
                "model_guided": self._controller._model_guided_hops if hasattr(self._controller, '_model_guided_hops') else 0,
                "time_s": round(leg_elapsed, 1),
                "success": success,
            })

        total_elapsed = time.time() - mission_start

        if leg_results:
            last_instruction = leg_results[-1]["instruction"]
            if should_ground_land(last_instruction):
                try:
                    client.land()
                except Exception as e:
                    logger.warning("  Ground landing after last leg failed: %s", e)
            else:
                try:
                    client.hover()
                except Exception as e:
                    logger.warning("  Hover after last leg failed: %s", e)

        telemetry.stop()
        if recorder:
            recorder.stop()
            video_path = recorder.make_video(cleanup_frames=False)
            if video_path:
                logger.info(f"  Mission video: {video_path}")

        trajectory = telemetry.get_trajectory()
        collisions = telemetry.get_collisions()

        csv_path = self._write_trajectory_csv(mission_id, trajectory)

        path_length = self._compute_path_length(trajectory)
        heading_changes = self._compute_heading_changes(trajectory)

        result = {
            "mission_id": mission_id,
            "name": mission_cfg["name"],
            "description": mission_cfg["description"],
            "num_legs": len(legs),
            "legs_completed": len(leg_results),
            "leg_results": leg_results,
            "total_time_s": round(total_elapsed, 1),
            "path_length": round(path_length, 1),
            "collision_count": len(collisions),
            "trajectory_points": len(trajectory),
            "heading_smoothness": round(float(np.std(heading_changes)), 4) if heading_changes else 0.0,
            "mean_heading_change": round(float(np.mean(np.abs(heading_changes))), 4) if heading_changes else 0.0,
            "csv_path": str(csv_path),
        }

        total_hops = sum(lr["hops"] for lr in leg_results)
        total_guided = sum(lr["model_guided"] for lr in leg_results)
        logger.info(f"  Mission {mission_id} complete: {len(leg_results)}/{len(legs)} legs, "
                    f"path={path_length:.1f}m, time={total_elapsed:.1f}s, "
                    f"hops={total_guided}/{total_hops} model-guided, "
                    f"collisions={len(collisions)}")

        return result

    def _compute_path_length(self, trajectory: list) -> float:
        path_length = 0.0
        prev = None
        for rec in trajectory:
            if rec.phase in {"TAKEOFF", "NAVIGATE"}:
                if prev is not None and prev.phase in {"TAKEOFF", "NAVIGATE"}:
                    dx = rec.x - prev.x
                    dy = rec.y - prev.y
                    dz = rec.z - prev.z
                    path_length += math.sqrt(dx * dx + dy * dy + dz * dz)
            prev = rec
        return path_length

    def _compute_heading_changes(self, trajectory: list) -> list:
        """Compute heading change (degrees) between consecutive NAVIGATE points."""
        nav_points = [(rec.x, rec.y) for rec in trajectory if rec.phase == "NAVIGATE"]
        if len(nav_points) < 3:
            return []
        changes = []
        for i in range(1, len(nav_points) - 1):
            dx1 = nav_points[i][0] - nav_points[i - 1][0]
            dy1 = nav_points[i][1] - nav_points[i - 1][1]
            dx2 = nav_points[i + 1][0] - nav_points[i][0]
            dy2 = nav_points[i + 1][1] - nav_points[i][1]
            h1 = math.atan2(dy1, dx1)
            h2 = math.atan2(dy2, dx2)
            delta = math.degrees(h2 - h1)
            while delta > 180:
                delta -= 360
            while delta < -180:
                delta += 360
            changes.append(delta)
        return changes

    def _compute_mission_metrics(self) -> dict:
        n = len(self._results)
        if n == 0:
            return {"aggregate": {}, "missions": []}

        total_path = sum(r["path_length"] for r in self._results)
        total_collisions = sum(r["collision_count"] for r in self._results)
        total_legs = sum(r["num_legs"] for r in self._results)
        completed_legs = sum(r["legs_completed"] for r in self._results)
        smoothness_vals = [r["heading_smoothness"] for r in self._results if r["heading_smoothness"] > 0]

        aggregate = {
            "num_missions": n,
            "total_legs": total_legs,
            "completed_legs": completed_legs,
            "total_path_length_m": round(total_path, 1),
            "total_collisions": total_collisions,
            "mean_heading_smoothness": round(float(np.mean(smoothness_vals)), 4) if smoothness_vals else 0.0,
            "mean_path_per_mission": round(total_path / n, 1),
        }
        return {"aggregate": aggregate, "missions": self._results}

    def _print_mission_summary(self, metrics: dict) -> None:
        agg = metrics["aggregate"]
        missions = metrics["missions"]

        print(f"\n{'='*80}")
        print(f"  GPS-FREE MISSION RESULTS — {agg['num_missions']} missions")
        print(f"{'='*80}")
        print(f"{'Mission':>8}  {'Name':<30}  {'Legs':>5}  {'Path(m)':>8}  "
              f"{'Time(s)':>8}  {'Collisions':>10}  {'Smoothness':>10}")
        print(f"{'-'*80}")

        for m in missions:
            print(f"{m['mission_id']:>8}  {m['name']:<30}  "
                  f"{m['legs_completed']}/{m['num_legs']:>3}  "
                  f"{m['path_length']:>8.1f}  "
                  f"{m['total_time_s']:>8.1f}  "
                  f"{m['collision_count']:>10}  "
                  f"{m['heading_smoothness']:>10.4f}")

        print(f"{'-'*80}")
        print(f"  Total path: {agg['total_path_length_m']:.1f}m  "
              f"Legs: {agg['completed_legs']}/{agg['total_legs']}  "
              f"Collisions: {agg['total_collisions']}  "
              f"Smoothness: {agg['mean_heading_smoothness']:.4f}")
        print(f"{'='*80}\n")

    def _run_task(self, client: AirSimClient, task_cfg: dict) -> dict:
        """Run a single task through the FSM."""
        task_id = task_cfg["id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"  Task {task_id}: \"{task_cfg['instruction']}\"")
        logger.info(f"  Start: {task_cfg['start']} → Goal: {task_cfg['goal']}")
        logger.info(f"{'='*60}")

        # Teleport drone to start position
        sx, sy, sz = task_cfg["start"]
        client.teleport_to(sx, sy, sz)

        # Reset controller for this task
        self._controller.reset(task_cfg)

        # Create telemetry (uses its own AirSim client connection)
        telemetry = TelemetryThread(
            vehicle_name=client.vehicle_name, rate_hz=self._telemetry_rate
        )

        # Frame recorder (optional)
        recorder = None
        if self._record_frames:
            frames_dir = self._output_dir / "frames" / f"task_{task_id}"
            recorder = FrameRecorder(
                output_dir=str(frames_dir),
                vehicle_name=client.vehicle_name,
                fps=self._record_fps,
            )

        needs_images = (
            hasattr(self._controller, '_vla')
            or hasattr(self._controller, '_vlm')
            or hasattr(self._controller, '_model')
            or hasattr(self._controller, '_detector')
        )
        fsm = DroneFSM(
            client=client,
            controller=self._controller,
            telemetry=telemetry,
            takeoff_altitude=self._takeoff_altitude,
            mission_timeout=self._mission_timeout,
            nav_speed=self._nav_speed,
            capture_images=needs_images,
        )

        # Start recorder just before execution
        if recorder:
            logger.info(f"  Starting FrameRecorder (task {task_id})...")
            recorder.start()
            logger.info(f"  FrameRecorder started (task {task_id})")

        logger.info(f"  Starting FSM execution (task {task_id})...")
        t_start = time.time()
        success = fsm.execute()
        t_elapsed = time.time() - t_start

        # Stop telemetry and recorder, collect data
        telemetry.stop()
        if recorder:
            recorder.stop()
            video_path = recorder.make_video(cleanup_frames=False)
            if video_path:
                logger.info(f"  Flight video: {video_path}")

        trajectory = telemetry.get_trajectory()
        collisions = telemetry.get_collisions()

        # Write trajectory CSV
        csv_path = self._write_trajectory_csv(task_id, trajectory)

        # Compute per-task metrics
        result = self._compute_task_metrics(
            task_cfg, trajectory, collisions, success, t_elapsed
        )
        result["csv_path"] = str(csv_path)

        status = "PASS" if result["success"] else "FAIL"
        logger.info(f"  Result: {status} — distance={result['final_distance_to_goal']:.2f}m, "
                    f"path={result['path_length']:.1f}m, time={t_elapsed:.1f}s, "
                    f"collisions={result['collision_count']}")

        return result

    def _write_trajectory_csv(self, task_id: int, trajectory: list) -> Path:
        """Write trajectory to CSV in the same format as the Gazebo PoC."""
        traj_dir = self._output_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        csv_path = traj_dir / f"task_{task_id}_trajectory.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "z", "phase"])
            if trajectory:
                t0 = trajectory[0].timestamp
                for rec in trajectory:
                    writer.writerow([
                        f"{rec.timestamp - t0:.4f}",
                        f"{rec.x:.4f}",
                        f"{rec.y:.4f}",
                        f"{rec.z:.4f}",
                        rec.phase,
                    ])

        logger.info(f"  Trajectory: {csv_path} ({len(trajectory)} points)")
        return csv_path

    def _compute_task_metrics(
        self,
        task_cfg: dict,
        trajectory: list,
        collisions: list,
        success: bool,
        elapsed: float,
    ) -> dict:
        """Compute per-task evaluation metrics."""
        goal = task_cfg["goal"]
        start = task_cfg["start"]
        constraints = task_cfg.get("constraints", {})

        if not trajectory:
            return {
                "task_id": task_cfg["id"],
                "instruction": task_cfg["instruction"],
                "goal": goal,
                "success": False,
                "final_distance_to_goal": float("inf"),
                "path_length": 0.0,
                "straight_line_distance": 0.0,
                "normalized_path_length": float("inf"),
                "time_to_goal_s": elapsed,
                "collision_count": len(collisions),
                "constraint_violations": 0,
            }

        # Final distance: minimum distance to goal during NAVIGATE phase
        # Final distance: use the drone's ACTUAL last position, not min-along-path
        last = trajectory[-1]
        final_dist = math.sqrt(
            (last.x - goal[0]) ** 2
            + (last.y - goal[1]) ** 2
            + (last.z - goal[2]) ** 2
        )

        # Also track closest approach for reference
        min_dist = float("inf")
        for rec in trajectory:
            if rec.phase == "NAVIGATE":
                d = math.sqrt(
                    (rec.x - goal[0]) ** 2
                    + (rec.y - goal[1]) ** 2
                    + (rec.z - goal[2]) ** 2
                )
                min_dist = min(min_dist, d)
        if min_dist == float("inf"):
            min_dist = final_dist

        # Path length (TAKEOFF + NAVIGATE phases)
        path_length = 0.0
        active_phases = {"TAKEOFF", "NAVIGATE"}
        prev = None
        for rec in trajectory:
            if rec.phase in active_phases:
                if prev is not None and prev.phase in active_phases:
                    dx = rec.x - prev.x
                    dy = rec.y - prev.y
                    dz = rec.z - prev.z
                    path_length += math.sqrt(dx * dx + dy * dy + dz * dz)
            prev = rec

        # Straight-line distance
        straight_line = math.sqrt(
            (goal[0] - start[0]) ** 2
            + (goal[1] - start[1]) ** 2
            + (goal[2] - start[2]) ** 2
        )

        npl = path_length / straight_line if straight_line > 1e-6 else float("inf")

        # Constraint violations during NAVIGATE
        min_alt = constraints.get("min_altitude", 2.0)
        max_alt = constraints.get("max_altitude", 50.0)
        geofence_r = constraints.get("geofence_radius", 1e6)
        violations = 0
        for rec in trajectory:
            if rec.phase == "NAVIGATE":
                alt = -rec.z  # NED to altitude
                if alt < min_alt - 0.5 or alt > max_alt + 0.5:
                    violations += 1
                horiz = math.sqrt(rec.x ** 2 + rec.y ** 2)
                if horiz > geofence_r:
                    violations += 1

        # Success requires: FSM said goal reached AND final position is
        # actually close AND collision count is reasonable.
        max_collisions_for_pass = 50
        success_dist_threshold = self._arrival_tolerance * 2.0
        actual_success = (
            success
            and final_dist < success_dist_threshold
            and len(collisions) <= max_collisions_for_pass
        )

        if success and not actual_success:
            reasons = []
            if final_dist >= success_dist_threshold:
                reasons.append(f"final_dist={final_dist:.1f}m >= {success_dist_threshold:.1f}m")
            if len(collisions) > max_collisions_for_pass:
                reasons.append(f"collisions={len(collisions)} > {max_collisions_for_pass}")
            logger.info(f"  Task {task_cfg['id']}: FSM reported success but "
                        f"overridden to FAIL ({', '.join(reasons)})")

        return {
            "task_id": task_cfg["id"],
            "instruction": task_cfg["instruction"],
            "goal": goal,
            "success": actual_success,
            "final_distance_to_goal": round(final_dist, 4),
            "closest_approach": round(min_dist, 4),
            "path_length": round(path_length, 4),
            "straight_line_distance": round(straight_line, 4),
            "normalized_path_length": round(npl, 4),
            "time_to_goal_s": round(elapsed, 2),
            "collision_count": len(collisions),
            "constraint_violations": violations,
        }

    def _compute_all_metrics(self) -> dict:
        """Compute aggregate metrics across all tasks."""
        n = len(self._results)
        if n == 0:
            return {"aggregate": {}, "tasks": []}

        successes = sum(1 for r in self._results if r["success"])
        fdgs = [r["final_distance_to_goal"] for r in self._results]
        npls = [r["normalized_path_length"] for r in self._results
                if r["normalized_path_length"] != float("inf")]
        ttgs = [r["time_to_goal_s"] for r in self._results]
        total_collisions = sum(r["collision_count"] for r in self._results)
        total_violations = sum(r["constraint_violations"] for r in self._results)

        aggregate = {
            "num_tasks": n,
            "success_rate": round(successes / n, 4),
            "mean_final_distance": round(float(np.mean(fdgs)), 4),
            "std_final_distance": round(float(np.std(fdgs)), 4),
            "mean_normalized_path_length": round(float(np.mean(npls)), 4) if npls else None,
            "mean_time_to_goal_s": round(float(np.mean(ttgs)), 2),
            "total_collisions": total_collisions,
            "total_constraint_violations": total_violations,
        }

        return {"aggregate": aggregate, "tasks": self._results}

    def _write_metrics(self, metrics: dict) -> None:
        """Write metrics.json to output directory."""
        out_path = self._output_dir / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics written to {out_path}")

    def _print_summary(self, metrics: dict) -> None:
        """Print a human-readable summary table."""
        agg = metrics["aggregate"]
        tasks = metrics["tasks"]

        print(f"\n{'='*72}")
        print(f"  BENCHMARK RESULTS — {agg['num_tasks']} tasks")
        print(f"{'='*72}")
        print(f"{'Task':>6}  {'Status':>7}  {'FDG(m)':>8}  {'NPL':>7}  "
              f"{'Time(s)':>8}  {'Collisions':>10}  {'Violations':>10}")
        print(f"{'-'*72}")

        for t in tasks:
            status = "PASS" if t["success"] else "FAIL"
            print(f"{t['task_id']:>6}  {status:>7}  "
                  f"{t['final_distance_to_goal']:>8.3f}  "
                  f"{t['normalized_path_length']:>7.2f}  "
                  f"{t['time_to_goal_s']:>8.1f}  "
                  f"{t['collision_count']:>10}  "
                  f"{t['constraint_violations']:>10}")

        print(f"{'-'*72}")
        print(f"  SR={agg['success_rate']:.0%}  "
              f"mean_FDG={agg['mean_final_distance']:.3f}m  "
              f"mean_NPL={agg.get('mean_normalized_path_length', 'N/A')}  "
              f"collisions={agg['total_collisions']}  "
              f"violations={agg['total_constraint_violations']}")
        print(f"{'='*72}\n")
