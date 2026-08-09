"""
vlm_controller.py — VLM Scene Understanding Controller.

Uses a Vision-Language Model to understand the drone's surroundings
and navigate toward language-described targets. The VLM provides
heading corrections and distance estimates which are converted
into AirSim waypoints.

Architecture:
    Camera + Instruction → VLM → "target is AHEAD-RIGHT, ~40m"
    → Parse heading offset → Apply to current yaw → Waypoint
    → Classical moveToPosition → New Position → Repeat
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState
from ..core.vlm_inference import VLMInference, NavigationHint

logger = logging.getLogger(__name__)


class VLMController(BaseController):
    """VLM-guided navigation controller.

    The VLM analyzes camera images and answers "where is the target?"
    with a heading offset and distance estimate. The controller converts
    these into AirSim waypoints, falling back to classical coordinate
    navigation when the VLM has low confidence.

    Args:
        model_path: Path or HuggingFace ID for the VLM.
        arrival_tolerance: Distance threshold for goal check (m).
        nav_speed: Cruise speed (m/s).
        waypoint_scale: Maximum hop distance (m).
        confidence_threshold: Below this, fall back to classical.
        max_hops: Maximum navigation iterations per task.
        device: CUDA device for VLM inference.
    """

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2-8B",
        arrival_tolerance: float = 1.5,
        nav_speed: float = 5.0,
        waypoint_scale: float = 15.0,
        confidence_threshold: float = 0.3,
        max_hops: int = 40,
        device: str = "auto",
    ):
        self._arrival_tolerance = arrival_tolerance
        self._nav_speed = nav_speed
        self._waypoint_scale = waypoint_scale
        self._confidence_threshold = confidence_threshold
        self._max_hops = max_hops

        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")

        self._vlm = VLMInference(model_path=model_path, device=device)

        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._instruction: str = ""
        self._constraints: dict = {}
        self._task_id: int = 0
        self._hop_count: int = 0
        self._vlm_guided_hops: int = 0

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._vlm_guided_hops = 0

        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)

        logger.info(
            f"VLMController reset — task {self._task_id}, "
            f"instruction='{self._instruction}', goal={self._goal}"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        goal_dist = self._distance_to_goal(state)

        if state.image is None:
            logger.warning("No camera image — classical fallback")
            return self._classical_action(state)

        hint = self._vlm.query(state.image, self._instruction)

        if hint.confidence < self._confidence_threshold:
            logger.debug(
                f"VLM hop {self._hop_count}: low confidence ({hint.confidence:.1f}) "
                f"— classical fallback (goal_dist={goal_dist:.1f}m)"
            )
            return self._classical_action(state)

        self._vlm_guided_hops += 1
        current_yaw = self._yaw_from_quaternion(state.orientation)
        target_yaw = current_yaw + math.radians(hint.heading_offset_deg)

        vlm_dist = hint.distance_estimate_m if hint.distance_estimate_m > 0 else goal_dist
        step = min(self._waypoint_scale, max(3.0, vlm_dist * 0.3))
        step = min(step, max(2.0, goal_dist * 0.5))

        target_x = state.position[0] + step * math.cos(target_yaw)
        target_y = state.position[1] + step * math.sin(target_yaw)
        target_z = self._maintain_altitude(state)

        target_z = self._clamp_altitude(target_z)

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"VLM hop {self._hop_count}: heading={hint.heading_offset_deg:+.0f}°, "
                f"conf={hint.confidence:.1f}, dist_est={hint.distance_estimate_m:.0f}m, "
                f"goal_dist={goal_dist:.1f}m, step={step:.1f}m, "
                f"waypoint=({target_x:.1f}, {target_y:.1f}, {target_z:.1f}), "
                f"reason='{hint.reasoning[:50]}'"
            )

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        if self._hop_count >= self._max_hops:
            logger.info(
                f"Max hops ({self._max_hops}) reached — "
                f"VLM guided {self._vlm_guided_hops}/{self._hop_count} hops"
            )
            return True

        dist = self._distance_to_goal(state)
        if dist < self._arrival_tolerance * 2.0:
            logger.info(
                f"Goal reached at {dist:.1f}m "
                f"(VLM guided {self._vlm_guided_hops}/{self._hop_count} hops)"
            )
            return True

        return False

    def _classical_action(self, state: DroneState) -> ControlAction:
        """Navigate directly toward goal coordinates."""
        gx, gy, gz = self._goal
        gz = self._clamp_altitude(gz)
        goal_dist = self._distance_to_goal(state)
        step = min(self._waypoint_scale, max(3.0, goal_dist * 0.5))

        dx = gx - state.position[0]
        dy = gy - state.position[1]
        horiz_dist = math.sqrt(dx * dx + dy * dy)
        if horiz_dist > 1e-3:
            ratio = step / horiz_dist
            tx = state.position[0] + dx * ratio
            ty = state.position[1] + dy * ratio
        else:
            tx, ty = gx, gy

        return ControlAction(
            target_position=(tx, ty, gz),
            velocity=self._effective_speed,
        )

    def _maintain_altitude(self, state: DroneState) -> float:
        """Keep current altitude unless goal altitude differs significantly."""
        goal_z = self._goal[2]
        current_z = state.position[2]
        diff = abs(goal_z - current_z)
        if diff > 3.0:
            return current_z + (goal_z - current_z) * 0.3
        return current_z

    def _distance_to_goal(self, state: DroneState) -> float:
        dx = state.position[0] - self._goal[0]
        dy = state.position[1] - self._goal[1]
        dz = state.position[2] - self._goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _clamp_altitude(self, z_ned: float) -> float:
        min_alt = self._constraints.get("min_altitude", 2.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        margin = 1.0
        z_min = -(max_alt - margin)
        z_max = -(min_alt + margin)
        return max(z_min, min(z_max, z_ned))

    @staticmethod
    def _yaw_from_quaternion(q: Tuple[float, float, float, float]) -> float:
        """Extract yaw (heading) from quaternion (w, x, y, z)."""
        w, x, y, z = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
