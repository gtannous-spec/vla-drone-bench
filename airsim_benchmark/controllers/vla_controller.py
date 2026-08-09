"""
vla_controller.py — VLA Hybrid Goal Selector Controller (Milestone 2).

Uses OpenVLA-7B as a vision-language directional oracle. The VLA observes
camera images + language instructions and outputs action deltas, which are
scaled into waypoints for the classical planner to execute.

Architecture:
    Camera + Instruction → OpenVLA → Direction Delta → Goal-Bias Blend → Waypoint
    Waypoint → Classical moveToPosition → New Position → Repeat

Goal detection uses multi-signal convergence:
    1. Action magnitude decay (VLA thinks it arrived) — guarded by min_hops
    2. Direction reversal (oscillation around target)
    3. Coordinate proximity fallback (hybrid safety net)
"""

import logging
import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState
from ..core.vla_inference import OpenVLAInference

logger = logging.getLogger(__name__)


class VLAHybridController(BaseController):
    """Hybrid VLA controller: VLA provides direction, classical planner executes.

    Args:
        model_path: Path to OpenVLA-7B weights.
        arrival_tolerance: Distance threshold for coordinate-based goal check (m).
        nav_speed: Cruise speed for waypoint navigation (m/s).
        waypoint_scale: Multiplier for VLA deltas → waypoint distance (m).
        convergence_threshold: Action norm below which VLA considers goal reached.
        max_hops: Maximum navigation iterations before declaring failure.
        min_hops_before_convergence: Minimum hops before convergence can trigger.
        goal_bias: Fraction of goal direction blended into VLA output (0=pure VLA, 1=pure classical).
        fallback_to_coords: Whether to use coordinate fallback for goal detection.
        device: CUDA device for VLA inference.
    """

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        arrival_tolerance: float = 1.5,
        nav_speed: float = 5.0,
        waypoint_scale: float = 15.0,
        convergence_threshold: float = 0.005,
        max_hops: int = 50,
        min_hops_before_convergence: int = 8,
        goal_bias: float = 0.3,
        fallback_to_coords: bool = True,
        device: str = "auto",
    ):
        self._arrival_tolerance = arrival_tolerance
        self._nav_speed = nav_speed
        self._waypoint_scale = waypoint_scale
        self._convergence_threshold = convergence_threshold
        self._max_hops = max_hops
        self._min_hops = min_hops_before_convergence
        self._goal_bias = goal_bias
        self._fallback_to_coords = fallback_to_coords

        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")

        self._vla = OpenVLAInference(model_path=model_path, device=device)

        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._instruction: str = ""
        self._constraints: dict = {}
        self._task_id: int = 0

        # Navigation state
        self._hop_count: int = 0
        self._recent_deltas: deque = deque(maxlen=5)
        self._last_action: Optional[np.ndarray] = None

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._recent_deltas.clear()
        self._last_action = None

        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)

        logger.info(
            f"VLAHybridController reset — task {self._task_id}, "
            f"instruction='{self._instruction}', "
            f"goal={self._goal}, scale={self._waypoint_scale}, "
            f"goal_bias={self._goal_bias}"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1

        # If no image available, fall back to coordinate goal
        if state.image is None:
            logger.warning("No camera image — falling back to coordinate goal")
            return self._fallback_action(state)

        # Run VLA inference
        direction = self._vla.predict_direction(state.image, self._instruction)
        self._last_action = direction

        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            logger.debug("VLA produced near-zero delta — falling back to coordinate goal")
            return self._fallback_action(state)

        # Record real deltas for convergence checks
        self._recent_deltas.append(direction.copy())

        # Normalize VLA direction
        vla_unit = direction / norm

        # Compute goal direction from current position
        goal_vec = np.array([
            self._goal[0] - state.position[0],
            self._goal[1] - state.position[1],
            self._goal[2] - state.position[2],
        ])
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 1e-3:
            goal_unit = goal_vec / goal_dist
        else:
            goal_unit = np.zeros(3)

        # Blend VLA direction with goal bias
        blended = (1.0 - self._goal_bias) * vla_unit + self._goal_bias * goal_unit
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-6:
            blended = blended / blended_norm

        # Scale hop distance: full scale when far, shorter near goal to avoid overshoot
        effective_scale = min(self._waypoint_scale, max(2.0, goal_dist * 0.5))
        waypoint_offset = blended * effective_scale

        target_x = state.position[0] + waypoint_offset[0]
        target_y = state.position[1] + waypoint_offset[1]
        target_z = state.position[2] + waypoint_offset[2]

        # Clamp altitude within constraints (with safety margin)
        target_z = self._clamp_altitude(target_z)

        target = (target_x, target_y, target_z)

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"VLA hop {self._hop_count}: delta=({direction[0]:.3f}, "
                f"{direction[1]:.3f}, {direction[2]:.3f}), "
                f"norm={norm:.4f}, goal_dist={goal_dist:.1f}m, "
                f"scale={effective_scale:.1f}, "
                f"waypoint=({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
            )

        return ControlAction(
            target_position=target,
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        # Signal 1: Max hops exceeded
        if self._hop_count >= self._max_hops:
            logger.info(f"Max hops ({self._max_hops}) reached — checking proximity")
            return self._check_coordinate_proximity(state)

        # Signal 2: Coordinate proximity (always check — this is the real success)
        if self._fallback_to_coords and self._check_coordinate_proximity(state):
            logger.info("Coordinate proximity reached — goal achieved!")
            return True

        # Signals 3 & 4 only meaningful when near the goal AND past min hops
        dist_to_goal = self._distance_to_goal(state)
        near_goal = dist_to_goal < self._waypoint_scale * 2.0  # within ~2 hop distances

        if self._hop_count >= self._min_hops and near_goal:
            if self._check_convergence():
                logger.info(
                    f"VLA convergence detected after {self._hop_count} hops "
                    f"at {dist_to_goal:.1f}m from goal"
                )
                return True

            if self._check_oscillation():
                logger.info(
                    f"VLA oscillation detected at {dist_to_goal:.1f}m from goal"
                )
                return True

        return False

    def _fallback_action(self, state: DroneState) -> ControlAction:
        """When VLA can't produce useful output, navigate toward coordinates."""
        gx, gy, gz = self._goal
        gz = self._clamp_altitude(gz)
        return ControlAction(
            target_position=(gx, gy, gz),
            velocity=self._effective_speed,
        )

    def _check_convergence(self) -> bool:
        """Check if recent action deltas are all below threshold."""
        if len(self._recent_deltas) < 3:
            return False
        norms = [np.linalg.norm(d) for d in self._recent_deltas]
        return all(n < self._convergence_threshold for n in norms[-3:])

    def _check_oscillation(self) -> bool:
        """Check if actions are reversing direction (oscillating around target)."""
        if len(self._recent_deltas) < 5:
            return False
        reversals = 0
        deltas = list(self._recent_deltas)
        for i in range(1, len(deltas)):
            dot = np.dot(deltas[i], deltas[i - 1])
            if dot < 0:
                reversals += 1
        return reversals >= 4

    def _distance_to_goal(self, state: DroneState) -> float:
        dx = state.position[0] - self._goal[0]
        dy = state.position[1] - self._goal[1]
        dz = state.position[2] - self._goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _check_coordinate_proximity(self, state: DroneState) -> bool:
        """Check Euclidean distance to goal coordinates."""
        return self._distance_to_goal(state) < self._arrival_tolerance * 3.0

    def _clamp_altitude(self, z_ned: float) -> float:
        """Clamp altitude within constraints with safety margin (NED: more negative = higher)."""
        min_alt = self._constraints.get("min_altitude", 2.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        margin = 1.0
        z_min = -(max_alt - margin)
        z_max = -(min_alt + margin)
        return max(z_min, min(z_max, z_ned))
