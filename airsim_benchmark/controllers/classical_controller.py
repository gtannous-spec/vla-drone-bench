"""
classical_controller.py — Classical Waypoint Controller (Milestone 1 baseline).

Navigates directly to goal coordinates using AirSim's moveToPositionAsync.
No VLA inference, no learning — pure coordinate-based flight.
"""

import logging
import math
from typing import Tuple

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)


class ClassicalWaypointController(BaseController):
    """Direct waypoint navigation using goal coordinates from config."""

    def __init__(self, arrival_tolerance: float = 1.5, nav_speed: float = 5.0):
        self._arrival_tolerance = arrival_tolerance
        self._nav_speed = nav_speed
        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._constraints: dict = {}
        self._task_id: int = 0

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._constraints = task_config.get("constraints", {})
        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)
        logger.info(f"ClassicalController reset — task {self._task_id}, "
                    f"goal={self._goal}, speed={self._effective_speed:.1f} m/s")

    def get_action(self, state: DroneState) -> ControlAction:
        gx, gy, gz = self._goal
        gz = self._clamp_altitude(gz)
        return ControlAction(
            target_position=(gx, gy, gz),
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        dist = self._distance_to_goal(state.position)
        return dist < self._arrival_tolerance

    def _distance_to_goal(self, position: Tuple[float, float, float]) -> float:
        dx = position[0] - self._goal[0]
        dy = position[1] - self._goal[1]
        dz = position[2] - self._goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _clamp_altitude(self, z_ned: float) -> float:
        """Clamp altitude within constraints (NED: more negative = higher)."""
        min_alt = self._constraints.get("min_altitude", 2.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        # NED: z = -altitude, so max_altitude -> most negative z
        z_min = -max_alt
        z_max = -min_alt
        return max(z_min, min(z_max, z_ned))
