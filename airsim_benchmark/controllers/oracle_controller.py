"""
oracle_controller.py — Waypoint Oracle for LoRA training data collection.

Generates diverse trajectories by breaking straight-to-goal paths into
segments with intermediate waypoints (zigzags, altitude changes, turns).

At each step the planned displacement is encoded as a *continuous* 8-D
action vector whose dimensions match the OpenFly action space:

    [stop, forward_dist, yaw_left, yaw_right, alt_up, 0, 0, 0]

Values are kept within the model's VLN-norm Q99 bounds so they tokenize
across the full 256-bin range, producing diverse training data.
"""

import logging
import math
import random
from typing import List, Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)

# OpenFly's 10 discrete actions → 8-D continuous vectors.
# Dims: [stop, forward_dist, yaw_left, yaw_right, up, down, left, right]
OPENFLY_ACTION_MAP = {
    0: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # stop
    1: np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # forward x1
    2: np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),  # turn left 30 deg
    3: np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=np.float32),  # turn right 30 deg
    4: np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=np.float32),   # go up
    5: np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32),   # go down
    6: np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=np.float32),   # move left
    7: np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=np.float32),   # move right
    8: np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # forward x2
    9: np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # forward x3
}

# Normalization stats (vln_norm) baked into the OpenFly checkpoint.
VLN_NORM_Q01 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
VLN_NORM_Q99 = np.array([1, 5, 15, 15, 2, 0, 0, 0], dtype=np.float32)


def quat_to_yaw_deg(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw (heading) in degrees from a (w, x, y, z) quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def classify_action(
    heading_delta_deg: float,
    altitude_delta: float,
    horiz_dist: float,
    at_goal: bool,
) -> int:
    """Map a movement to one of the 10 discrete OpenFly actions."""
    if at_goal:
        return 0  # stop

    if altitude_delta > 1.5:
        return 4  # go up
    if altitude_delta < -1.5:
        return 5  # go down

    if heading_delta_deg > 20:
        return 2  # turn left
    if heading_delta_deg < -20:
        return 3  # turn right

    if horiz_dist > 25:
        return 9  # forward x3
    if horiz_dist > 12:
        return 8  # forward x2
    return 1      # forward x1


def generate_waypoints(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    n_segments: int = 6,
    lateral_noise: float = 20.0,
    alt_noise: float = 3.0,
    min_alt: float = -20.0,
    max_alt: float = -5.0,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """Break a straight path into segments with randomised intermediate waypoints."""
    rng = random.Random(seed)
    waypoints = [start]
    for i in range(1, n_segments):
        t = i / n_segments
        # Linear interpolation with lateral and altitude noise
        base_x = start[0] + t * (goal[0] - start[0])
        base_y = start[1] + t * (goal[1] - start[1])
        base_z = start[2] + t * (goal[2] - start[2])
        nx = base_x + rng.uniform(-lateral_noise, lateral_noise)
        ny = base_y + rng.uniform(-lateral_noise, lateral_noise)
        nz = max(min_alt, min(max_alt, base_z + rng.uniform(-alt_noise, alt_noise)))
        waypoints.append((nx, ny, nz))
    waypoints.append(goal)
    return waypoints


class OracleController(BaseController):
    """Waypoint oracle that follows intermediate waypoints and produces
    *continuous* action labels for LoRA training.

    Unlike the classical controller (which always flies directly to the
    goal), this controller navigates through randomised intermediate
    waypoints.  Each step's planned displacement is encoded as a
    continuous 8-D vector (matching OpenFly's action-token dimensions)
    so the training data has high diversity instead of only a handful
    of one-hot patterns.
    """

    def __init__(
        self,
        nav_speed: float = 5.0,
        hop_distance: float = 15.0,
        n_segments: int = 6,
        lateral_noise: float = 20.0,
        alt_noise: float = 3.0,
    ) -> None:
        self._nav_speed = nav_speed
        self._hop_distance = hop_distance
        self._n_segments = n_segments
        self._lateral_noise = lateral_noise
        self._alt_noise = alt_noise

        self._waypoints: List[Tuple[float, float, float]] = []
        self._wp_idx: int = 0
        self._instruction: str = ""
        self._goal: Optional[Tuple[float, float, float]] = None
        self._hop_count: int = 0
        self._max_hops: int = 50
        self._prev_yaw: float = 0.0

        self.last_action_id: int = -1
        self.last_action_vec: np.ndarray = np.zeros(8, dtype=np.float32)

    def reset(self, task_config: dict) -> None:
        self._instruction = task_config.get("instruction", "")
        self._hop_count = 0
        self._max_hops = task_config.get("max_hops", 50)
        self._prev_yaw = 0.0
        self.last_action_id = -1
        self.last_action_vec = np.zeros(8, dtype=np.float32)

        start = tuple(task_config.get("start", [0, 0, -10]))
        self._goal = tuple(task_config["goal"]) if "goal" in task_config else None

        constraints = task_config.get("constraints", {})
        min_alt_val = constraints.get("min_altitude", 5.0)
        max_alt_val = constraints.get("max_altitude", 20.0)

        if self._goal is not None:
            seed = hash((task_config.get("id", 0), task_config.get("run_id", 0)))
            self._waypoints = generate_waypoints(
                start, self._goal,
                n_segments=self._n_segments,
                lateral_noise=self._lateral_noise,
                alt_noise=self._alt_noise,
                min_alt=-max_alt_val,
                max_alt=-min_alt_val,
                seed=seed,
            )
        else:
            self._waypoints = [start]
        self._wp_idx = 1 if len(self._waypoints) > 1 else 0

        logger.info(
            f"OracleController reset — {len(self._waypoints)} waypoints, "
            f"instruction='{self._instruction}'"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1

        # Current yaw from orientation quaternion
        yaw_deg = quat_to_yaw_deg(*state.orientation)

        # Advance waypoint index when close enough
        if self._wp_idx < len(self._waypoints):
            wp = self._waypoints[self._wp_idx]
            dx = wp[0] - state.position[0]
            dy = wp[1] - state.position[1]
            dz = wp[2] - state.position[2]
            dist_to_wp = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_to_wp < self._hop_distance * 0.8 and self._wp_idx < len(self._waypoints) - 1:
                self._wp_idx += 1
                wp = self._waypoints[self._wp_idx]
                dx = wp[0] - state.position[0]
                dy = wp[1] - state.position[1]
                dz = wp[2] - state.position[2]
        else:
            wp = self._waypoints[-1]
            dx = wp[0] - state.position[0]
            dy = wp[1] - state.position[1]
            dz = wp[2] - state.position[2]

        horiz_dist = math.sqrt(dx * dx + dy * dy)

        # Heading to target
        target_yaw = math.degrees(math.atan2(dy, dx))
        heading_delta = target_yaw - yaw_deg
        # Normalise to [-180, 180]
        heading_delta = (heading_delta + 180) % 360 - 180

        at_goal = (self._goal is not None
                   and math.sqrt(sum((a - b) ** 2 for a, b in zip(state.position, self._goal))) < 5.0)

        action_id = classify_action(heading_delta, dz, horiz_dist, at_goal)
        self.last_action_id = action_id

        self._prev_yaw = yaw_deg

        # Fly toward current waypoint
        step_scale = min(self._hop_distance, horiz_dist + abs(dz))
        if horiz_dist > 1e-3:
            unit_x = dx / horiz_dist
            unit_y = dy / horiz_dist
        else:
            unit_x, unit_y = 1.0, 0.0
        planned_horiz = min(step_scale, horiz_dist)
        target_x = state.position[0] + unit_x * planned_horiz
        target_y = state.position[1] + unit_y * planned_horiz
        target_z = wp[2]

        # Continuous action vector from the planned displacement.
        # Dims: [stop, forward_dist, yaw_left, yaw_right, alt_up, 0, 0, 0]
        # Values are clamped to the VLN-norm Q99 bounds so they tokenize
        # across the full 256-bin range.
        if at_goal:
            self.last_action_vec = np.array(
                [1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32,
            )
        else:
            forward = min(planned_horiz, VLN_NORM_Q99[1])  # ≤ 5
            yaw_left = max(0.0, min(heading_delta, VLN_NORM_Q99[2]))   # ≤ 15
            yaw_right = max(0.0, min(-heading_delta, VLN_NORM_Q99[3])) # ≤ 15
            planned_alt = target_z - state.position[2]
            alt_up = max(0.0, min(planned_alt, VLN_NORM_Q99[4]))       # ≤ 2
            self.last_action_vec = np.array(
                [0.0, forward, yaw_left, yaw_right, alt_up, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._nav_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        if self._hop_count >= self._max_hops:
            return True
        if self._goal is None:
            return False
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(state.position, self._goal)))
        return dist < 5.0
