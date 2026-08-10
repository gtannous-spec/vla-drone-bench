"""
oracle_controller.py — Intent-driven oracle for LoRA training collection.

Privileged geometry (landmark pose + FOV cone) selects hops from the
instruction's intent: search until the landmark is in FOV, approach,
land on a surface, land on the ground, climb/descend, or ego turns.

Each hop is labelled with a planned 8-D OpenFly action vector. The
collector overwrites labels with measured encode_displacement; both
paths must be able to encode descent (dim5).
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from airsim_benchmark.collection.geometry import (
    bearing_to_xy,
    clip_heading_error,
    landmark_in_fov,
    wrap_yaw_deg,
)
from airsim_benchmark.core.action_space import VLN_Q01, VLN_Q99

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)

# Keep historical names; Q99 must alias core.action_space (dim5=2).
VLN_NORM_Q01 = VLN_Q01
VLN_NORM_Q99 = VLN_Q99

_STOP_VEC = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
_SEARCH_FWD = 2.5
_APPROACH_FWD = 5.0
_EGO_FWD = 2.5
_VERT_STEP = 2.0
_TURN_CLIP = 15.0
_GROUND_Z = -1.5
_GROUND_STOP_Z = -1.9
_SURFACE_TOL = 1.5
_OVERHEAD_M = 3.0
_EGO_FINISH_DEG = 80.0
_AROUND_FINISH_DEG = 160.0
_CRUISE_FWD = 1.5

Pos3 = Tuple[float, float, float]


def quat_to_yaw_deg(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw (heading) in degrees from a (w, x, y, z) quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _as_xyz(value) -> Optional[Pos3]:
    if value is None or value == "null":
        return None
    seq = list(value)
    if len(seq) < 3:
        return None
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _offset_xy(x: float, y: float, heading_deg: float, fwd: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return x + fwd * math.cos(rad), y + fwd * math.sin(rad)


class OracleController(BaseController):
    """Intent machine that flies hops caused by the instruction, not GPS goals."""

    def __init__(
        self,
        nav_speed: float = 5.0,
        hop_distance: float = 15.0,
        n_segments: int = 6,
        lateral_noise: float = 20.0,
        alt_noise: float = 3.0,
        perturb_prob: float = 0.25,
        perturb_yaw_max: float = 45.0,
        perturb_offset_max: float = 8.0,
    ) -> None:
        self._nav_speed = nav_speed
        self._hop_distance = hop_distance
        self._n_segments = n_segments
        self._lateral_noise = lateral_noise
        self._alt_noise = alt_noise
        self._perturb_prob = perturb_prob
        self._perturb_yaw_max = perturb_yaw_max
        self._perturb_offset_max = perturb_offset_max

        self._instruction: str = ""
        self._intent: str = ""
        self._start: Pos3 = (0.0, 0.0, -10.0)
        self._landmark: Optional[Pos3] = None
        self._landmark_radius: float = 8.0
        self._target_alt_ned: float = -10.0
        self._surface_z: Optional[float] = None
        self._min_alt: float = 5.0
        self._max_alt: float = 20.0
        self._start_yaw: Optional[float] = None
        self._hop_count: int = 0
        self._max_hops: int = 50
        self._stop_streak: int = 0

        self.last_action_id: int = -1
        self.last_action_vec: np.ndarray = np.zeros(8, dtype=np.float32)
        self.last_in_fov: bool = False

    def reset(self, task_config: dict) -> None:
        self._instruction = task_config.get("instruction", "")
        self._intent = str(task_config.get("intent") or "")
        self._start = _as_xyz(task_config.get("start", [0, 0, -10])) or (0.0, 0.0, -10.0)
        self._max_hops = int(task_config.get("max_hops", 50))
        self._hop_count = 0
        self._stop_streak = 0

        self._landmark = _as_xyz(task_config.get("landmark_position"))
        self._landmark_radius = float(task_config.get("landmark_radius", 8.0))
        if "target_alt_ned" in task_config and task_config["target_alt_ned"] is not None:
            self._target_alt_ned = float(task_config["target_alt_ned"])
        else:
            self._target_alt_ned = self._start[2]
        if "surface_z" in task_config and task_config["surface_z"] is not None:
            self._surface_z = float(task_config["surface_z"])
        else:
            self._surface_z = None

        constraints = task_config.get("constraints") or {}
        self._min_alt = float(constraints.get("min_altitude", 5.0))
        self._max_alt = float(constraints.get("max_altitude", 20.0))

        if "start_yaw" in task_config and task_config["start_yaw"] is not None:
            self._start_yaw = float(task_config["start_yaw"])
        else:
            self._start_yaw = None

        self.last_action_id = -1
        self.last_action_vec = np.zeros(8, dtype=np.float32)
        self.last_in_fov = False

        logger.info(
            "OracleController reset — intent=%s landmark=%s instruction='%s'",
            self._intent or "(none)",
            self._landmark,
            self._instruction,
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        x, y, z = (float(state.position[0]), float(state.position[1]), float(state.position[2]))
        yaw = quat_to_yaw_deg(*state.orientation)
        if self._start_yaw is None:
            self._start_yaw = yaw

        if self._landmark is not None:
            self.last_in_fov = bool(landmark_in_fov((x, y, z), yaw, self._landmark))
            horiz = math.hypot(self._landmark[0] - x, self._landmark[1] - y)
        else:
            self.last_in_fov = False
            horiz = float("inf")

        intent = self._intent
        if intent == "search_then_approach":
            return self._act_search_then_approach(x, y, z, yaw, horiz)
        if intent == "land_on_surface":
            return self._act_land_on_surface(x, y, z, yaw, horiz)
        if intent == "land_ground":
            return self._act_land_ground(x, y, z, yaw)
        if intent == "climb":
            return self._act_climb(x, y, z, yaw)
        if intent == "descend":
            return self._act_descend(x, y, z, yaw)
        if intent == "ego_turn_left":
            return self._act_ego_turn(x, y, z, yaw, turn_deg=_TURN_CLIP, finish_deg=_EGO_FINISH_DEG)
        if intent == "ego_turn_right":
            return self._act_ego_turn(x, y, z, yaw, turn_deg=-_TURN_CLIP, finish_deg=_EGO_FINISH_DEG)
        if intent == "ego_turn_around":
            return self._act_ego_turn(x, y, z, yaw, turn_deg=_TURN_CLIP, finish_deg=_AROUND_FINISH_DEG)

        if self._landmark is not None:
            return self._act_search_then_approach(x, y, z, yaw, horiz)
        return self._emit_stop((x, y, z))

    def is_goal_reached(self, state: DroneState) -> bool:
        del state
        return self._hop_count >= self._max_hops or self._stop_streak >= 2

    def _search_or_approach_hop(
        self, x: float, y: float, z: float, yaw: float, horiz: float, in_fov: bool,
    ) -> ControlAction:
        assert self._landmark is not None
        lx, ly = self._landmark[0], self._landmark[1]
        error = bearing_to_xy((x, y), yaw, (lx, ly))
        turn = clip_heading_error(error, _TURN_CLIP)
        heading = yaw + turn
        fwd = _SEARCH_FWD if not in_fov else min(_APPROACH_FWD, horiz)
        tx, ty = _offset_xy(x, y, heading, fwd)
        return self._emit_move((x, y, z), yaw, (tx, ty, z), planned_turn=turn)

    def _act_search_then_approach(
        self, x: float, y: float, z: float, yaw: float, horiz: float,
    ) -> ControlAction:
        if self._landmark is None:
            return self._emit_stop((x, y, z))
        if horiz < self._landmark_radius:
            return self._emit_stop((x, y, z))
        if not self.last_in_fov:
            return self._search_or_approach_hop(x, y, z, yaw, horiz, in_fov=False)
        return self._search_or_approach_hop(x, y, z, yaw, horiz, in_fov=True)

    def _act_land_on_surface(
        self, x: float, y: float, z: float, yaw: float, horiz: float,
    ) -> ControlAction:
        if self._landmark is None:
            return self._emit_stop((x, y, z))
        overhead = min(_OVERHEAD_M, 0.5 * self._landmark_radius)
        if horiz >= overhead:
            return self._search_or_approach_hop(
                x, y, z, yaw, horiz, in_fov=self.last_in_fov,
            )
        surface_z = self._surface_z if self._surface_z is not None else self._landmark[2]
        if abs(z - surface_z) > _SURFACE_TOL:
            dz = max(-_VERT_STEP, min(_VERT_STEP, surface_z - z))
            return self._emit_move((x, y, z), yaw, (x, y, z + dz), planned_turn=0.0)
        return self._emit_stop((x, y, z))

    def _act_land_ground(self, x: float, y: float, z: float, yaw: float) -> ControlAction:
        if z < _GROUND_STOP_Z:
            target_z = min(z + _VERT_STEP, _GROUND_Z)
            return self._emit_move((x, y, z), yaw, (x, y, target_z), planned_turn=0.0)
        return self._emit_stop((x, y, z))

    def _act_climb(self, x: float, y: float, z: float, yaw: float) -> ControlAction:
        if z > self._target_alt_ned:
            dz_up = min(_VERT_STEP, z - self._target_alt_ned)
            tx, ty = _offset_xy(x, y, yaw, _CRUISE_FWD)
            return self._emit_move((x, y, z), yaw, (tx, ty, z - dz_up), planned_turn=0.0)
        return self._emit_stop((x, y, z))

    def _act_descend(self, x: float, y: float, z: float, yaw: float) -> ControlAction:
        if z < self._target_alt_ned:
            dz_down = min(_VERT_STEP, self._target_alt_ned - z)
            tx, ty = _offset_xy(x, y, yaw, _CRUISE_FWD)
            return self._emit_move((x, y, z), yaw, (tx, ty, z + dz_down), planned_turn=0.0)
        return self._emit_stop((x, y, z))

    def _act_ego_turn(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        turn_deg: float,
        finish_deg: float,
    ) -> ControlAction:
        start_yaw = self._start_yaw if self._start_yaw is not None else yaw
        if abs(wrap_yaw_deg(yaw - start_yaw)) >= finish_deg:
            return self._emit_stop((x, y, z))
        heading = yaw + turn_deg
        tx, ty = _offset_xy(x, y, heading, _EGO_FWD)
        return self._emit_move((x, y, z), yaw, (tx, ty, z), planned_turn=turn_deg)

    def _emit_stop(self, pos: Pos3) -> ControlAction:
        self.last_action_vec = _STOP_VEC.copy()
        self.last_action_id = -1
        self._stop_streak += 1
        return ControlAction(target_position=pos, velocity=self._nav_speed)

    def _emit_move(
        self,
        pos: Pos3,
        yaw: float,
        target: Pos3,
        planned_turn: Optional[float] = None,
    ) -> ControlAction:
        self._stop_streak = 0
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dz = target[2] - pos[2]
        forward = min(math.hypot(dx, dy), float(VLN_NORM_Q99[1]))
        if planned_turn is None:
            if math.hypot(dx, dy) > 1e-6:
                planned_turn = wrap_yaw_deg(math.degrees(math.atan2(dy, dx)) - yaw)
            else:
                planned_turn = 0.0
        turn = clip_heading_error(float(planned_turn), _TURN_CLIP)
        yaw_left = max(0.0, min(turn, float(VLN_NORM_Q99[2])))
        yaw_right = max(0.0, min(-turn, float(VLN_NORM_Q99[3])))
        alt_up = max(0.0, min(-dz, float(VLN_NORM_Q99[4])))
        alt_down = max(0.0, min(dz, float(VLN_NORM_Q99[5])))
        self.last_action_vec = np.array(
            [0.0, forward, yaw_left, yaw_right, alt_up, alt_down, 0.0, 0.0],
            dtype=np.float32,
        )
        self.last_action_id = -1
        return ControlAction(target_position=target, velocity=self._nav_speed)
