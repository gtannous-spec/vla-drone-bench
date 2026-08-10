"""Decide how to execute a waypoint hop (no AirSim import).

SimpleFlight's default 3 m lookahead swallows 1.5–2.5 m hops. XY hops need
lookahead 0.5. Pure-Z hops use timed ``moveByVelocity`` (lookahead ``moveToZ``
from hover only crawls). ForwardOnly overshoots heading; callers use
MaxDegreeOfFreedom + clipped yaw.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

XY_EPS = 0.2
Z_EPS = 0.25
YAW_CLIP_DEG = 20.0
STUCK_M = 0.2

Pos3 = Tuple[float, float, float]


def wrap_yaw_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw (heading) in degrees from a (w, x, y, z) quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def plan_move_to(
    current: Pos3,
    target: Pos3,
    yaw_deg: float,
) -> Dict:
    """Classify a waypoint as noop, vertical-only, or XY with clipped yaw."""
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    dz = target[2] - current[2]
    horiz = math.hypot(dx, dy)
    if horiz <= XY_EPS:
        if abs(dz) < Z_EPS:
            return {"kind": "noop"}
        vel = plan_vertical_velocity(current[2], target[2])
        return {"kind": "vertical", "z": float(target[2]), **vel}
    desired = math.degrees(math.atan2(dy, dx))
    turn = wrap_yaw_deg(desired - yaw_deg)
    turn = max(-YAW_CLIP_DEG, min(YAW_CLIP_DEG, turn))
    return {
        "kind": "xy",
        "x": float(target[0]),
        "y": float(target[1]),
        "z": float(target[2]),
        "yaw_deg": yaw_deg + turn,
        "lookahead": 0.5,
        "drivetrain": "MaxDegreeOfFreedom",
    }


def plan_vertical_velocity(
    current_z: float,
    target_z: float,
    speed: float = 2.0,
    max_duration: float = 10.0,
) -> Dict:
    """Timed body-frame-free NED velocity for a pure-Z hop.

    ``moveToZAsync`` with lookahead 0.5 does not leave hover (~0.14 m/hop).
    ``moveByVelocityAsync(0, 0, vz, duration)`` is not lookahead-gated.
    NED: +vz is down.
    """
    dz = float(target_z) - float(current_z)
    speed = max(abs(float(speed)), 0.1)
    vz = math.copysign(speed, dz) if dz != 0.0 else 0.0
    duration = min(abs(dz) / speed, float(max_duration)) if speed else 0.0
    return {"vz": vz, "duration": duration}


def should_call_airsim_land(z_ned: float, max_alt_ned: float = -4.0) -> bool:
    """True when already near the ground (NED +z down), so land() is short."""
    return float(z_ned) >= float(max_alt_ned)


def is_stuck_hop(
    pos: Pos3,
    pos_after: Pos3,
    planned_stop: bool,
    threshold: float = STUCK_M,
) -> bool:
    """True when a non-stop hop barely moved (collision freeze / ignored cmd)."""
    if planned_stop:
        return False
    dx = pos_after[0] - pos[0]
    dy = pos_after[1] - pos[1]
    dz = pos_after[2] - pos[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) < threshold
