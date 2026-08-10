import math
from typing import Tuple

Pos = Tuple[float, float, float]


def wrap_yaw_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def clip_heading_error(error_deg: float, max_abs: float = 15.0) -> float:
    return max(-max_abs, min(max_abs, error_deg))


def bearing_to_xy(
    pos_xy: Tuple[float, float],
    yaw_deg: float,
    target_xy: Tuple[float, float],
) -> float:
    dx = target_xy[0] - pos_xy[0]
    dy = target_xy[1] - pos_xy[1]
    target_yaw = math.degrees(math.atan2(dy, dx))
    return wrap_yaw_deg(target_yaw - yaw_deg)


def landmark_in_fov(
    pos: Pos,
    yaw_deg: float,
    landmark: Pos,
    half_fov_deg: float = 40.0,
    max_range_m: float = 150.0,
    min_forward_m: float = 1.0,
) -> bool:
    dx = landmark[0] - pos[0]
    dy = landmark[1] - pos[1]
    dist = math.hypot(dx, dy)
    if dist > max_range_m or dist < 1e-6:
        return False
    yaw_rad = math.radians(yaw_deg)
    x_body = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    y_body = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    if x_body < min_forward_m:
        return False
    bearing = math.degrees(math.atan2(y_body, x_body))
    return abs(bearing) < half_fov_deg
