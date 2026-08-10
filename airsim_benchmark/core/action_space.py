"""Shared OpenFly 8-D action encoding and v4 normalization bounds."""

import math
from typing import Tuple

import numpy as np

ACTION_DIM = 8
VLN_Q01 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
VLN_Q99 = np.array([1, 5, 15, 15, 2, 2, 0, 0], dtype=np.float32)


def _wrap_yaw_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def encode_displacement(
    pos_before: Tuple[float, ...],
    pos_after: Tuple[float, ...],
    yaw_before_deg: float,
    yaw_after_deg: float,
    at_goal: bool = False,
) -> np.ndarray:
    if at_goal:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    dx = pos_after[0] - pos_before[0]
    dy = pos_after[1] - pos_before[1]
    dz = pos_after[2] - pos_before[2]  # NED: +z is down

    forward = min(math.hypot(dx, dy), float(VLN_Q99[1]))
    yaw_delta = _wrap_yaw_deg(yaw_after_deg - yaw_before_deg)
    yaw_left = max(0.0, min(yaw_delta, float(VLN_Q99[2])))
    yaw_right = max(0.0, min(-yaw_delta, float(VLN_Q99[3])))
    alt_up = max(0.0, min(-dz, float(VLN_Q99[4])))
    alt_down = max(0.0, min(dz, float(VLN_Q99[5])))
    return np.array(
        [0.0, forward, yaw_left, yaw_right, alt_up, alt_down, 0.0, 0.0],
        dtype=np.float32,
    )


def normalize_action(action_vec: np.ndarray) -> np.ndarray:
    q_range = VLN_Q99 - VLN_Q01
    safe_range = np.where(q_range > 0, q_range, 1.0)
    normalized = np.where(
        q_range > 0,
        2.0 * (action_vec - VLN_Q01) / safe_range - 1.0,
        0.0,
    )
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def denormalize_action(normalized: np.ndarray) -> np.ndarray:
    q_range = VLN_Q99 - VLN_Q01
    return np.where(
        q_range > 0,
        0.5 * (normalized + 1.0) * q_range + VLN_Q01,
        0.0,
    ).astype(np.float32)


def polar_to_ned_delta(
    forward_m: float,
    yaw_left_deg: float,
    yaw_right_deg: float,
    alt_up_m: float,
    alt_down_m: float,
    current_yaw_deg: float,
    stop: float,
    creep_m: float = 2.0,
    stop_threshold: float = 0.5,
) -> Tuple[float, float, float, bool]:
    """Convert polar action to NED displacement. hover=True means stay put."""
    if stop > stop_threshold:
        return 0.0, 0.0, 0.0, True

    net_yaw = yaw_left_deg - yaw_right_deg
    fwd = forward_m
    if fwd < 0.4 and abs(net_yaw) > 2.0:
        fwd = creep_m
    heading_rad = math.radians(current_yaw_deg + net_yaw)
    dx = fwd * math.cos(heading_rad)
    dy = fwd * math.sin(heading_rad)
    dz = -alt_up_m + alt_down_m
    return dx, dy, dz, False


def regression_action_to_ned(
    action_np,
    current_yaw_deg: float,
    creep_m: float = 2.0,
) -> Tuple[np.ndarray, bool]:
    """Convert 8-D denormalized polar action to NED metres.

    action_np is [stop, fwd, yawL, yawR, up, down, ...].
    Returns (np.array([dx, dy, dz]), hover). hover=True means stay put.
    """
    action_np = np.asarray(action_np, dtype=np.float64)
    alt_up = float(action_np[4]) if action_np.shape[0] > 4 else 0.0
    alt_down = float(action_np[5]) if action_np.shape[0] > 5 else 0.0
    dx, dy, dz, hover = polar_to_ned_delta(
        forward_m=float(action_np[1]),
        yaw_left_deg=float(action_np[2]),
        yaw_right_deg=float(action_np[3]),
        alt_up_m=alt_up,
        alt_down_m=alt_down,
        current_yaw_deg=float(current_yaw_deg),
        stop=float(action_np[0]),
        creep_m=creep_m,
    )
    return np.array([dx, dy, dz], dtype=np.float64), bool(hover)
