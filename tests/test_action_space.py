import math

import numpy as np
import pytest

from airsim_benchmark.core.action_space import (
    VLN_Q99,
    encode_displacement,
    polar_to_ned_delta,
    normalize_action,
    denormalize_action,
)


def test_q99_allows_descent():
    assert VLN_Q99[5] == 2.0
    assert VLN_Q99[6] == 0.0


def test_encode_descent_writes_dim5():
    # NED: +z is down. z from -10 to -8 is 2 m descent.
    vec = encode_displacement(
        pos_before=(0.0, 0.0, -10.0),
        pos_after=(3.0, 0.0, -8.0),
        yaw_before_deg=0.0,
        yaw_after_deg=0.0,
        at_goal=False,
    )
    assert vec[4] == pytest.approx(0.0)
    assert vec[5] == pytest.approx(2.0)
    assert vec[1] == pytest.approx(3.0)


def test_encode_stop():
    vec = encode_displacement(
        (0, 0, -10), (0, 0, -10), 0.0, 0.0, at_goal=True
    )
    assert vec[0] == 1.0
    assert all(v == 0.0 for v in vec[1:])


def test_normalize_roundtrip_dim5():
    raw = np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32)
    n = normalize_action(raw)
    assert n[5] == pytest.approx(1.0)
    back = denormalize_action(n)
    assert back[5] == pytest.approx(2.0)


def test_polar_stop_is_hover_not_north():
    dx, dy, dz, hover = polar_to_ned_delta(
        forward_m=0.0, yaw_left_deg=0.0, yaw_right_deg=0.0,
        alt_up_m=0.0, alt_down_m=0.0, current_yaw_deg=0.0, stop=0.9,
    )
    assert hover is True
    assert (dx, dy, dz) == (0.0, 0.0, 0.0)


def test_polar_yaw_creep_when_forward_near_zero():
    dx, dy, dz, hover = polar_to_ned_delta(
        forward_m=0.0, yaw_left_deg=15.0, yaw_right_deg=0.0,
        alt_up_m=0.0, alt_down_m=0.0, current_yaw_deg=0.0, stop=0.0,
        creep_m=2.0,
    )
    assert hover is False
    heading = math.radians(15.0)
    assert dx == pytest.approx(2.0 * math.cos(heading), abs=1e-5)
    assert dy == pytest.approx(2.0 * math.sin(heading), abs=1e-5)


def test_polar_descent_uses_dim5():
    dx, dy, dz, hover = polar_to_ned_delta(
        forward_m=4.0, yaw_left_deg=0.0, yaw_right_deg=0.0,
        alt_up_m=0.0, alt_down_m=2.0, current_yaw_deg=0.0, stop=0.0,
    )
    assert dz == pytest.approx(2.0)
    assert dx == pytest.approx(4.0)


def test_encode_yaw_wrap_is_short_left_turn():
    # 170° → -170° is a +20° left wrap, not a -340° right turn.
    vec = encode_displacement(
        pos_before=(0.0, 0.0, -10.0),
        pos_after=(0.0, 0.0, -10.0),
        yaw_before_deg=170.0,
        yaw_after_deg=-170.0,
        at_goal=False,
    )
    assert vec[2] == pytest.approx(15.0)
    assert vec[3] == pytest.approx(0.0)


def test_normalize_zero_range_dims_stay_zero():
    raw = np.array([0, 0, 0, 0, 0, 0, 99.0, -3.5], dtype=np.float32)
    n = normalize_action(raw)
    assert n[6] == pytest.approx(0.0)
    assert n[7] == pytest.approx(0.0)
    assert not np.isnan(n).any()
    back = denormalize_action(n)
    assert back[6] == pytest.approx(0.0)
    assert back[7] == pytest.approx(0.0)
    garbage_norm = np.array([0, 0, 0, 0, 0, 0, 0.5, -1.0], dtype=np.float32)
    d = denormalize_action(garbage_norm)
    assert d[6] == pytest.approx(0.0)
    assert d[7] == pytest.approx(0.0)
    assert not np.isnan(d).any()


def test_polar_forward_faces_east_at_yaw_90():
    dx, dy, dz, hover = polar_to_ned_delta(
        forward_m=4.0, yaw_left_deg=0.0, yaw_right_deg=0.0,
        alt_up_m=0.0, alt_down_m=0.0, current_yaw_deg=90.0, stop=0.0,
    )
    assert hover is False
    assert dx == pytest.approx(0.0, abs=1e-5)
    assert dy == pytest.approx(4.0, abs=1e-5)
