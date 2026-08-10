"""Polar execute contract: physical NED metres, hover/stop, no north fallback.

Does not import or instantiate OpenFlyController (that would load the 7B model).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from airsim_benchmark.core.action_space import regression_action_to_ned

_CONTROLLER = (
    Path(__file__).resolve().parents[1]
    / "airsim_benchmark"
    / "controllers"
    / "openfly_controller.py"
)


def _controller_src() -> str:
    return _CONTROLLER.read_text(encoding="utf-8")


def test_stop_high_is_hover_zero_delta():
    action = np.array([0.9, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta, hover = regression_action_to_ned(action, current_yaw_deg=0.0)
    assert hover is True
    np.testing.assert_allclose(delta, [0.0, 0.0, 0.0])


def test_yaw_left_zero_forward_creeps_along_15_deg():
    action = np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta, hover = regression_action_to_ned(action, current_yaw_deg=0.0, creep_m=2.0)
    assert hover is False
    heading = math.radians(15.0)
    assert delta[0] == pytest.approx(2.0 * math.cos(heading), abs=1e-5)
    assert delta[1] == pytest.approx(2.0 * math.sin(heading), abs=1e-5)
    assert delta[2] == pytest.approx(0.0)


def test_down_and_forward_physical_metres():
    action = np.array([0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float64)
    delta, hover = regression_action_to_ned(action, current_yaw_deg=0.0)
    assert hover is False
    assert delta[2] == pytest.approx(2.0)
    assert delta[0] == pytest.approx(4.0)


def test_current_yaw_90_forward_is_east():
    action = np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta, hover = regression_action_to_ned(action, current_yaw_deg=90.0)
    assert hover is False
    assert delta[0] == pytest.approx(0.0, abs=1e-5)
    assert delta[1] == pytest.approx(4.0, abs=1e-5)


def test_controller_uses_shared_denorm_not_q99_dim5_zero():
    src = _controller_src()
    assert "denormalize_action" in src
    assert "regression_action_to_ned" in src
    assert "[1, 5, 15, 15, 2, 0, 0, 0]" not in src


def test_controller_gps_free_physical_metres_not_unit_times_scale():
    src = _controller_src()
    assert "offset = model_unit * self._waypoint_scale" not in src
    assert "_last_hover" in src
    assert "_hover_action" in src
    assert "_stop_streak" in src
    assert "_stop_streak >= 2" in src


def test_controller_no_image_or_exception_hovers_not_north():
    src = _controller_src()
    assert "return self._hover_action(state)" in src
    assert "return self._fallback_action(state)" not in src
