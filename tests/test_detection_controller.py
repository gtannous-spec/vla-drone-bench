"""Tests for DetectionController — TRACK/COAST/SEARCH state machine."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from airsim_benchmark.controllers.detection_controller import (
    DetectionController,
    COAST_K,
    CAMERA_FOV_DEG,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)
from airsim_benchmark.controllers.base_controller import DroneState, ControlAction
from airsim_benchmark.core.detection_inference import Detection


def _quaternion_from_yaw_deg(yaw_deg: float):
    """Create (w, x, y, z) quaternion from yaw in degrees (NED frame)."""
    yaw_rad = math.radians(yaw_deg) / 2.0
    return (math.cos(yaw_rad), 0.0, 0.0, math.sin(yaw_rad))


def _make_state(x=0.0, y=0.0, z=-10.0, yaw_deg=0.0, with_image=True):
    """Create a DroneState for testing."""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) if with_image else None
    return DroneState(
        position=(x, y, z),
        velocity=(0.0, 0.0, 0.0),
        orientation=_quaternion_from_yaw_deg(yaw_deg),
        image=image,
    )


def _detection_at(cx=320.0, cy=240.0, w=100.0, h=80.0, score=0.8):
    """Create a Detection with center at (cx, cy) and given dimensions."""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return Detection(bbox_xyxy=(x1, y1, x2, y2), score=score, phrase="target")


def _make_controller(detect_returns=None):
    """Create DetectionController with a mock detector."""
    mock_detector = MagicMock()
    if detect_returns is None:
        detect_returns = []
    mock_detector.detect.return_value = detect_returns

    ctrl = DetectionController(detector=mock_detector, max_hops=50)
    ctrl.reset({"instruction": "fly to the red car", "constraints": {}})
    return ctrl, mock_detector


class TestTrackMode:
    """Test TRACK mode: target detected, steer toward it."""

    def test_centered_detection_flies_straight(self):
        """A centered bbox (cx=320) should produce near-zero heading offset."""
        det = _detection_at(cx=320, cy=240, w=100, h=80, score=0.8)
        ctrl, mock = _make_controller([det])
        state = _make_state(x=0, y=0, z=-10, yaw_deg=0)

        action = ctrl.get_action(state)

        assert ctrl._nav_mode == "TRACK"
        dx = action.target_position[0] - state.position[0]
        dy = action.target_position[1] - state.position[1]
        heading_deg = math.degrees(math.atan2(dy, dx))
        assert abs(heading_deg) < 5.0, f"Expected ~straight, got heading={heading_deg:.1f}°"

    def test_left_detection_turns_left(self):
        """A left-of-center bbox (cx=80) should produce negative heading offset."""
        det = _detection_at(cx=80, cy=240, w=100, h=80, score=0.8)
        ctrl, mock = _make_controller([det])
        state = _make_state(x=0, y=0, z=-10, yaw_deg=0)

        action = ctrl.get_action(state)

        assert ctrl._nav_mode == "TRACK"
        dx = action.target_position[0] - state.position[0]
        dy = action.target_position[1] - state.position[1]
        heading_deg = math.degrees(math.atan2(dy, dx))
        assert heading_deg < -15.0, f"Expected left turn, got heading={heading_deg:.1f}°"

    def test_right_detection_turns_right(self):
        """A right-of-center bbox (cx=560) should produce positive heading offset."""
        det = _detection_at(cx=560, cy=240, w=100, h=80, score=0.8)
        ctrl, mock = _make_controller([det])
        state = _make_state(x=0, y=0, z=-10, yaw_deg=0)

        action = ctrl.get_action(state)

        assert ctrl._nav_mode == "TRACK"
        dx = action.target_position[0] - state.position[0]
        dy = action.target_position[1] - state.position[1]
        heading_deg = math.degrees(math.atan2(dy, dx))
        assert heading_deg > 15.0, f"Expected right turn, got heading={heading_deg:.1f}°"


class TestCoastMode:
    """Test COAST mode: target briefly lost, maintain heading."""

    def test_transitions_to_coast_after_detection_lost(self):
        """After TRACK, losing detection should enter COAST for K-1 hops."""
        det = _detection_at(cx=320, cy=240, score=0.8)
        ctrl, mock = _make_controller([det])
        state = _make_state()

        ctrl.get_action(state)
        assert ctrl._nav_mode == "TRACK"

        mock.detect.return_value = []
        ctrl.get_action(state)
        assert ctrl._nav_mode == "COAST"

    def test_coast_maintains_forward_motion(self):
        """COAST should produce forward motion along last heading."""
        det = _detection_at(cx=320, cy=240, score=0.8)
        ctrl, mock = _make_controller([det])
        state = _make_state(x=0, y=0, z=-10, yaw_deg=0)

        ctrl.get_action(state)
        mock.detect.return_value = []
        action = ctrl.get_action(state)

        assert ctrl._nav_mode == "COAST"
        dx = action.target_position[0] - state.position[0]
        assert dx > 0, "Expected forward motion in COAST"


class TestSearchMode:
    """Test SEARCH mode: target truly lost, systematic scanning."""

    def test_enters_search_after_k_misses(self):
        """After COAST_K hops without detection, should enter SEARCH."""
        ctrl, mock = _make_controller([])
        state = _make_state()

        for _ in range(COAST_K + 1):
            ctrl.get_action(state)

        assert ctrl._nav_mode == "SEARCH"

    def test_search_rotates(self):
        """SEARCH mode should produce lateral motion (rotation)."""
        ctrl, mock = _make_controller([])
        state = _make_state(x=0, y=0, z=-10, yaw_deg=0)

        for _ in range(4):
            action = ctrl.get_action(state)

        assert ctrl._nav_mode == "SEARCH"
        dy = action.target_position[1] - state.position[1]
        assert abs(dy) > 0.1, "Expected some lateral motion in SEARCH"

    def test_reacquire_returns_to_track(self):
        """Re-detecting the target after SEARCH should return to TRACK."""
        ctrl, mock = _make_controller([])
        state = _make_state()

        for _ in range(5):
            ctrl.get_action(state)
        assert ctrl._nav_mode == "SEARCH"

        det = _detection_at(cx=320, cy=240, score=0.8)
        mock.detect.return_value = [det]
        ctrl.get_action(state)
        assert ctrl._nav_mode == "TRACK"


class TestGoalReached:
    """Test is_goal_reached termination conditions."""

    def test_max_hops_triggers_goal(self):
        """is_goal_reached should be True after max_hops."""
        ctrl, mock = _make_controller([])
        ctrl._leg_max_hops = 5
        state = _make_state()

        for _ in range(5):
            ctrl.get_action(state)

        assert ctrl.is_goal_reached(state)

    def test_landing_triggers_goal(self):
        """3 consecutive large-bbox detections should trigger landing goal."""
        big_det = _detection_at(cx=320, cy=240, w=200, h=200, score=0.9)
        ctrl, mock = _make_controller([big_det])
        ctrl.reset({"instruction": "land on the red car", "constraints": {}})
        state = _make_state()

        for _ in range(4):
            ctrl.get_action(state)

        assert ctrl._consecutive_land >= 3
        assert ctrl.is_goal_reached(state)

    def test_not_reached_before_max(self):
        """Goal should not be reached before max_hops with no landing."""
        ctrl, mock = _make_controller([])
        ctrl._leg_max_hops = 10
        state = _make_state()

        for _ in range(5):
            ctrl.get_action(state)

        assert not ctrl.is_goal_reached(state)


class TestHoverNoImage:
    """Test hover behavior when no image is available."""

    def test_hover_on_no_image(self):
        """With no image, controller should hover at current position."""
        ctrl, mock = _make_controller([])
        state = _make_state(x=5, y=3, z=-12, with_image=False)

        action = ctrl.get_action(state)

        assert action.target_position == state.position
        assert action.velocity == 1.0


class TestHelpers:
    """Test helper methods."""

    def test_yaw_from_quaternion_zero(self):
        q = (1.0, 0.0, 0.0, 0.0)
        yaw = DetectionController._yaw_from_quaternion(q)
        assert abs(yaw) < 1e-6

    def test_yaw_from_quaternion_90(self):
        yaw_rad = math.pi / 2.0
        q = (math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))
        yaw = DetectionController._yaw_from_quaternion(q)
        assert abs(yaw - math.pi / 2.0) < 1e-6

    def test_clamp_altitude(self):
        ctrl, _ = _make_controller([])
        ctrl._constraints = {"min_altitude": 3.0, "max_altitude": 50.0}
        assert ctrl._clamp_altitude(-60.0) == -49.0
        assert ctrl._clamp_altitude(-1.0) == -4.0
        assert ctrl._clamp_altitude(-20.0) == -20.0
