"""Tests for the AugmentedController — detection-augmented VLA pipeline.

Tests the integration of instruction parser, subtask FSM, and multi-object
detection into a unified navigation controller. Uses mock detections to
avoid requiring GPU or GroundingDINO.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from airsim_benchmark.controllers.base_controller import ControlAction, DroneState
from airsim_benchmark.controllers.augmented_controller import (
    AugmentedController,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)
from airsim_benchmark.core.detection_inference import Detection
from airsim_benchmark.core.instruction_parser import Subtask
from airsim_benchmark.core.subtask_fsm import SubtaskFSM


def _make_state(
    x=0.0, y=0.0, z=-10.0, yaw_deg=0.0, with_image=True,
) -> DroneState:
    yaw_rad = math.radians(yaw_deg)
    w = math.cos(yaw_rad / 2)
    z_q = math.sin(yaw_rad / 2)
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) if with_image else None
    return DroneState(
        position=(x, y, z),
        velocity=(0.0, 0.0, 0.0),
        orientation=(w, 0.0, 0.0, z_q),
        image=img,
    )


def _make_detection(
    cx=320.0, cy=240.0, w=50.0, h=50.0, score=0.8, phrase="car",
) -> Detection:
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return Detection(bbox_xyxy=(x1, y1, x2, y2), score=score, phrase=phrase)


class TestAugmentedControllerInit(unittest.TestCase):
    """Test controller initialization and reset."""

    def test_reset_parses_instruction_into_subtasks(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl.reset({"instruction": "fly to the red car"})

        self.assertGreater(len(ctrl.subtasks), 0)
        self.assertIsNotNone(ctrl.subtask_fsm)
        self.assertFalse(ctrl.subtask_fsm.is_complete)

    def test_reset_multi_step_instruction(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl.reset({
            "instruction": "fly to the red car then land on the rooftop",
        })

        self.assertGreaterEqual(len(ctrl.subtasks), 2)

    def test_reset_clears_state(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl.reset({"instruction": "fly to the red car"})
        self.assertEqual(ctrl._hop_count, 0)
        self.assertEqual(ctrl._nav_mode, "SEARCH")
        self.assertEqual(ctrl._consecutive_land, 0)


class TestAugmentedControllerNavigation(unittest.TestCase):
    """Test navigation behavior with mock detections."""

    def _make_controller(self):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        ctrl = AugmentedController(detector=mock_detector)
        return ctrl, mock_detector

    def test_search_when_no_detection(self):
        ctrl, mock_det = self._make_controller()
        ctrl.reset({"instruction": "fly to the red car"})

        state = _make_state()
        action = ctrl.get_action(state)

        self.assertIsInstance(action, ControlAction)
        self.assertEqual(ctrl._nav_mode, "SEARCH")

    def test_track_when_detection_found(self):
        ctrl, mock_det = self._make_controller()
        ctrl.reset({"instruction": "fly to the red car"})

        det = _make_detection(cx=400, score=0.9)
        mock_det.detect.return_value = [det]

        state = _make_state()
        action = ctrl.get_action(state)

        self.assertEqual(ctrl._nav_mode, "TRACK")
        self.assertGreater(ctrl._memory_heading, 0)

    def test_coast_after_lost_detection(self):
        ctrl, mock_det = self._make_controller()
        ctrl.reset({"instruction": "fly to the red car"})

        det = _make_detection(cx=320, score=0.7)
        mock_det.detect.return_value = [det]
        ctrl.get_action(_make_state())
        self.assertEqual(ctrl._nav_mode, "TRACK")

        mock_det.detect.return_value = []
        ctrl.get_action(_make_state())
        self.assertEqual(ctrl._nav_mode, "COAST")

    def test_search_after_coast_expires(self):
        ctrl, mock_det = self._make_controller()
        ctrl.reset({"instruction": "fly to the red car"})

        det = _make_detection(score=0.7)
        mock_det.detect.return_value = [det]
        ctrl.get_action(_make_state())

        mock_det.detect.return_value = []
        for _ in range(5):
            ctrl.get_action(_make_state())

        self.assertEqual(ctrl._nav_mode, "SEARCH")

    def test_hover_when_no_image(self):
        ctrl, mock_det = self._make_controller()
        ctrl.reset({"instruction": "fly to the red car"})

        state = _make_state(with_image=False)
        action = ctrl.get_action(state)

        self.assertEqual(action.target_position, state.position)

    def test_hover_when_all_subtasks_complete(self):
        ctrl, mock_det = self._make_controller()
        ctrl.reset({"instruction": "fly to the red car"})
        ctrl._fsm._current_index = ctrl._fsm.total_subtasks

        state = _make_state()
        action = ctrl.get_action(state)

        self.assertEqual(action.target_position, state.position)


class TestSubtaskAdvancement(unittest.TestCase):
    """Test that the controller advances through subtasks correctly."""

    def test_navigate_subtask_completes_on_close_detection(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)
        ctrl.reset({
            "instruction": "fly to the red car then land on the rooftop",
        })

        initial_subtask = ctrl.subtask_fsm.current_subtask
        self.assertIn(initial_subtask.detect, ["red car", "rooftop"])

        large_det = _make_detection(
            cx=320, cy=240, w=200, h=200, score=0.9,
        )
        mock_detector.detect.return_value = [large_det]

        for _ in range(5):
            ctrl.get_action(_make_state())

        if ctrl.subtask_fsm.current_index > 0:
            self.assertNotEqual(
                ctrl.subtask_fsm.current_subtask.detect,
                initial_subtask.detect,
            )

    def test_goal_reached_when_all_subtasks_done(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)
        ctrl.reset({"instruction": "fly to the red car"})

        ctrl._fsm._current_index = ctrl._fsm.total_subtasks

        self.assertTrue(ctrl.is_goal_reached(_make_state()))

    def test_goal_reached_on_max_hops(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector, max_hops=5)
        ctrl.reset({"instruction": "fly to the red car"})

        mock_detector.detect.return_value = []
        for _ in range(5):
            ctrl.get_action(_make_state())

        self.assertTrue(ctrl.is_goal_reached(_make_state()))

    def test_goal_reached_on_consecutive_landing(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)
        ctrl.reset({"instruction": "fly to the red car"})
        ctrl._consecutive_land = 3

        self.assertTrue(ctrl.is_goal_reached(_make_state(z=-3.0)))


class TestHoverAndCircleActions(unittest.TestCase):
    """Test stationary (hover/inspect) and circle subtask handling."""

    def test_hover_subtask_stays_in_place(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl._subtasks = [Subtask(action="hover", detect="parking lot")]
        ctrl._fsm = SubtaskFSM(ctrl._subtasks, hover_hops=3)
        ctrl._hop_count = 0
        ctrl._nav_mode = "SEARCH"
        ctrl._hops_since_detection = 999

        state = _make_state(x=10.0, y=20.0, z=-10.0)
        action = ctrl.get_action(state)

        self.assertEqual(action.target_position, state.position)

    def test_hover_subtask_completes_after_n_hops(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl._subtasks = [
            Subtask(action="hover", detect="parking lot"),
            Subtask(action="navigate", detect="red car"),
        ]
        ctrl._fsm = SubtaskFSM(ctrl._subtasks, hover_hops=2)
        ctrl._hop_count = 0
        ctrl._nav_mode = "SEARCH"
        ctrl._constraints = {}
        ctrl._hops_since_detection = 999
        ctrl._instruction = "hover over parking lot then fly to red car"

        mock_detector.detect.return_value = []

        for _ in range(3):
            ctrl.get_action(_make_state())

        self.assertEqual(ctrl._fsm.current_index, 1)

    def test_circle_subtask_moves_laterally(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl._subtasks = [Subtask(action="circle", detect="house")]
        ctrl._fsm = SubtaskFSM(ctrl._subtasks, circle_hops=12)
        ctrl._hop_count = 0
        ctrl._nav_mode = "SEARCH"
        ctrl._constraints = {}
        ctrl._hops_since_detection = 999

        state = _make_state(x=0.0, y=0.0)
        action = ctrl.get_action(state)

        self.assertNotEqual(action.target_position[:2], (0.0, 0.0))


class TestNearbyConstraint(unittest.TestCase):
    """Test multi-object spatial proximity checking."""

    def test_nearby_check_passes_when_close(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        primary = _make_detection(cx=200, cy=240, score=0.8, phrase="car")
        nearby = _make_detection(cx=250, cy=240, score=0.7, phrase="building")

        mock_detector.detect.side_effect = lambda img, q: (
            [primary] if "car" in q else [nearby]
        )

        ctrl.reset({"instruction": "fly to the car near the building"})

        image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        subtask = Subtask(action="navigate", detect="car", nearby="building")

        result = ctrl._check_nearby(image, subtask, primary)
        self.assertTrue(result)

    def test_nearby_check_fails_when_not_detected(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        primary = _make_detection(cx=200, cy=240, score=0.8, phrase="car")
        mock_detector.detect.return_value = []

        ctrl.reset({"instruction": "fly to the car near the building"})

        image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        subtask = Subtask(action="navigate", detect="car", nearby="building")

        result = ctrl._check_nearby(image, subtask, primary)
        self.assertFalse(result)


class TestLandingBehavior(unittest.TestCase):
    """Test landing-related logic."""

    def test_descend_action_lowers_altitude(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)
        ctrl.reset({"instruction": "land on the rooftop"})

        state = _make_state(z=-10.0)
        action = ctrl._descend_action(state)

        self.assertGreater(action.target_position[2], state.position[2])

    def test_landing_triggers_on_large_bbox(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl._subtasks = [Subtask(action="land", detect="rooftop", done_when="proximity")]
        ctrl._fsm = SubtaskFSM(ctrl._subtasks)
        ctrl._hop_count = 0
        ctrl._nav_mode = "SEARCH"
        ctrl._memory_heading = 0.0
        ctrl._memory_confidence = 0.0
        ctrl._hops_since_detection = 999
        ctrl._consecutive_land = 0
        ctrl._search_hops = 0
        ctrl._search_revolutions = 0
        ctrl._constraints = {}
        ctrl._instruction = "land on the rooftop"

        large_det = _make_detection(
            cx=320, cy=350, w=300, h=200, score=0.9, phrase="rooftop",
        )
        mock_detector.detect.return_value = [large_det]

        state = _make_state(z=-8.0)
        ctrl.get_action(state)
        action = ctrl.get_action(state)

        self.assertGreater(action.target_position[2], state.position[2])


class TestStateProperties(unittest.TestCase):
    """Test controller state accessors."""

    def test_subtask_fsm_property(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)
        self.assertIsNone(ctrl.subtask_fsm)

        ctrl.reset({"instruction": "fly to the red car"})
        self.assertIsNotNone(ctrl.subtask_fsm)

    def test_subtasks_property(self):
        mock_detector = MagicMock()
        ctrl = AugmentedController(detector=mock_detector)

        ctrl.reset({"instruction": "fly to the red car"})
        subtasks = ctrl.subtasks
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 0)
        self.assertIsInstance(subtasks[0], Subtask)


if __name__ == "__main__":
    unittest.main()
