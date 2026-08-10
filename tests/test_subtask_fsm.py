"""Tests for SubtaskFSM — pure-Python, no GPU or AirSim required."""

from dataclasses import dataclass
from typing import Optional

import pytest

try:
    from airsim_benchmark.core.instruction_parser import Subtask
except ImportError:

    @dataclass
    class Subtask:
        action: str
        detect: str
        nearby: Optional[str] = None
        done_when: str = "close"


from airsim_benchmark.core.subtask_fsm import SubtaskFSM


class TestSubtaskFSM:
    def test_single_navigate(self):
        fsm = SubtaskFSM([Subtask(action="navigate", detect="red car")])
        assert fsm.current_subtask.detect == "red car"
        assert not fsm.is_complete
        assert not fsm.check_completion(bbox_area_ratio=0.01)
        assert fsm.check_completion(bbox_area_ratio=0.04)
        fsm.advance()
        assert fsm.is_complete

    def test_navigate_then_land(self):
        fsm = SubtaskFSM([
            Subtask(action="navigate", detect="red car"),
            Subtask(action="land", detect="building", done_when="proximity"),
        ])
        assert fsm.current_index == 0
        assert fsm.check_completion(bbox_area_ratio=0.04)
        fsm.advance()

        assert fsm.current_index == 1
        assert fsm.current_subtask.action == "land"
        assert not fsm.check_completion(bbox_area_ratio=0.06, altitude_ned=-10.0)
        assert fsm.check_completion(
            bbox_area_ratio=0.06, altitude_ned=-4.5, min_altitude=4.0
        )

    def test_hover_completes_by_hops(self):
        fsm = SubtaskFSM(
            [Subtask(action="hover", detect="car")], hover_hops=3
        )
        assert not fsm.check_completion()
        assert not fsm.check_completion()
        assert fsm.check_completion()

    def test_circle_completes_by_hops(self):
        fsm = SubtaskFSM(
            [Subtask(action="circle", detect="house")], circle_hops=4
        )
        for _ in range(3):
            assert not fsm.check_completion()
        assert fsm.check_completion()

    def test_empty_subtasks(self):
        fsm = SubtaskFSM([])
        assert fsm.is_complete
        assert fsm.current_subtask is None

    def test_three_step_mission(self):
        fsm = SubtaskFSM(
            [
                Subtask(action="navigate", detect="white house"),
                Subtask(action="circle", detect="white house"),
                Subtask(action="land", detect="building"),
            ],
            circle_hops=2,
        )

        assert fsm.current_subtask.action == "navigate"
        fsm.check_completion(bbox_area_ratio=0.04)
        fsm.advance()

        assert fsm.current_subtask.action == "circle"
        fsm.check_completion()
        assert not fsm.is_complete
        fsm.check_completion()
        fsm.advance()

        assert fsm.current_subtask.action == "land"
        fsm.check_completion(
            bbox_area_ratio=0.06, altitude_ned=-4.5, min_altitude=4.0
        )
        fsm.advance()

        assert fsm.is_complete

    def test_reset(self):
        fsm = SubtaskFSM([Subtask(action="navigate", detect="car")])
        fsm.check_completion(bbox_area_ratio=0.04)
        fsm.advance()
        assert fsm.is_complete
        fsm.reset()
        assert not fsm.is_complete
        assert fsm.current_index == 0

    def test_total_subtasks(self):
        fsm = SubtaskFSM([
            Subtask(action="navigate", detect="a"),
            Subtask(action="land", detect="b"),
        ])
        assert fsm.total_subtasks == 2

    def test_inspect_completes_like_hover(self):
        """inspect should use hover_hops, same as hover."""
        fsm = SubtaskFSM(
            [Subtask(action="inspect", detect="bridge")], hover_hops=2
        )
        assert not fsm.check_completion()
        assert fsm.check_completion()

    def test_unknown_action_never_completes(self):
        fsm = SubtaskFSM([Subtask(action="unknown_action", detect="x")])
        for _ in range(20):
            assert not fsm.check_completion()

    def test_advance_past_end(self):
        fsm = SubtaskFSM([Subtask(action="navigate", detect="car")])
        fsm.advance()
        assert fsm.is_complete
        result = fsm.advance()
        assert result is None
        assert fsm.check_completion()
