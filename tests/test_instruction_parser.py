"""Tests for the instruction parser module.

Tests rule-based parsing and JSON subtask parsing only (no GPU required).
"""

import pytest

from airsim_benchmark.core.instruction_parser import (
    Subtask,
    parse_instruction,
    parse_json_subtasks,
)


class TestSubtask:
    def test_subtask_fields(self):
        s = Subtask(action="navigate", detect="red car")
        assert s.action == "navigate"
        assert s.detect == "red car"
        assert s.nearby is None
        assert s.done_when == "close"

    def test_subtask_with_all_fields(self):
        s = Subtask(action="land", detect="building", nearby="parking lot", done_when="proximity")
        assert s.action == "land"
        assert s.detect == "building"
        assert s.nearby == "parking lot"
        assert s.done_when == "proximity"


class TestRuleBasedParsing:
    def test_simple_navigate(self):
        result = parse_instruction("Fly toward the red car")
        assert len(result) == 1
        assert result[0].action == "navigate"
        assert "red car" in result[0].detect.lower() or "car" in result[0].detect.lower()

    def test_navigate_then_land(self):
        result = parse_instruction("Fly toward the red car, then land on the closest rooftop")
        assert len(result) == 2
        assert result[0].action == "navigate"
        assert result[1].action == "land"

    def test_land_instruction(self):
        result = parse_instruction("Land on the rooftop near the red car")
        assert len(result) == 1
        assert result[0].action == "land"

    def test_multi_step(self):
        result = parse_instruction("Navigate to the white house, then circle it, then land")
        assert len(result) == 3
        assert result[0].action == "navigate"
        assert result[1].action == "circle"
        assert result[2].action == "land"

    def test_hover_action(self):
        result = parse_instruction("Hover over the intersection")
        assert len(result) == 1
        assert result[0].action == "hover"

    def test_inspect_action(self):
        result = parse_instruction("Inspect the rooftop")
        assert len(result) == 1
        assert result[0].action == "inspect"

    def test_and_then_split(self):
        result = parse_instruction("Fly to the house and then land on the driveway")
        assert len(result) == 2
        assert result[0].action == "navigate"
        assert result[1].action == "land"

    def test_land_sets_proximity_done_when(self):
        result = parse_instruction("Land on the rooftop")
        assert result[0].done_when == "proximity"

    def test_navigate_sets_close_done_when(self):
        result = parse_instruction("Fly to the red car")
        assert result[0].done_when == "close"

    def test_nearby_extraction(self):
        result = parse_instruction("Land on the rooftop near the red car")
        assert result[0].nearby == "red car"


class TestJsonParsing:
    def test_clean_json(self):
        json_str = '[{"action": "navigate", "detect": "blue truck", "nearby": "parking lot", "done_when": "close"}]'
        result = parse_json_subtasks(json_str)
        assert len(result) == 1
        assert result[0].detect == "blue truck"
        assert result[0].nearby == "parking lot"

    def test_json_in_markdown(self):
        text = '```json\n[{"action": "land", "detect": "building"}]\n```'
        result = parse_json_subtasks(text)
        assert len(result) == 1
        assert result[0].action == "land"

    def test_malformed_returns_empty(self):
        result = parse_json_subtasks("this is not json at all")
        assert result == []

    def test_missing_fields_use_defaults(self):
        json_str = '[{"action": "navigate", "detect": "car"}]'
        result = parse_json_subtasks(json_str)
        assert result[0].done_when == "close"
        assert result[0].nearby is None

    def test_multiple_subtasks(self):
        json_str = """[
            {"action": "navigate", "detect": "white house", "done_when": "close"},
            {"action": "circle", "detect": "white house", "done_when": "close"},
            {"action": "land", "detect": "building", "done_when": "proximity"}
        ]"""
        result = parse_json_subtasks(json_str)
        assert len(result) == 3
        assert result[2].done_when == "proximity"

    def test_json_with_extra_text(self):
        text = 'Here is the plan:\n[{"action": "navigate", "detect": "red car"}]\nDone!'
        result = parse_json_subtasks(text)
        assert len(result) == 1
        assert result[0].detect == "red car"

    def test_empty_detect_skipped(self):
        json_str = '[{"action": "navigate", "detect": ""}, {"action": "land", "detect": "house"}]'
        result = parse_json_subtasks(json_str)
        assert len(result) == 1
        assert result[0].action == "land"
