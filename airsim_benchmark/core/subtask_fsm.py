"""Subtask FSM — sequences through parser-generated subtasks.

Sits above the flight-phase FSM (drone_fsm.py) and below the controller.
The instruction parser produces an ordered list of Subtask objects; this FSM
tracks which subtask is active and decides when to advance based on sensor
feedback (bbox area, altitude).
"""

from __future__ import annotations

from typing import List, Optional

try:
    from .instruction_parser import Subtask
except ImportError:

    from dataclasses import dataclass

    @dataclass
    class Subtask:
        action: str
        detect: str
        nearby: Optional[str] = None
        done_when: str = "close"


class SubtaskFSM:
    """Sequences through parser-generated subtasks, advancing when each
    subtask's completion condition is met.

    Args:
        subtasks: Ordered list of Subtask objects from the instruction parser.
        proximity_threshold: Bbox area ratio that means "close" (default 0.03).
        landing_area_threshold: Bbox area ratio for landing trigger (default 0.05).
        hover_hops: Number of hops to hover for "hover"/"inspect" actions (default 5).
        circle_hops: Number of hops for "circle" action (default 12).
    """

    def __init__(
        self,
        subtasks: List[Subtask],
        proximity_threshold: float = 0.03,
        landing_area_threshold: float = 0.05,
        hover_hops: int = 5,
        circle_hops: int = 12,
    ):
        self._subtasks = subtasks
        self._current_index = 0
        self._proximity_threshold = proximity_threshold
        self._landing_area_threshold = landing_area_threshold
        self._hover_hops = hover_hops
        self._circle_hops = circle_hops
        self._hops_in_current = 0

    @property
    def current_subtask(self) -> Optional[Subtask]:
        """The current active subtask, or None if all complete."""
        if self._current_index >= len(self._subtasks):
            return None
        return self._subtasks[self._current_index]

    @property
    def is_complete(self) -> bool:
        """True when all subtasks are done."""
        return self._current_index >= len(self._subtasks)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def total_subtasks(self) -> int:
        return len(self._subtasks)

    def advance(self) -> Optional[Subtask]:
        """Move to the next subtask. Returns the new current subtask or None."""
        self._current_index += 1
        self._hops_in_current = 0
        return self.current_subtask

    def check_completion(
        self,
        bbox_area_ratio: float = 0.0,
        altitude_ned: float = -10.0,
        min_altitude: float = 4.0,
    ) -> bool:
        """Check if the current subtask is complete based on sensor data.

        Args:
            bbox_area_ratio: Area ratio of detected target bbox (0 if not detected).
            altitude_ned: Current drone altitude in NED (negative = above ground).
            min_altitude: Mission minimum altitude constraint.

        Returns:
            True if the current subtask should be marked complete.
        """
        self._hops_in_current += 1
        subtask = self.current_subtask
        if subtask is None:
            return True

        if subtask.action == "navigate":
            return bbox_area_ratio > self._proximity_threshold

        elif subtask.action == "land":
            at_floor = altitude_ned >= -(min_altitude + 1.0)
            return bbox_area_ratio > self._landing_area_threshold and at_floor

        elif subtask.action in ("hover", "inspect"):
            return self._hops_in_current >= self._hover_hops

        elif subtask.action == "circle":
            return self._hops_in_current >= self._circle_hops

        return False

    def reset(self):
        """Reset to the beginning."""
        self._current_index = 0
        self._hops_in_current = 0
