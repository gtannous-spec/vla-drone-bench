"""
base_controller.py — Abstract controller interface.

All navigation controllers (classical, VLA, DRL) implement this ABC,
allowing the FSM and benchmark runner to remain controller-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class DroneState:
    """Current drone state passed to controllers each tick."""
    position: Tuple[float, float, float]      # (x, y, z) NED metres
    velocity: Tuple[float, float, float]      # (vx, vy, vz) NED m/s
    orientation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    image: Optional[np.ndarray] = field(default=None, repr=False)  # RGB (H,W,3)


@dataclass
class ControlAction:
    """Action output from a controller."""
    target_position: Tuple[float, float, float]  # (x, y, z) NED
    velocity: float = 5.0                        # cruise speed m/s


class BaseController(ABC):
    """Abstract base class for drone navigation controllers.

    Lifecycle per task:
        1. reset(task_config) — called before FSM starts
        2. get_action(state) — called each navigation tick
        3. is_goal_reached(state) — checked after each action
    """

    @abstractmethod
    def reset(self, task_config: dict) -> None:
        """Initialize controller for a new task.

        Args:
            task_config: Dictionary with keys 'id', 'instruction', 'start',
                        'goal', 'constraints' from benchmark_config.yaml.
        """
        ...

    @abstractmethod
    def get_action(self, state: DroneState) -> ControlAction:
        """Compute the next navigation action given current state.

        Args:
            state: Current drone state including position and optional camera image.

        Returns:
            ControlAction specifying the target position and speed.
        """
        ...

    @abstractmethod
    def is_goal_reached(self, state: DroneState) -> bool:
        """Check whether the task goal has been achieved.

        Args:
            state: Current drone state.

        Returns:
            True if the drone is within arrival tolerance of the goal.
        """
        ...
