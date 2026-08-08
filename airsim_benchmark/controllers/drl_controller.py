"""
drl_controller.py — Deep Reinforcement Learning Controller stub (Milestone 3).

Placeholder that demonstrates the controller interface.
Will wrap a trained DRL policy network in a future milestone.
"""

import logging
from typing import Tuple

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)


class DRLController(BaseController):
    """DRL policy controller for learned navigation.

    NOT YET IMPLEMENTED — this stub raises NotImplementedError.
    The interface shows how a DRL policy will integrate:
      1. Encode observation (position, velocity, optional image)
      2. Forward pass through policy network
      3. Decode action into position/velocity commands
    """

    def __init__(self, policy_path: str = "", device: str = "cuda"):
        self._policy_path = policy_path
        self._device = device
        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self, task_config: dict) -> None:
        self._goal = tuple(task_config["goal"])
        logger.info(f"DRLController reset — goal={self._goal}")

    def get_action(self, state: DroneState) -> ControlAction:
        raise NotImplementedError(
            "DRL policy inference not implemented yet. "
            "This controller will be completed in Milestone 3."
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        raise NotImplementedError("DRL goal checking not implemented yet.")
