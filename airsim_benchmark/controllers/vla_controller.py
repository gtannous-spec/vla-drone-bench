"""
vla_controller.py — OpenVLA Controller stub (Milestone 2).

Placeholder that demonstrates the controller interface.
Will be implemented with OpenVLA/NaVILA inference in a future milestone.
"""

import logging
from typing import Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)


class VLAController(BaseController):
    """Vision-Language-Action controller using OpenVLA inference.

    NOT YET IMPLEMENTED — this stub raises NotImplementedError.
    The interface shows how a VLA model will integrate:
      1. Receive camera image + language instruction
      2. Run VLA inference to predict action deltas
      3. Convert deltas to absolute position targets
    """

    def __init__(self, model_path: str = "", inference_hz: float = 2.0):
        self._model_path = model_path
        self._inference_hz = inference_hz
        self._instruction: str = ""
        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self, task_config: dict) -> None:
        self._instruction = task_config.get("instruction", "")
        self._goal = tuple(task_config["goal"])
        logger.info(f"VLAController reset — instruction: '{self._instruction}'")

    def get_action(self, state: DroneState) -> ControlAction:
        raise NotImplementedError(
            "VLA inference not implemented yet. "
            "This controller will be completed in Milestone 2."
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        raise NotImplementedError("VLA goal checking not implemented yet.")
