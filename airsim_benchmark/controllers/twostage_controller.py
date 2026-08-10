"""
twostage_controller.py — Two-Stage VLM Navigation Controller (GPS-free).

Stage 1 (Perceive): InternVL2 analyzes the camera image + instruction and
returns a heading direction, distance estimate, and confidence.

Stage 2 (Act): A classical controller steers the drone along the VLM's
suggested heading using small, correctable hops.

This controller does NOT require goal coordinates — it navigates purely
from camera images and text instructions, making it suitable for GPS-free
mission legs.

Architecture:
    Camera + Instruction → InternVL2 → "target is LEFT, ~30m, HIGH conf"
    → heading_offset applied to current yaw → small waypoint hop
    → AirSim moveToPosition → New image → Repeat

Landing is triggered when the VLM says the target is BELOW or very close
and the instruction contains "land".
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState
from ..core.vlm_inference import VLMInference, NavigationHint

logger = logging.getLogger(__name__)

LANDING_PROMPT = """\
You are the navigation system of a drone flying over a suburban neighborhood.
You see this image from the front-facing camera.

Task: {instruction}

Is the drone close enough to begin landing? Respond in EXACTLY this format:
LAND: <YES or NO>
ALTITUDE_ACTION: <DESCEND, MAINTAIN, or CLIMB>
REASONING: <one sentence>"""


class TwoStageController(BaseController):
    """GPS-free two-stage VLM navigation controller.

    Works without goal coordinates. The VLM provides heading corrections
    at each hop, and the controller steers the drone accordingly. Landing
    is detected by the VLM when the target is close/below.

    Args:
        model_path: HuggingFace model ID for the VLM.
        nav_speed: Cruise speed (m/s).
        hop_distance: Distance per hop (m) — kept small for course correction.
        max_hops: Maximum navigation iterations per leg.
        query_interval: Query the VLM every N hops (save compute).
        device: CUDA device.
    """

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3-8B-hf",
        nav_speed: float = 5.0,
        hop_distance: float = 5.0,
        max_hops: int = 50,
        query_interval: int = 1,
        device: str = "auto",
    ):
        self._nav_speed = nav_speed
        self._hop_distance = hop_distance
        self._max_hops = max_hops
        self._query_interval = max(1, query_interval)

        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")

        self._vlm = VLMInference(model_path=model_path, device=device)

        self._instruction: str = ""
        self._constraints: dict = {}
        self._task_id: int = 0
        self._hop_count: int = 0
        self._vlm_guided_hops: int = 0
        self._last_hint: Optional[NavigationHint] = None
        self._consecutive_land: int = 0
        self._goal: Optional[Tuple[float, float, float]] = None
        self._leg_max_hops: int = max_hops

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config.get("id", 0)
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._vlm_guided_hops = 0
        self._last_hint = None
        self._consecutive_land = 0
        self._goal = tuple(task_config["goal"]) if "goal" in task_config else None
        self._leg_max_hops = task_config.get("max_hops", self._max_hops)

        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)

        mode = "GPS-free" if self._goal is None else f"goal={self._goal}"
        logger.info(
            f"TwoStageController reset — task {self._task_id}, "
            f"instruction='{self._instruction}', {mode}, "
            f"hop_dist={self._hop_distance}m"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1

        if state.image is None:
            logger.warning("No camera image — hover")
            return self._hover_action(state)

        should_query = (
            self._hop_count % self._query_interval == 0
            or self._last_hint is None
            or self._hop_count <= 3
        )

        if should_query:
            hint = self._vlm.query(state.image, self._instruction)
            self._last_hint = hint
            self._vlm_guided_hops += 1
        else:
            hint = self._last_hint

        current_yaw_rad = self._yaw_from_quaternion(state.orientation)
        current_yaw_deg = math.degrees(current_yaw_rad)

        wants_land = self._instruction_wants_landing()
        if wants_land and hint.confidence >= 0.3:
            land_hint = self._check_landing(state)
            if land_hint:
                self._consecutive_land += 1
                if self._consecutive_land >= 2:
                    logger.info(
                        f"TwoStage hop {self._hop_count}: LANDING "
                        f"(2 consecutive land signals)"
                    )
                    return self._descend_action(state)
            else:
                self._consecutive_land = 0

        if hint.confidence < 0.1:
            logger.info(
                f"TwoStage hop {self._hop_count}: NO confidence — "
                f"rotating to search, reason='{hint.reasoning[:60]}'"
            )
            return self._search_action(state, current_yaw_deg)

        if hint.confidence < 0.3:
            logger.info(
                f"TwoStage hop {self._hop_count}: LOW confidence — "
                f"small forward hop, reason='{hint.reasoning[:60]}'"
            )
            return self._forward_action(state, current_yaw_rad, 2.0)

        heading_offset_rad = math.radians(hint.heading_offset_deg)
        target_yaw = current_yaw_rad + heading_offset_rad

        dist_est = hint.distance_estimate_m
        step = min(self._hop_distance, max(2.0, dist_est * 0.3))

        target_x = state.position[0] + step * math.cos(target_yaw)
        target_y = state.position[1] + step * math.sin(target_yaw)

        alt_action = self._decide_altitude(state, hint)
        target_z = self._clamp_altitude(state.position[2] + alt_action)

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"TwoStage hop {self._hop_count}: "
                f"heading={hint.heading_offset_deg:+.0f}°, "
                f"conf={hint.confidence:.1f}, "
                f"dist={dist_est:.0f}m, "
                f"step={step:.1f}m, "
                f"alt_delta={alt_action:+.1f}m, "
                f"reason='{hint.reasoning[:60]}'"
            )

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        if self._hop_count >= self._leg_max_hops:
            logger.info(
                f"Max hops ({self._leg_max_hops}) reached — "
                f"VLM guided {self._vlm_guided_hops}/{self._hop_count}"
            )
            return True

        if self._consecutive_land >= 3:
            logger.info(
                f"Landing complete — 3 consecutive land signals, "
                f"VLM guided {self._vlm_guided_hops}/{self._hop_count}"
            )
            return True

        if self._goal is not None:
            dist = self._distance_to_goal(state)
            if dist < 5.0:
                logger.info(f"Goal reached at {dist:.1f}m")
                return True

        return False

    def _instruction_wants_landing(self) -> bool:
        lower = self._instruction.lower()
        return "land" in lower

    def _check_landing(self, state: DroneState) -> bool:
        """Ask the VLM if it's time to land."""
        if state.image is None:
            return False
        prompt = LANDING_PROMPT.format(instruction=self._instruction)
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(state.image)

            if self._vlm._model_family == "internvl":
                response = self._vlm._query_internvl(pil_img, prompt)
            else:
                response = self._vlm._query_llava(pil_img, prompt)

            should_land = "LAND: YES" in response.upper()
            if should_land:
                logger.info(f"TwoStage: VLM says LAND YES — {response[:100]}")
            return should_land
        except Exception as e:
            logger.debug(f"Landing check failed: {e}")
            return False

    def _search_action(self, state: DroneState, current_yaw_deg: float) -> ControlAction:
        """Rotate in place to search for the target."""
        search_yaw_rad = math.radians(current_yaw_deg + 30.0)
        creep = 2.0
        target_x = state.position[0] + creep * math.cos(search_yaw_rad)
        target_y = state.position[1] + creep * math.sin(search_yaw_rad)
        return ControlAction(
            target_position=(target_x, target_y, state.position[2]),
            velocity=2.0,
        )

    def _forward_action(self, state: DroneState, yaw_rad: float,
                        dist: float) -> ControlAction:
        """Small hop forward along current heading."""
        target_x = state.position[0] + dist * math.cos(yaw_rad)
        target_y = state.position[1] + dist * math.sin(yaw_rad)
        return ControlAction(
            target_position=(target_x, target_y, state.position[2]),
            velocity=self._effective_speed,
        )

    def _hover_action(self, state: DroneState) -> ControlAction:
        return ControlAction(
            target_position=state.position,
            velocity=1.0,
        )

    def _descend_action(self, state: DroneState) -> ControlAction:
        """Descend 2m toward the ground/rooftop."""
        target_z = min(state.position[2] + 2.0, -1.5)
        return ControlAction(
            target_position=(state.position[0], state.position[1], target_z),
            velocity=2.0,
        )

    def _decide_altitude(self, state: DroneState, hint: NavigationHint) -> float:
        """Decide altitude change based on VLM reasoning."""
        reasoning = hint.reasoning.lower()
        if "above" in reasoning or "below" in reasoning or "descend" in reasoning:
            return 1.5
        if "climb" in reasoning or "higher" in reasoning:
            return -2.0
        return 0.0

    def _distance_to_goal(self, state: DroneState) -> float:
        if self._goal is None:
            return float("inf")
        dx = state.position[0] - self._goal[0]
        dy = state.position[1] - self._goal[1]
        dz = state.position[2] - self._goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _clamp_altitude(self, z_ned: float) -> float:
        min_alt = self._constraints.get("min_altitude", 3.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        return max(-(max_alt - 1.0), min(-(min_alt + 1.0), z_ned))

    @staticmethod
    def _yaw_from_quaternion(q: Tuple[float, float, float, float]) -> float:
        w, x, y, z = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
