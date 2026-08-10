"""
augmented_controller.py — Detection-Augmented VLA Controller.

Integrates the full AI-parsed detection pipeline:
  1. Instruction parser decomposes text into ordered subtasks
  2. SubtaskFSM sequences through subtasks
  3. GroundingDINO detects objects per-subtask (multi-object spatial queries)
  4. TRACK/COAST/SEARCH state machine steers the drone

For multi-object subtasks (e.g. "land near X where Y is visible"), uses
detect_multi() and check_spatial_proximity() to verify both objects
before advancing.

Pipeline per hop:
  instruction → parse_instruction() → SubtaskFSM.current_subtask
  → detect(image, subtask.detect) → TRACK/COAST/SEARCH → ControlAction
  → SubtaskFSM.check_completion() → maybe advance to next subtask
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState
from ..core.detection_inference import (
    Detection,
    ObjectDetector,
    bbox_to_heading_offset,
    bbox_to_distance_estimate,
    check_spatial_proximity,
)
from ..core.instruction_parser import Subtask, parse_instruction
from ..core.subtask_fsm import SubtaskFSM
from ..core.target_phrase import _build_multi_query

logger = logging.getLogger(__name__)

COAST_K = 3
SEARCH_YAW_STEP = 30.0
SEARCH_ADVANCE_DISTANCE = 10.0
HEADING_EMA_ALPHA = 0.7
CONFIDENCE_DECAY = 0.8
CAMERA_FOV_DEG = 90.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SEARCH_ALTITUDE_CAP_NED = -25.0
SEARCH_ALTITUDE_STEP = 3.0


class AugmentedController(BaseController):
    """Detection-augmented VLA controller with AI instruction parsing.

    Combines instruction decomposition, multi-object detection, and
    subtask sequencing into a single GPS-free navigation controller.

    Args:
        detector: Pre-constructed ObjectDetector (for testing).
        model_id: GroundingDINO model ID if detector is None.
        parser_model: Optional VLM for LLM-based instruction parsing.
        nav_speed: Cruise speed (m/s).
        hop_distance: Max distance per hop (m).
        max_hops: Default max hops per mission.
        device: CUDA device string.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        model_id: str = "IDEA-Research/grounding-dino-base",
        parser_model=None,
        nav_speed: float = 5.0,
        hop_distance: float = 5.0,
        max_hops: int = 50,
        device: str = "auto",
    ):
        self._nav_speed = nav_speed
        self._hop_distance = hop_distance
        self._max_hops = max_hops
        self._parser_model = parser_model

        if detector is not None:
            self._detector = detector
        else:
            if device == "auto":
                try:
                    import torch
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            self._detector = ObjectDetector(model_id=model_id, device=device)

        self._instruction: str = ""
        self._constraints: dict = {}
        self._subtasks: List[Subtask] = []
        self._fsm: Optional[SubtaskFSM] = None
        self._hop_count: int = 0
        self._leg_max_hops: int = max_hops

        self._nav_mode: str = "SEARCH"
        self._memory_heading: float = 0.0
        self._memory_confidence: float = 0.0
        self._hops_since_detection: int = 999
        self._consecutive_land: int = 0

        self._search_hops: int = 0
        self._search_revolutions: int = 0

        self._approaching_reference: bool = False
        self._reference_reached: bool = False

    def reset(self, task_config: dict) -> None:
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._leg_max_hops = task_config.get("max_hops", self._max_hops)

        self._subtasks = parse_instruction(
            self._instruction, model=self._parser_model,
        )
        self._fsm = SubtaskFSM(self._subtasks)

        self._hop_count = 0
        self._nav_mode = "SEARCH"
        self._memory_heading = 0.0
        self._memory_confidence = 0.0
        self._hops_since_detection = 999
        self._consecutive_land = 0
        self._search_hops = 0
        self._search_revolutions = 0
        self._approaching_reference = False
        self._reference_reached = False

        subtask_summary = [
            f"{i}: {s.action}({s.detect}" + (f", near={s.nearby}" if s.nearby else "") + ")"
            for i, s in enumerate(self._subtasks)
        ]
        logger.info(
            f"AugmentedController reset — instruction='{self._instruction}', "
            f"subtasks=[{', '.join(subtask_summary)}]"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        self._hops_since_detection += 1

        if self._fsm is None or self._fsm.is_complete:
            return self._hover_action(state)

        if state.image is None:
            logger.warning("No camera image — hover")
            return self._hover_action(state)

        subtask = self._fsm.current_subtask
        current_yaw_rad = self._yaw_from_quaternion(state.orientation)

        if subtask.action in ("hover", "inspect"):
            return self._handle_stationary(state, subtask)

        if subtask.action == "circle":
            return self._handle_circle(state, subtask, current_yaw_rad)

        if subtask.nearby and not self._reference_reached:
            if not self._approaching_reference:
                self._approaching_reference = True
                logger.info(
                    f"Augmented hop {self._hop_count}: approaching reference "
                    f"'{subtask.nearby}' before searching for '{subtask.detect}'"
                )
            ref_query = _build_multi_query(subtask.nearby)
            detections = self._detector.detect(state.image, ref_query)
            best = self._pick_best(detections)

            if best is not None and best.score >= 0.25:
                self._enter_track(best)
                area = best.area_ratio(IMAGE_WIDTH, IMAGE_HEIGHT)
                if area > 0.02:
                    self._reference_reached = True
                    self._reset_nav_state(new_subtask=False)
                    logger.info(
                        f"Augmented hop {self._hop_count}: reference "
                        f"'{subtask.nearby}' reached (area={area:.3f}), "
                        f"now searching for '{subtask.detect}'"
                    )
            elif self._hops_since_detection < COAST_K:
                self._enter_coast()
            else:
                self._enter_search()
        else:
            primary_query = _build_multi_query(subtask.detect)
            detections = self._detector.detect(state.image, primary_query)
            best = self._pick_best(detections)

            if best is not None and best.score >= 0.25:
                self._enter_track(best)

                area = best.area_ratio(IMAGE_WIDTH, IMAGE_HEIGHT)
                self._try_advance_subtask(area, state.position[2])

                if subtask.action == "land" and area > 0.03:
                    if state.position[2] < -6.0:
                        approach_z = min(3.0, -5.0 - state.position[2])
                        approach_target_z = state.position[2] + approach_z
                        logger.debug(
                            f"Landing approach: descending while tracking "
                            f"z={state.position[2]:.1f} → {approach_target_z:.1f}"
                        )

                    if area > 0.30:
                        self._consecutive_land += 1
                        if self._consecutive_land >= 2:
                            logger.info(
                                f"Augmented hop {self._hop_count}: TOUCHDOWN on "
                                f"'{subtask.detect}' (area={area:.3f}, "
                                f"z={state.position[2]:.1f})"
                            )
                            return self._descend_action(state)
                    else:
                        self._consecutive_land = 0
                else:
                    self._consecutive_land = 0
            elif self._hops_since_detection < COAST_K:
                self._enter_coast()
            else:
                self._enter_search()

        self._log_hop(subtask)

        if self._nav_mode == "TRACK":
            return self._track_action(state, current_yaw_rad)
        elif self._nav_mode == "COAST":
            return self._coast_action(state, current_yaw_rad)
        else:
            return self._search_action(state, current_yaw_rad)

    def is_goal_reached(self, state: DroneState) -> bool:
        if self._hop_count >= self._leg_max_hops:
            logger.info(f"Max hops ({self._leg_max_hops}) reached")
            return True
        if self._fsm is not None and self._fsm.is_complete:
            logger.info(
                f"All {self._fsm.total_subtasks} subtasks complete "
                f"after {self._hop_count} hops"
            )
            return True
        if self._consecutive_land >= 3:
            if state.position[2] >= -3.5:
                logger.info(
                    f"Landing complete — z={state.position[2]:.1f} (on surface)"
                )
                return True
            else:
                logger.debug(
                    f"Landing signals but z={state.position[2]:.1f}, "
                    f"continuing descent to surface"
                )
        return False

    @property
    def subtask_fsm(self) -> Optional[SubtaskFSM]:
        return self._fsm

    @property
    def subtasks(self) -> List[Subtask]:
        return list(self._subtasks)

    def _try_advance_subtask(self, area: float, altitude_ned: float) -> None:
        """Check if the current subtask should advance."""
        if self._fsm is None or self._fsm.is_complete:
            return
        min_alt = self._constraints.get("min_altitude", 4.0)
        if self._fsm.check_completion(
            bbox_area_ratio=area,
            altitude_ned=altitude_ned,
            min_altitude=min_alt,
        ):
            prev = self._fsm.current_subtask
            next_sub = self._fsm.advance()
            logger.info(
                f"Augmented hop {self._hop_count}: subtask "
                f"'{prev.action}({prev.detect})' complete, "
                + (f"next='{next_sub.action}({next_sub.detect})'"
                   if next_sub else "ALL DONE")
            )
            self._reset_nav_state(new_subtask=True)

    def _reset_nav_state(self, new_subtask: bool = False) -> None:
        """Reset navigation state. Only resets reference flags on new subtask."""
        self._nav_mode = "SEARCH"
        self._memory_heading = 0.0
        self._memory_confidence = 0.0
        self._hops_since_detection = 999
        self._consecutive_land = 0
        self._search_hops = 0
        self._search_revolutions = 0
        if new_subtask:
            self._approaching_reference = False
            self._reference_reached = False

    def _check_nearby(
        self, image: np.ndarray, subtask: Subtask, primary_det: Detection,
    ) -> bool:
        """Verify the nearby object constraint using multi-object detection."""
        nearby_query = _build_multi_query(subtask.nearby)
        nearby_dets = self._detector.detect(image, nearby_query)
        if not nearby_dets:
            return False
        return check_spatial_proximity(
            [primary_det], nearby_dets,
            image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
        )

    def _handle_stationary(
        self, state: DroneState, subtask: Subtask,
    ) -> ControlAction:
        """Handle hover/inspect actions by staying in place."""
        min_alt = self._constraints.get("min_altitude", 4.0)
        if self._fsm.check_completion(altitude_ned=state.position[2], min_altitude=min_alt):
            prev = self._fsm.current_subtask
            next_sub = self._fsm.advance()
            logger.info(
                f"Augmented hop {self._hop_count}: stationary subtask "
                f"'{prev.action}' complete, "
                + (f"next='{next_sub.action}({next_sub.detect})'"
                   if next_sub else "ALL DONE")
            )
            self._reset_nav_state(new_subtask=True)
        return self._hover_action(state)

    def _handle_circle(
        self, state: DroneState, subtask: Subtask, current_yaw_rad: float,
    ) -> ControlAction:
        """Execute a circle action around the current position."""
        min_alt = self._constraints.get("min_altitude", 4.0)
        if self._fsm.check_completion(altitude_ned=state.position[2], min_altitude=min_alt):
            prev = self._fsm.current_subtask
            next_sub = self._fsm.advance()
            logger.info(
                f"Augmented hop {self._hop_count}: circle subtask complete, "
                + (f"next='{next_sub.action}({next_sub.detect})'"
                   if next_sub else "ALL DONE")
            )
            self._reset_nav_state(new_subtask=True)
            return self._hover_action(state)

        yaw_offset = SEARCH_YAW_STEP
        circle_yaw = current_yaw_rad + math.radians(yaw_offset)
        radius = 5.0
        target_x = state.position[0] + radius * math.cos(circle_yaw)
        target_y = state.position[1] + radius * math.sin(circle_yaw)
        return ControlAction(
            target_position=(target_x, target_y, state.position[2]),
            velocity=3.0,
        )

    def _pick_best(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return detections[0]

    def _enter_track(self, detection: Detection) -> None:
        heading_offset = bbox_to_heading_offset(
            detection.center_x, IMAGE_WIDTH, CAMERA_FOV_DEG,
        )
        self._memory_heading = (
            HEADING_EMA_ALPHA * heading_offset
            + (1.0 - HEADING_EMA_ALPHA) * self._memory_heading
        )
        self._memory_confidence = detection.score
        self._hops_since_detection = 0
        self._nav_mode = "TRACK"
        if self._search_hops > 0:
            self._search_hops = 0
            self._search_revolutions = 0

    def _enter_coast(self) -> None:
        self._memory_confidence *= CONFIDENCE_DECAY
        self._nav_mode = "COAST"

    def _enter_search(self) -> None:
        self._nav_mode = "SEARCH"

    def _track_action(
        self, state: DroneState, current_yaw_rad: float,
    ) -> ControlAction:
        heading_rad = current_yaw_rad + math.radians(self._memory_heading)
        best_area = self._memory_confidence
        distance_est = bbox_to_distance_estimate(best_area)
        step = min(self._hop_distance, max(2.0, distance_est * 0.3))
        target_x = state.position[0] + step * math.cos(heading_rad)
        target_y = state.position[1] + step * math.sin(heading_rad)

        subtask = self._fsm.current_subtask if self._fsm else None
        if subtask and subtask.action == "land" and state.position[2] < -6.0:
            target_z = min(state.position[2] + 3.0, -5.0)
        else:
            target_z = self._maybe_descend_to_cruise(state)

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._nav_speed,
        )

    def _coast_action(
        self, state: DroneState, current_yaw_rad: float,
    ) -> ControlAction:
        heading_rad = current_yaw_rad + math.radians(self._memory_heading)
        step = 3.0
        target_x = state.position[0] + step * math.cos(heading_rad)
        target_y = state.position[1] + step * math.sin(heading_rad)
        return ControlAction(
            target_position=(target_x, target_y, state.position[2]),
            velocity=self._nav_speed,
        )

    def _search_action(
        self, state: DroneState, current_yaw_rad: float,
    ) -> ControlAction:
        self._search_hops += 1
        hops_per_revolution = int(360.0 / SEARCH_YAW_STEP)

        if self._search_hops % hops_per_revolution == 0:
            self._search_revolutions += 1

        if self._search_revolutions >= 2:
            self._search_revolutions = 0
            self._search_hops = 0
            heading_rad = current_yaw_rad + math.radians(self._memory_heading)
            target_x = state.position[0] + SEARCH_ADVANCE_DISTANCE * math.cos(heading_rad)
            target_y = state.position[1] + SEARCH_ADVANCE_DISTANCE * math.sin(heading_rad)
            return ControlAction(
                target_position=(target_x, target_y, state.position[2]),
                velocity=self._nav_speed,
            )

        search_yaw_rad = current_yaw_rad + math.radians(SEARCH_YAW_STEP)
        creep = 2.0
        target_x = state.position[0] + creep * math.cos(search_yaw_rad)
        target_y = state.position[1] + creep * math.sin(search_yaw_rad)

        target_z = state.position[2]
        if self._search_hops % hops_per_revolution == 0 and self._search_revolutions < 2:
            new_z = state.position[2] - SEARCH_ALTITUDE_STEP
            target_z = max(new_z, SEARCH_ALTITUDE_CAP_NED)

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=2.0,
        )

    def _maybe_descend_to_cruise(self, state: DroneState) -> float:
        cruise_alt = -10.0
        if state.position[2] < cruise_alt - 1.0:
            return state.position[2] + 1.0
        return state.position[2]

    def _hover_action(self, state: DroneState) -> ControlAction:
        return ControlAction(
            target_position=state.position,
            velocity=1.0,
        )

    def _descend_action(self, state: DroneState) -> ControlAction:
        """Descend vertically — no horizontal movement during landing."""
        target_z = state.position[2] + 3.0
        if target_z > -1.5:
            target_z = -1.5
        logger.info(
            f"Augmented hop {self._hop_count}: VERTICAL DESCEND "
            f"z={state.position[2]:.1f} -> {target_z:.1f} "
            f"(holding x={state.position[0]:.1f}, y={state.position[1]:.1f})"
        )
        return ControlAction(
            target_position=(state.position[0], state.position[1], target_z),
            velocity=1.5,
        )

    @staticmethod
    def _yaw_from_quaternion(q: Tuple[float, float, float, float]) -> float:
        w, x, y, z = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _log_hop(self, subtask: Subtask) -> None:
        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            fsm_idx = self._fsm.current_index if self._fsm else 0
            fsm_total = self._fsm.total_subtasks if self._fsm else 0
            logger.info(
                f"Augmented hop {self._hop_count}: mode={self._nav_mode}, "
                f"heading={self._memory_heading:+.1f}deg, "
                f"subtask={fsm_idx+1}/{fsm_total} "
                f"'{subtask.action}({subtask.detect})', "
                f"hops_since_det={self._hops_since_detection}"
            )
