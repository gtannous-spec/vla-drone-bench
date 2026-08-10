"""
detection_controller.py — GroundingDINO detection + classical steering.

GPS-free navigation using open-vocabulary object detection for target
localization, with a TRACK/COAST/SEARCH state machine for weather-robust
recovery when detection fails.

Multi-query strategy: when the primary detection query fails (e.g. "rooftop"
from aerial view), tries alternative queries (e.g. "building", "house").
If the instruction references a known landmark ("nearest rooftop to the
red car"), uses that landmark for initial navigation.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState
from ..core.detection_inference import (
    Detection, ObjectDetector, bbox_to_heading_offset, bbox_to_distance_estimate,
)
from ..core.target_phrase import extract_target, TargetInfo

logger = logging.getLogger(__name__)

COAST_K = 3
SEARCH_YAW_STEP = 30.0
SEARCH_ALTITUDE_STEP = 3.0
SEARCH_MAX_ALTITUDE_NED = -15.0
SEARCH_ALTITUDE_CAP_NED = -25.0
SEARCH_ADVANCE_DISTANCE = 10.0
HEADING_EMA_ALPHA = 0.7
CONFIDENCE_DECAY = 0.8
CAMERA_FOV_DEG = 90.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
LANDING_BBOX_AREA_TRIGGER = 0.03


class DetectionController(BaseController):
    """GPS-free navigation using GroundingDINO object detection.

    Args:
        detector: Pre-constructed ObjectDetector (for dependency injection in tests).
        model_id: HuggingFace model ID if detector is None.
        nav_speed: Cruise speed (m/s).
        hop_distance: Maximum distance per hop (m).
        max_hops: Default max navigation iterations per leg.
        device: CUDA device.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        model_id: str = "IDEA-Research/grounding-dino-base",
        nav_speed: float = 5.0,
        hop_distance: float = 5.0,
        max_hops: int = 50,
        device: str = "auto",
    ):
        self._nav_speed = nav_speed
        self._hop_distance = hop_distance
        self._max_hops = max_hops

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
        self._target: Optional[TargetInfo] = None
        self._hop_count: int = 0
        self._leg_max_hops: int = max_hops

        self._nav_mode: str = "SEARCH"
        self._memory_heading: float = 0.0
        self._memory_confidence: float = 0.0
        self._hops_since_detection: int = 999
        self._consecutive_land: int = 0

        self._search_hops: int = 0
        self._search_revolutions: int = 0
        self._search_start_altitude: float = -10.0

        self._active_query: str = ""
        self._using_reference: bool = False
        self._reference_reached: bool = False

    def reset(self, task_config: dict) -> None:
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._leg_max_hops = task_config.get("max_hops", self._max_hops)
        self._target = extract_target(self._instruction)

        self._hop_count = 0
        self._nav_mode = "SEARCH"
        self._memory_heading = 0.0
        self._memory_confidence = 0.0
        self._hops_since_detection = 999
        self._consecutive_land = 0

        self._search_hops = 0
        self._search_revolutions = 0
        self._search_start_altitude = -10.0

        self._active_query = self._target.multi_query or self._target.phrase
        self._using_reference = False
        self._reference_reached = False

        if (self._target.reference_phrase
                and self._target.reference_phrase != self._target.phrase):
            self._using_reference = True
            self._active_query = self._target.reference_phrase

        logger.info(
            f"DetectionController reset — instruction='{self._instruction}', "
            f"target='{self._target.phrase}', "
            f"multi_query='{self._active_query}', "
            f"wants_land={self._target.wants_land}, "
            f"reference={self._target.reference_phrase}"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        self._hops_since_detection += 1

        if state.image is None:
            logger.warning("No camera image — hover")
            return self._hover_action(state)

        detections = self._detector.detect(state.image, self._active_query)
        best = self._pick_best(detections)

        current_yaw_rad = self._yaw_from_quaternion(state.orientation)

        if best is not None and best.score >= 0.25:
            self._enter_track(best, current_yaw_rad)

            if self._using_reference and not self._reference_reached:
                area = best.area_ratio(IMAGE_WIDTH, IMAGE_HEIGHT)
                if area > 0.02:
                    self._reference_reached = True
                    self._switch_to_target_queries()
        elif self._hops_since_detection < COAST_K:
            self._enter_coast()
        else:
            self._enter_search()

        if self._target.wants_land and best is not None:
            area = best.area_ratio(IMAGE_WIDTH, IMAGE_HEIGHT)
            if area > LANDING_BBOX_AREA_TRIGGER:
                self._consecutive_land += 1
                if self._consecutive_land >= 2:
                    logger.info(
                        f"Detection hop {self._hop_count}: LANDING "
                        f"(area={area:.3f}, consecutive={self._consecutive_land})"
                    )
                    return self._descend_action(state)
            else:
                self._consecutive_land = 0

        if self._target.wants_land and self._reference_reached:
            return self._descend_action(state)

        self._log_hop(self._nav_mode, self._memory_heading, self._hop_distance)

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
        if self._consecutive_land >= 3:
            logger.info("Landing complete — 3 consecutive land signals")
            return True
        return False

    def _switch_to_target_queries(self) -> None:
        """After reaching reference landmark, switch to detecting the actual target."""
        self._using_reference = False
        self._active_query = self._target.multi_query or self._target.phrase
        self._hops_since_detection = 999
        logger.info(
            f"Detection hop {self._hop_count}: reference reached, "
            f"switching to target query '{self._active_query}'"
        )

    def _pick_best(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return detections[0]

    def _enter_track(self, detection: Detection, current_yaw_rad: float) -> None:
        heading_offset = bbox_to_heading_offset(
            detection.center_x, IMAGE_WIDTH, CAMERA_FOV_DEG
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

    def _track_action(self, state: DroneState, current_yaw_rad: float) -> ControlAction:
        heading_rad = current_yaw_rad + math.radians(self._memory_heading)

        best_area = self._memory_confidence
        distance_est = bbox_to_distance_estimate(best_area)
        step = min(self._hop_distance, max(2.0, distance_est * 0.3))

        target_x = state.position[0] + step * math.cos(heading_rad)
        target_y = state.position[1] + step * math.sin(heading_rad)

        target_z = self._maybe_descend_to_cruise(state)

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._nav_speed,
        )

    def _coast_action(self, state: DroneState, current_yaw_rad: float) -> ControlAction:
        heading_rad = current_yaw_rad + math.radians(self._memory_heading)
        step = 3.0

        target_x = state.position[0] + step * math.cos(heading_rad)
        target_y = state.position[1] + step * math.sin(heading_rad)

        return ControlAction(
            target_position=(target_x, target_y, state.position[2]),
            velocity=self._nav_speed,
        )

    def _search_action(self, state: DroneState, current_yaw_rad: float) -> ControlAction:
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

        yaw_offset = SEARCH_YAW_STEP
        search_yaw_rad = current_yaw_rad + math.radians(yaw_offset)
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
        target_z = state.position[2] + 2.5
        if target_z > -1.5:
            target_z = -1.5
        logger.info(
            f"Detection hop {self._hop_count}: DESCEND "
            f"z={state.position[2]:.1f} -> {target_z:.1f}"
        )
        return ControlAction(
            target_position=(state.position[0], state.position[1], target_z),
            velocity=2.0,
        )

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

    def _log_hop(self, mode: str, heading_offset: float, step: float) -> None:
        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"Detection hop {self._hop_count}: mode={mode}, "
                f"heading={heading_offset:+.1f}°, step={step:.1f}m, "
                f"query='{self._active_query}', "
                f"hops_since_det={self._hops_since_detection}"
            )
