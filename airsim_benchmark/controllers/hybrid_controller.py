"""
hybrid_controller.py — Detection-Corrected VLN Hybrid Controller.

Combines two neural models for instruction-grounded aerial navigation:
  - GroundingDINO: visual grounding (WHERE is the target in the image?)
  - OpenFly-Agent-7B: flight dynamics (HOW to fly there smoothly?)

Architecture per hop:
  Camera Image + Instruction
      │                │
      ▼                ▼
  GroundingDINO    OpenFly-7B
      │                │
      ▼                ▼
  heading_offset    8D action (stop, fwd, yawL, yawR, up, dn, ...)
      │                │
      └──── BLEND ─────┘
               │
               ▼
      Corrected Action → AirSim waypoint

The blending strategy:
  - YAW: override OpenFly's yaw with GroundingDINO's heading offset
  - FORWARD: keep OpenFly's forward speed (learned dynamics)
  - ALTITUDE: keep OpenFly's altitude control (learned dynamics)
  - STOP: override with detection proximity (bbox area threshold)

When detection is unavailable (COAST/SEARCH), falls back to OpenFly's
base output for exploration behavior.
"""

import logging
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState
from ..core.detection_inference import (
    Detection,
    ObjectDetector,
    bbox_to_heading_offset,
    bbox_to_distance_estimate,
)
from ..core.instruction_parser import Subtask, parse_instruction
from ..core.subtask_fsm import SubtaskFSM
from ..core.target_phrase import _build_multi_query

logger = logging.getLogger(__name__)

COAST_K = 3
HEADING_EMA_ALPHA = 0.7
CAMERA_FOV_DEG = 90.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class HybridController(BaseController):
    """Detection-corrected VLN controller using GroundingDINO + OpenFly-7B.

    GroundingDINO provides WHERE to go (heading correction).
    OpenFly-7B provides HOW to fly (speed, altitude, smoothness).

    Args:
        detector: Pre-constructed ObjectDetector (for testing).
        detection_model_id: GroundingDINO model ID if detector is None.
        openfly_model_path: OpenFly-7B model path or HuggingFace ID.
        nav_speed: Cruise speed (m/s).
        hop_distance: Max distance per hop (m).
        max_hops: Default max hops per mission.
        detection_weight: How much DINO heading overrides OpenFly yaw (0-1).
        device: CUDA device string.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        detection_model_id: str = "IDEA-Research/grounding-dino-base",
        openfly_model_path: str = "IPEC-COMMUNITY/openfly-agent-7b",
        parser_model=None,
        nav_speed: float = 5.0,
        hop_distance: float = 5.0,
        max_hops: int = 50,
        detection_weight: float = 0.8,
        device: str = "auto",
    ):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for HybridController")

        self._nav_speed = nav_speed
        self._hop_distance = hop_distance
        self._max_hops = max_hops
        self._parser_model = parser_model
        self._detection_weight = detection_weight

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        if detector is not None:
            self._detector = detector
        else:
            self._detector = ObjectDetector(
                model_id=detection_model_id, device=device,
            )

        self._openfly = self._load_openfly(openfly_model_path, device)

        self._instruction: str = ""
        self._constraints: dict = {}
        self._subtasks: List[Subtask] = []
        self._fsm: Optional[SubtaskFSM] = None
        self._hop_count: int = 0
        self._leg_max_hops: int = max_hops

        self._nav_mode: str = "SEARCH"
        self._memory_heading: float = 0.0
        self._hops_since_detection: int = 999
        self._consecutive_land: int = 0

        self._approaching_reference: bool = False
        self._reference_reached: bool = False
        self._last_land_area: float = 0.0

        self._keyframes: list = []
        self._last_yaw_deg: float = 0.0

    def _load_openfly(self, model_path: str, device: str):
        """Load OpenFly-7B for flight dynamics inference."""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from huggingface_hub import snapshot_download
        import os, shutil, json as _json

        if os.path.isdir(model_path):
            openfly_dir = model_path
        else:
            openfly_dir = snapshot_download(model_path)

        openfly_platform = "/tmp/openfly-platform/train/extern/hf"
        for py_file in ["configuration_prismatic.py", "modeling_prismatic.py",
                        "processing_prismatic.py"]:
            src = os.path.join(openfly_platform, py_file)
            dst = os.path.join(openfly_dir, py_file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        _config_path = os.path.join(openfly_dir, "config.json")
        with open(_config_path, "r") as _f:
            _cfg = _json.load(_f)
        _cfg["auto_map"] = {
            "AutoConfig": "configuration_prismatic.OpenFlyConfig",
            "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
        }
        with open(_config_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)

        _preproc_path = os.path.join(openfly_dir, "preprocessor_config.json")
        with open(_preproc_path, "r") as _f:
            _pcfg = _json.load(_f)
        _pcfg["auto_map"] = {
            "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
            "AutoProcessor": "processing_prismatic.PrismaticProcessor",
        }
        with open(_preproc_path, "w") as _f:
            _json.dump(_pcfg, _f, indent=2)

        processor = AutoProcessor.from_pretrained(
            openfly_dir, trust_remote_code=True,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            openfly_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        ).to(device)

        logger.info(f"OpenFly-7B loaded on {device}")
        return {"model": model, "processor": processor, "dir": openfly_dir}

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
        self._hops_since_detection = 999
        self._consecutive_land = 0
        self._approaching_reference = False
        self._reference_reached = False
        self._keyframes = []
        self._last_yaw_deg = 0.0

        subtask_summary = [
            f"{i}: {s.action}({s.detect}" + (f", near={s.nearby}" if s.nearby else "") + ")"
            for i, s in enumerate(self._subtasks)
        ]
        logger.info(
            f"HybridController reset — instruction='{self._instruction}', "
            f"subtasks=[{', '.join(subtask_summary)}]"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        self._hops_since_detection += 1

        from airsim_benchmark.controllers.oracle_controller import quat_to_yaw_deg
        self._last_yaw_deg = quat_to_yaw_deg(*state.orientation)
        current_yaw_rad = math.radians(self._last_yaw_deg)

        if self._fsm is None or self._fsm.is_complete:
            return self._hover_action(state)

        if state.image is None:
            return self._hover_action(state)

        subtask = self._fsm.current_subtask

        if subtask.action in ("hover", "inspect"):
            min_alt = self._constraints.get("min_altitude", 4.0)
            if self._fsm.check_completion(
                altitude_ned=state.position[2], min_altitude=min_alt,
            ):
                self._fsm.advance()
                self._reset_nav_state(new_subtask=True)
            return self._hover_action(state)

        openfly_delta = self._openfly_inference(state.image, self._instruction)

        if subtask.nearby and not self._reference_reached:
            detect_query = _build_multi_query(subtask.nearby)
        else:
            detect_query = _build_multi_query(subtask.detect)

        detections = self._detector.detect(state.image, detect_query)
        best = self._pick_best(detections)
        dino_heading = None

        if best is not None and best.score >= 0.25:
            dino_heading = bbox_to_heading_offset(
                best.center_x, IMAGE_WIDTH, CAMERA_FOV_DEG,
            )
            self._memory_heading = (
                HEADING_EMA_ALPHA * dino_heading
                + (1.0 - HEADING_EMA_ALPHA) * self._memory_heading
            )
            self._hops_since_detection = 0
            self._nav_mode = "TRACK"

            area = best.area_ratio(IMAGE_WIDTH, IMAGE_HEIGHT)
            if subtask.action == "land":
                self._last_land_area = area

            if subtask.nearby and not self._reference_reached:
                if not self._approaching_reference:
                    self._approaching_reference = True
                    logger.info(
                        f"Hybrid hop {self._hop_count}: approaching reference "
                        f"'{subtask.nearby}'"
                    )
                if area > 0.02:
                    self._reference_reached = True
                    self._reset_nav_state(new_subtask=False)
                    logger.info(
                        f"Hybrid hop {self._hop_count}: reference "
                        f"'{subtask.nearby}' reached, switching to '{subtask.detect}'"
                    )
            else:
                self._try_advance_subtask(area, state.position[2])

                if subtask.action == "land" and area > 0.03:
                    if area > 0.30:
                        self._consecutive_land += 1
                        if self._consecutive_land >= 2:
                            logger.info(
                                f"Hybrid hop {self._hop_count}: TOUCHDOWN on "
                                f"'{subtask.detect}' (area={area:.3f}, "
                                f"z={state.position[2]:.1f})"
                            )
                            return self._descend_action(state)
                    else:
                        self._consecutive_land = 0
                else:
                    self._consecutive_land = 0
        elif self._hops_since_detection < COAST_K:
            self._nav_mode = "COAST"
        else:
            self._nav_mode = "SEARCH"

        blended = self._blend_actions(
            openfly_delta, dino_heading, current_yaw_rad, state,
        )

        target_x = state.position[0] + blended[0]
        target_y = state.position[1] + blended[1]
        target_z = self._clamp_altitude(state.position[2] + blended[2])

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"Hybrid hop {self._hop_count}: mode={self._nav_mode}, "
                f"dino_heading={dino_heading if dino_heading is not None else 'N/A'}, "
                f"openfly_delta=({openfly_delta[0]:.2f},{openfly_delta[1]:.2f},{openfly_delta[2]:.2f}), "
                f"blended=({blended[0]:.2f},{blended[1]:.2f},{blended[2]:.2f}), "
                f"subtask='{subtask.action}({subtask.detect})'"
            )

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._nav_speed,
        )

    def _blend_actions(
        self,
        openfly_delta: np.ndarray,
        dino_heading_deg: Optional[float],
        current_yaw_rad: float,
        state: DroneState,
    ) -> np.ndarray:
        """Blend OpenFly's flight dynamics with DINO's heading correction.

        TRACK mode: DINO heading drives direction, OpenFly drives speed/altitude.
        COAST mode: Use last known DINO heading + OpenFly dynamics.
        SEARCH mode: Use pure OpenFly output (exploration).

        Landing approach: when tracking a land target, descend to z=-5
        while approaching so the drone arrives at rooftop altitude.
        """
        openfly_horiz = math.sqrt(openfly_delta[0]**2 + openfly_delta[1]**2)
        if openfly_horiz < 0.5:
            forward_step = self._hop_distance
            alt_delta = 0.0
        else:
            forward_step = max(2.0, min(self._hop_distance, openfly_horiz))
            alt_delta = openfly_delta[2]

        if self._nav_mode == "TRACK" and dino_heading_deg is not None:
            target_yaw = current_yaw_rad + math.radians(self._memory_heading)
        elif self._nav_mode == "COAST":
            target_yaw = current_yaw_rad + math.radians(self._memory_heading)
            forward_step = min(forward_step, 3.0)
        else:
            if openfly_horiz > 0.5:
                openfly_yaw = math.atan2(openfly_delta[1], openfly_delta[0])
                target_yaw = openfly_yaw
            else:
                target_yaw = current_yaw_rad + math.radians(30.0)
            forward_step = 2.0

        subtask = self._fsm.current_subtask if self._fsm else None
        if (subtask and subtask.action == "land"
                and self._nav_mode == "TRACK"
                and hasattr(self, '_last_land_area')):
            land_area = self._last_land_area
            if land_area > 0.05 and state.position[2] < -4.0:
                descent_rate = min(3.0, land_area * 15.0)
                alt_delta = descent_rate
                target_alt = state.position[2] + alt_delta
                if self._hop_count % 3 == 0:
                    logger.info(
                        f"Hybrid hop {self._hop_count}: landing approach "
                        f"descending {descent_rate:.1f}m (area={land_area:.2f}, "
                        f"z={state.position[2]:.1f} → {target_alt:.1f})"
                    )

        dx = forward_step * math.cos(target_yaw)
        dy = forward_step * math.sin(target_yaw)
        dz = np.clip(alt_delta, -2.0, 3.0)

        return np.array([dx, dy, dz])

    def _openfly_inference(
        self, image: np.ndarray, instruction: str,
    ) -> np.ndarray:
        """Run OpenFly-7B inference, return 3D NED delta."""
        from PIL import Image as PILImage
        from airsim_benchmark.core.action_space import (
            denormalize_action, regression_action_to_ned,
        )

        model = self._openfly["model"]
        processor = self._openfly["processor"]

        pil_img = PILImage.fromarray(image).convert("RGB")

        if self._hop_count % 3 == 1 or len(self._keyframes) == 0:
            self._keyframes.append(pil_img.copy())

        if len(self._keyframes) >= 2:
            image_list = [self._keyframes[-2], self._keyframes[-1], pil_img]
        elif len(self._keyframes) == 1:
            image_list = [self._keyframes[0], pil_img, pil_img]
        else:
            image_list = [pil_img, pil_img, pil_img]

        inputs = processor(instruction, image_list).to(
            self._device, dtype=torch.bfloat16,
        )
        inputs.pop("attention_mask", None)

        unnorm_key = "vln_norm"
        n_action = model.get_action_dim(unnorm_key)
        pad = getattr(model.config, "pad_to_multiple_of", 64)
        eff_vocab = model.config.text_config.vocab_size - pad
        n_bins = 16
        bin_centers = np.linspace(-1.0, 1.0, n_bins)

        with torch.inference_mode():
            input_ids = inputs["input_ids"]
            if not torch.all(input_ids[:, -1] == 29871):
                trigger = torch.tensor(
                    [[29871]], device=input_ids.device, dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, trigger], dim=1)

            generated = model.generate(
                input_ids, pixel_values=inputs["pixel_values"],
                max_new_tokens=n_action, do_sample=False,
            )
            predicted_ids = generated[0, -n_action:].cpu().numpy()

        discretized = np.clip(eff_vocab - predicted_ids - 1, 0, n_bins - 1)
        normalized = bin_centers[discretized]

        stats = model.get_action_stats(unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        q99, q01 = np.array(stats["q99"]), np.array(stats["q01"])
        action = np.where(
            mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized,
        )

        delta, hover = regression_action_to_ned(
            np.array(action, dtype=np.float64), self._last_yaw_deg,
        )
        return delta

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
                logger.info(f"Landing complete at z={state.position[2]:.1f} (on surface)")
                return True
            else:
                logger.debug(
                    f"Landing signals but z={state.position[2]:.1f}, "
                    f"continuing descent"
                )
        return False

    def _try_advance_subtask(self, area: float, altitude_ned: float) -> None:
        if self._fsm is None or self._fsm.is_complete:
            return
        min_alt = self._constraints.get("min_altitude", 4.0)
        if self._fsm.check_completion(
            bbox_area_ratio=area, altitude_ned=altitude_ned, min_altitude=min_alt,
        ):
            prev = self._fsm.current_subtask
            next_sub = self._fsm.advance()
            logger.info(
                f"Hybrid hop {self._hop_count}: subtask "
                f"'{prev.action}({prev.detect})' complete, "
                + (f"next='{next_sub.action}({next_sub.detect})'"
                   if next_sub else "ALL DONE")
            )
            self._reset_nav_state(new_subtask=True)

    def _reset_nav_state(self, new_subtask: bool = False) -> None:
        self._nav_mode = "SEARCH"
        self._memory_heading = 0.0
        self._hops_since_detection = 999
        self._consecutive_land = 0
        if new_subtask:
            self._approaching_reference = False
            self._reference_reached = False

    def _pick_best(self, detections: List[Detection]) -> Optional[Detection]:
        return detections[0] if detections else None

    def _hover_action(self, state: DroneState) -> ControlAction:
        return ControlAction(target_position=state.position, velocity=1.0)

    def _descend_action(self, state: DroneState) -> ControlAction:
        """Descend vertically — no horizontal movement during landing."""
        target_z = state.position[2] + 3.0
        if target_z > -1.5:
            target_z = -1.5
        logger.info(
            f"Hybrid hop {self._hop_count}: VERTICAL DESCEND "
            f"z={state.position[2]:.1f} -> {target_z:.1f} "
            f"(holding x={state.position[0]:.1f}, y={state.position[1]:.1f})"
        )
        return ControlAction(
            target_position=(state.position[0], state.position[1], target_z),
            velocity=1.5,
        )

    def _clamp_altitude(self, z_ned: float) -> float:
        min_alt = self._constraints.get("min_altitude", 3.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        return max(-(max_alt - 1.0), min(-(min_alt + 1.0), z_ned))
