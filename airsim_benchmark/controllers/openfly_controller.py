"""
openfly_controller.py — OpenFly-Agent VLN Controller.

Uses OpenFly-Agent-7B (IPEC-COMMUNITY/openfly-agent-7b), a keyframe-aware
Vision-Language Navigation model specifically trained on 100k aerial drone
trajectories. Unlike general-purpose VLMs, this model directly outputs
flight action deltas from camera images and language instructions.

Architecture:
    Camera RGB + Instruction → OpenFly-Agent → Action Delta (dx, dy, dz, ...)
    → Scale to waypoint → AirSim moveToPosition → Repeat

The model is based on OpenVLA, fine-tuned for aerial VLN with an adaptive
frame-level token-sampling mechanism for handling rapid visual changes
during flight.

Reference: "OpenFly: A Comprehensive Platform for Aerial Vision-Language
Navigation" (Gao et al., 2025, arXiv:2502.18041)
"""

import logging
import math
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    _HAS_OPENFLY = True
except ImportError as e:
    _HAS_OPENFLY = False
    logger.warning(f"torch/transformers not available — OpenFly disabled ({e})")


class OpenFlyController(BaseController):
    """OpenFly-Agent VLN controller for drone navigation.

    The model takes camera images + language instructions and predicts
    flight action deltas, which are scaled into AirSim waypoints.

    Args:
        model_path: HuggingFace model ID or local path.
        arrival_tolerance: Distance threshold for goal check (m).
        nav_speed: Cruise speed (m/s).
        waypoint_scale: Multiplier for action deltas → waypoint distance (m).
        max_hops: Maximum navigation iterations per task.
        goal_bias: Fraction of goal direction blended into model output.
        device: CUDA device for inference.
    """

    def __init__(
        self,
        model_path: str = "IPEC-COMMUNITY/openfly-agent-7b",
        arrival_tolerance: float = 1.5,
        nav_speed: float = 5.0,
        waypoint_scale: float = 15.0,
        max_hops: int = 50,
        goal_bias: float = 0.2,
        device: str = "auto",
    ):
        if not _HAS_OPENFLY:
            raise RuntimeError(
                "PyTorch and transformers>=4.47.0 are required for OpenFly. "
                "Install with: pip install torch transformers timm accelerate"
            )

        self._arrival_tolerance = arrival_tolerance
        self._nav_speed = nav_speed
        self._waypoint_scale = waypoint_scale
        self._max_hops = max_hops
        self._goal_bias = goal_bias

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")
        self._device = device

        logger.info(f"Loading OpenFly-Agent from '{model_path}' on {device}...")
        t0 = time.time()

        # OpenFly's HF repo lacks custom architecture .py files. Fix by:
        # 1. Download OpenFly weights to local cache
        # 2. Symlink the architecture code from base openvla-7b into that dir
        # 3. Load from the combined local directory
        import os, shutil
        from huggingface_hub import snapshot_download

        base_model_dir = os.path.expanduser("~/models/openvla-7b")
        if not os.path.isfile(os.path.join(base_model_dir, "modeling_prismatic.py")):
            raise RuntimeError(
                f"Base model architecture files not found at {base_model_dir}. "
                f"Need ~/models/openvla-7b with modeling_prismatic.py"
            )

        # Download OpenFly weights (if not already cached)
        if os.path.isdir(model_path):
            openfly_dir = model_path
        else:
            openfly_dir = snapshot_download(model_path)
            logger.info(f"OpenFly weights cached at: {openfly_dir}")

        # Copy architecture .py files from base into OpenFly directory
        for py_file in ["configuration_prismatic.py", "modeling_prismatic.py",
                        "processing_prismatic.py"]:
            src = os.path.join(base_model_dir, py_file)
            dst = os.path.join(openfly_dir, py_file)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)
                logger.info(f"Copied {py_file} → OpenFly dir")

        # OpenFly's config.json and preprocessor_config.json lack the auto_map
        # that tells transformers which custom classes to load. Inject from base.
        import json as _json

        _config_path = os.path.join(openfly_dir, "config.json")
        with open(_config_path, "r") as _f:
            _cfg = _json.load(_f)
        if "auto_map" not in _cfg:
            _cfg["auto_map"] = {
                "AutoConfig": "configuration_prismatic.OpenVLAConfig",
                "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
            }
            with open(_config_path, "w") as _f:
                _json.dump(_cfg, _f, indent=2)
            logger.info("Injected auto_map into config.json")

        _preproc_path = os.path.join(openfly_dir, "preprocessor_config.json")
        with open(_preproc_path, "r") as _f:
            _pcfg = _json.load(_f)
        if "auto_map" not in _pcfg:
            _pcfg["auto_map"] = {
                "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
                "AutoProcessor": "processing_prismatic.PrismaticProcessor",
            }
            with open(_preproc_path, "w") as _f:
                _json.dump(_pcfg, _f, indent=2)
            logger.info("Injected auto_map into preprocessor_config.json")

        # Now load from the combined local directory
        self._processor = AutoProcessor.from_pretrained(
            openfly_dir, trust_remote_code=True
        )

        self._model = AutoModelForVision2Seq.from_pretrained(
            openfly_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(device)

        load_time = time.time() - t0
        logger.info(f"OpenFly-Agent loaded in {load_time:.1f}s")

        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._instruction: str = ""
        self._constraints: dict = {}
        self._task_id: int = 0
        self._hop_count: int = 0
        self._model_guided_hops: int = 0
        self._inference_count: int = 0
        self._recent_actions: deque = deque(maxlen=5)
        self._keyframes: list = []

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._model_guided_hops = 0
        self._recent_actions.clear()
        self._keyframes = []

        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)

        logger.info(
            f"OpenFlyController reset — task {self._task_id}, "
            f"instruction='{self._instruction}', goal={self._goal}, "
            f"scale={self._waypoint_scale}, goal_bias={self._goal_bias}"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        goal_dist = self._distance_to_goal(state)

        if state.image is None:
            logger.warning("No camera image — classical fallback")
            return self._fallback_action(state)

        try:
            direction = self._predict_direction(state.image, self._instruction)
        except Exception as e:
            logger.warning(f"OpenFly inference failed ({e}) — classical fallback")
            return self._fallback_action(state)

        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return self._fallback_action(state)

        self._model_guided_hops += 1
        self._recent_actions.append(direction.copy())

        # Normalize model direction
        model_unit = direction / norm

        # Compute goal direction
        goal_vec = np.array([
            self._goal[0] - state.position[0],
            self._goal[1] - state.position[1],
            self._goal[2] - state.position[2],
        ])
        goal_norm = np.linalg.norm(goal_vec)
        goal_unit = goal_vec / goal_norm if goal_norm > 1e-3 else np.zeros(3)

        # Blend model direction with goal bias
        blended = (1.0 - self._goal_bias) * model_unit + self._goal_bias * goal_unit
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-6:
            blended = blended / blended_norm

        # Scale hop distance (shorter near goal to prevent overshoot)
        effective_scale = min(self._waypoint_scale, max(2.0, goal_dist * 0.5))
        offset = blended * effective_scale

        target_x = state.position[0] + offset[0]
        target_y = state.position[1] + offset[1]
        target_z = state.position[2] + offset[2]
        target_z = self._clamp_altitude(target_z)

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"OpenFly hop {self._hop_count}: delta=({direction[0]:.3f}, "
                f"{direction[1]:.3f}, {direction[2]:.3f}), "
                f"norm={norm:.4f}, goal_dist={goal_dist:.1f}m, "
                f"scale={effective_scale:.1f}, "
                f"waypoint=({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
            )

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        if self._hop_count >= self._max_hops:
            dist = self._distance_to_goal(state)
            logger.info(
                f"Max hops ({self._max_hops}) — dist={dist:.1f}m, "
                f"model guided {self._model_guided_hops}/{self._hop_count}"
            )
            return True

        dist = self._distance_to_goal(state)
        if dist < self._arrival_tolerance * 2.5:
            logger.info(
                f"Goal reached at {dist:.1f}m "
                f"(model guided {self._model_guided_hops}/{self._hop_count} hops)"
            )
            return True

        return False

    def _predict_direction(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Run OpenFly-Agent inference and return 3D direction delta."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)

        inputs = self._processor(text=instruction, images=[pil_img], return_tensors="pt")
        inputs = inputs.to(self._device, dtype=torch.bfloat16)

        t0 = time.time()
        with torch.inference_mode():
            try:
                action = self._model.predict_action(
                    **inputs,
                    unnorm_key="vln_norm",
                    do_sample=False,
                )
            except ValueError:
                action = self._model.predict_action(
                    **inputs,
                    unnorm_key="bridge_orig",
                    do_sample=False,
                )
        elapsed = time.time() - t0
        self._inference_count += 1

        action_np = np.array(action, dtype=np.float64)

        if self._inference_count <= 3 or self._inference_count % 10 == 0:
            logger.debug(
                f"OpenFly inference #{self._inference_count} ({elapsed:.2f}s): "
                f"raw=({action_np[0]:.4f}, {action_np[1]:.4f}, {action_np[2]:.4f})"
            )

        # Extract position deltas (first 3 components)
        return action_np[:3]

    def _fallback_action(self, state: DroneState) -> ControlAction:
        """Navigate toward goal coordinates when model can't produce output."""
        gx, gy, gz = self._goal
        gz = self._clamp_altitude(gz)
        return ControlAction(
            target_position=(gx, gy, gz),
            velocity=self._effective_speed,
        )

    def _distance_to_goal(self, state: DroneState) -> float:
        dx = state.position[0] - self._goal[0]
        dy = state.position[1] - self._goal[1]
        dz = state.position[2] - self._goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _clamp_altitude(self, z_ned: float) -> float:
        min_alt = self._constraints.get("min_altitude", 2.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        margin = 1.0
        z_min = -(max_alt - margin)
        z_max = -(min_alt + margin)
        return max(z_min, min(z_max, z_ned))
