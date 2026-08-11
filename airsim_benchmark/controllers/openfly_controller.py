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
import warnings
from collections import deque
from typing import Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*use_fast.*is unset.*")
warnings.filterwarnings("ignore", message=".*AutoModelForVision2Seq.*is deprecated.*")

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

        # Download OpenFly weights (if not already cached)
        if os.path.isdir(model_path):
            openfly_dir = model_path
        else:
            openfly_dir = snapshot_download(model_path)
            logger.info(f"OpenFly weights cached at: {openfly_dir}")

        # Copy architecture .py files from OpenFly-Platform into cache dir
        # (these are different from base OpenVLA — 3-image vision backbone)
        openfly_platform = "/tmp/openfly-platform/train/extern/hf"
        for py_file in ["configuration_prismatic.py", "modeling_prismatic.py",
                        "processing_prismatic.py"]:
            src = os.path.join(openfly_platform, py_file)
            dst = os.path.join(openfly_dir, py_file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # OpenFly's config.json and preprocessor_config.json lack the auto_map
        # that tells transformers which custom classes to load. Inject from base.
        import json as _json

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

        # Now load from the combined local directory
        self._processor = AutoProcessor.from_pretrained(
            openfly_dir, trust_remote_code=True
        )

        self._model = AutoModelForVision2Seq.from_pretrained(
            openfly_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        ).to(device)

        load_time = time.time() - t0
        logger.info(f"OpenFly-Agent loaded in {load_time:.1f}s")

        # Verify vision backbone weights are actually loaded (not zeros/meta)
        vit_params = [p for n, p in self._model.named_parameters()
                      if "vision" in n.lower() or "visual" in n.lower()
                      or "patch_embed" in n.lower() or "blocks.0" in n.lower()]
        if vit_params:
            sample = vit_params[0]
            if sample.is_meta:
                logger.error("[DIAG] Vision backbone weights are META (uninitialized)!")
            elif sample.abs().sum().item() == 0:
                logger.error("[DIAG] Vision backbone weights are ALL ZEROS!")
            else:
                logger.info(f"[DIAG] Vision backbone OK: sample param "
                            f"mean={sample.mean().item():.6f}, "
                            f"std={sample.std().item():.6f}")
        else:
            logger.warning("[DIAG] Could not find vision backbone params to verify")

        # Wrap generate() to log raw token IDs for diagnostics
        _orig_generate = self._model.generate
        _diag_log = logger
        def _instrumented_generate(*args, **kwargs):
            result = _orig_generate(*args, **kwargs)
            n_action = 8
            token_ids = result[0, -n_action:].cpu().tolist()
            _diag_log.debug(f"[DIAG] raw action token IDs: {token_ids}")
            return result
        self._model.generate = _instrumented_generate

        # Monkey-patch prepare_inputs_for_generation (log only first call)
        _orig_prepare = self._model.prepare_inputs_for_generation
        _prepare_logged = [False]
        def _diag_prepare(input_ids, pixel_values=None, **kwargs):
            if not _prepare_logged[0] and pixel_values is not None:
                _diag_log.info(f"[DIAG] prepare_inputs: pixel_values confirmed shape={tuple(pixel_values.shape)}")
                _prepare_logged[0] = True
            return _orig_prepare(input_ids, pixel_values=pixel_values, **kwargs)
        self._model.prepare_inputs_for_generation = _diag_prepare

        self._unnorm_key: str = "vln_norm"
        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._instruction: str = ""
        self._constraints: dict = {}
        self._task_id: int = 0
        self._hop_count: int = 0
        self._model_guided_hops: int = 0
        self._inference_count: int = 0
        self._recent_actions: deque = deque(maxlen=5)
        self._recent_dists: deque = deque(maxlen=10)
        self._keyframes: list = []

        logger.info(
            "OpenFly action space (unnorm_key=%s): %d dims, "
            "q01=%s, q99=%s",
            self._unnorm_key,
            self._model.get_action_dim(self._unnorm_key),
            self._model.get_action_stats(self._unnorm_key)["q01"],
            self._model.get_action_stats(self._unnorm_key)["q99"],
        )

        # --- Debug: decode token 31999 to see what it actually represents ---
        tok = self._processor.tokenizer
        logger.info(f"[DIAG-TOKEN] tokenizer.decode(31999) = {repr(tok.decode([31999]))}")
        logger.info(f"[DIAG-TOKEN] tokenizer.decode(31998) = {repr(tok.decode([31998]))}")
        logger.info(f"[DIAG-TOKEN] tokenizer.decode(31997) = {repr(tok.decode([31997]))}")
        logger.info(f"[DIAG-TOKEN] tokenizer.decode(32000) = {repr(tok.decode([32000]))}")
        # Show a few tokens around the boundary
        for tid in [31743, 31744, 31745, 31999, 32000, 32063]:
            try:
                logger.info(f"[DIAG-TOKEN] id={tid} → {repr(tok.decode([tid]))}")
            except Exception as e:
                logger.info(f"[DIAG-TOKEN] id={tid} → ERROR: {e}")

        # --- Debug: verify vocabulary sizes ---
        tokenizer_len = len(tok)
        embed_weight = self._model.get_output_embeddings().weight
        logger.info(
            f"[DIAG-VOCAB] len(tokenizer)={tokenizer_len}, "
            f"model.config.text_config.vocab_size={self._model.config.text_config.vocab_size}, "
            f"model.vocab_size={self._model.vocab_size}, "
            f"lm_head embedding rows={embed_weight.shape[0]}, "
            f"pad_to_multiple_of={getattr(self._model.config, 'pad_to_multiple_of', 'N/A')}"
        )

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._model_guided_hops = 0
        self._recent_actions.clear()
        self._recent_dists.clear()
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

        # Adaptive goal bias: increase when diverging or close to goal
        adaptive_bias = self._goal_bias
        if self._hop_count >= 3 and len(self._recent_dists) >= 3:
            if self._recent_dists[-1] > self._recent_dists[-3]:
                adaptive_bias = min(0.95, self._goal_bias + 0.4)
        if goal_dist < 20.0:
            adaptive_bias = max(adaptive_bias, 0.8)
        elif goal_dist < 40.0:
            adaptive_bias = max(adaptive_bias, 0.6)

        # Blend model direction with goal bias
        blended = (1.0 - adaptive_bias) * model_unit + adaptive_bias * goal_unit
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-6:
            blended = blended / blended_norm

        # Scale hop distance (shorter near goal to prevent overshoot)
        effective_scale = min(self._waypoint_scale, max(2.0, goal_dist * 0.4))
        offset = blended * effective_scale

        target_x = state.position[0] + offset[0]
        target_y = state.position[1] + offset[1]
        target_z = state.position[2] + offset[2]
        target_z = self._clamp_altitude(target_z)

        # Track distance history for divergence detection
        self._recent_dists.append(goal_dist)

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"OpenFly hop {self._hop_count}: delta=({direction[0]:.3f}, "
                f"{direction[1]:.3f}, {direction[2]:.3f}), "
                f"norm={norm:.4f}, goal_dist={goal_dist:.1f}m, "
                f"bias={adaptive_bias:.2f}, scale={effective_scale:.1f}, "
                f"waypoint=({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
            )

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        dist = self._distance_to_goal(state)

        if dist < self._arrival_tolerance * 2.0:
            logger.info(
                f"Goal reached at {dist:.1f}m "
                f"(model guided {self._model_guided_hops}/{self._hop_count} hops)"
            )
            return True

        if self._hop_count >= self._max_hops:
            if self._hop_count == self._max_hops:
                logger.info(
                    f"Max hops ({self._max_hops}) exhausted — dist={dist:.1f}m, "
                    f"model guided {self._model_guided_hops}/{self._hop_count} "
                    f"[NOT reached, will timeout]"
                )
            return False

        return False

    def _predict_direction(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Run OpenFly-Agent inference and return 3D direction delta.

        OpenFly's vision backbone expects 3 images:
          [keyframe_1, keyframe_2, current_frame]
        Keyframes capture historical observations; current is the live view.
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image).convert("RGB")

        # Maintain keyframe buffer (store every 3rd hop)
        if self._hop_count % 3 == 1 or len(self._keyframes) == 0:
            self._keyframes.append(pil_img.copy())

        # Build 3-image list: [keyframe_1, keyframe_2, current]
        if len(self._keyframes) >= 2:
            image_list = [self._keyframes[-2], self._keyframes[-1], pil_img]
        elif len(self._keyframes) == 1:
            image_list = [self._keyframes[0], pil_img, pil_img]
        else:
            image_list = [pil_img, pil_img, pil_img]

        inputs = self._processor(instruction, image_list).to(
            self._device, dtype=torch.bfloat16
        )
        inputs.pop("attention_mask", None)

        # Diagnostic: confirm pixel_values is present and has expected shape
        if self._inference_count == 0:
            logger.info(f"[DIAG] inputs keys: {list(inputs.keys())}")
            pv = inputs.get("pixel_values")
            if pv is not None:
                logger.info(f"[DIAG] pixel_values: shape={tuple(pv.shape)}, dtype={pv.dtype}")
                pv_f = pv.float()
                logger.info(
                    f"[DIAG-IMG] pixel_values stats: "
                    f"min={pv_f.min().item():.4f}, max={pv_f.max().item():.4f}, "
                    f"mean={pv_f.mean().item():.4f}, std={pv_f.std().item():.4f}"
                )
                logger.info(
                    f"[DIAG-IMG] per-image means: "
                    f"img0={pv_f[0].mean().item():.4f}, "
                    f"img1={pv_f[1].mean().item():.4f}, "
                    f"img2={pv_f[2].mean().item():.4f}"
                )
            else:
                logger.error("[DIAG] pixel_values is MISSING from processor output!")
            logger.info(f"[DIAG] input_ids shape: {tuple(inputs['input_ids'].shape)}")

            # Verify the raw PIL image is not blank
            img_arr = np.array(pil_img)
            logger.info(
                f"[DIAG-IMG] raw PIL image: shape={img_arr.shape}, "
                f"dtype={img_arr.dtype}, min={img_arr.min()}, max={img_arr.max()}, "
                f"mean={img_arr.mean():.1f}"
            )

        t0 = time.time()
        with torch.inference_mode():
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            # Append action-trigger token (29871) if not present
            if not torch.all(input_ids[:, -1] == 29871):
                trigger = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, trigger], dim=1)

            n_action = self._model.get_action_dim(self._unnorm_key)
            generated_ids = self._model.generate(
                input_ids,
                pixel_values=pixel_values,
                max_new_tokens=n_action,
                do_sample=False,
            )

            # Decode actions — use config vocab_size (32064) not model.vocab_size
            # which may incorrectly report 32000 (base LLaMA without action tokens)
            predicted_token_ids = generated_ids[0, -n_action:].cpu().numpy()
            config_vocab = getattr(self._model.config, "text_config", self._model.config)
            vocab_size = getattr(config_vocab, "vocab_size", self._model.vocab_size)

            # Diagnostic: log the vocab size and full generation for first few inferences
            if self._inference_count < 3:
                logger.info(
                    f"[DIAG] vocab_size={vocab_size}, model.vocab_size={self._model.vocab_size}, "
                    f"n_action_bins={self._model.bin_centers.shape[0]}, n_action={n_action}"
                )
                full_gen = generated_ids[0].cpu().tolist()
                logger.info(f"[DIAG] full generated_ids (last 20): {full_gen[-20:]}")
                logger.info(f"[DIAG] input_ids tokens: {input_ids[0].cpu().tolist()}")

            discretized = np.clip(vocab_size - predicted_token_ids - 1, 0, self._model.bin_centers.shape[0] - 1)
            normalized = self._model.bin_centers[discretized]

            if self._inference_count < 3:
                logger.info(f"[DIAG] discretized bins: {discretized.tolist()}")
                logger.info(f"[DIAG] normalized values: {normalized.tolist()}")

            stats = self._model.get_action_stats(self._unnorm_key)
            mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
            q99, q01 = np.array(stats["q99"]), np.array(stats["q01"])
            action = np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)

        elapsed = time.time() - t0
        self._inference_count += 1

        action_np = np.array(action, dtype=np.float64)

        if self._inference_count <= 5 or self._inference_count % 10 == 0:
            all_dims = ", ".join(f"{v:.4f}" for v in action_np)
            logger.info(
                f"[DIAG] inference #{self._inference_count} ({elapsed:.2f}s): "
                f"all_dims=[{all_dims}]"
            )

        # vln_norm dims: [stop_flag, dist, heading1, heading2, pitch, 0, 0, 0]
        return action_np[1:4]

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
        min_alt = self._constraints.get("min_altitude", 8.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        margin = 1.0
        z_min = -(max_alt - margin)
        z_max = -(min_alt + margin)
        return max(z_min, min(z_max, z_ned))
