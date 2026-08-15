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
import os
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


class ActionTokenLogitsProcessor:
    """Force generation to only produce tokens in the action range.

    Text tokens outside [act_lo, act_hi] are masked to -inf so the model
    can only output valid action-bin tokens (31744-31999 by default).
    """

    def __init__(self, act_lo: int, act_hi: int):
        self.act_lo = act_lo
        self.act_hi = act_hi

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.act_lo:self.act_hi + 1] = 0.0
        return scores + mask


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
        lora_path: str = "",
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

        # Fix A: extend tokenizer to match lm_head so generate() can
        # produce all token IDs (the base tokenizer has 32001 entries but
        # the lm_head has 32064 rows — the gap blocks action token sampling)
        _tok = self._processor.tokenizer
        _lm_head_size = self._model.get_output_embeddings().weight.shape[0]
        _n_missing = _lm_head_size - len(_tok)
        if _n_missing > 0:
            _prev_len = len(_tok)
            _new_tokens = [f"<action_{i}>" for i in range(_n_missing)]
            _tok.add_tokens(_new_tokens, special_tokens=True)
            logger.info(
                f"[FIX-A] Extended tokenizer: {_prev_len} -> {len(_tok)} "
                f"(lm_head={_lm_head_size})"
            )
        else:
            logger.info(
                f"[FIX-A] Tokenizer already covers lm_head "
                f"(len={len(_tok)}, lm_head={_lm_head_size})"
            )

        # LoRA adapter loading: merge adapter weights into the base model
        # so inference speed is unchanged (no adapter overhead at runtime).
        if lora_path and os.path.isdir(lora_path):
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, lora_path)
                self._model = self._model.merge_and_unload()
                logger.info(f"[LORA] Adapter merged from {lora_path}")
            except Exception as e:
                logger.warning(f"[LORA] Failed to load adapter ({e}), using base model")
        elif lora_path:
            logger.warning(f"[LORA] Path not found: {lora_path}, using base model")

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

        # Wrap generate() to log raw token IDs and store them for collapse
        # detection (Fix B uses _last_gen_tokens to trigger the fallback)
        _orig_generate = self._model.generate
        _diag_log = logger
        _self_ref = self
        self._last_gen_tokens = []  # type: list[int]
        def _instrumented_generate(*args, **kwargs):
            # Inject action-token constraint into every generate() call
            if hasattr(_self_ref, "_action_logit_processor"):
                existing = kwargs.get("logits_processor", [])
                kwargs["logits_processor"] = list(existing) + [
                    _self_ref._action_logit_processor
                ]
            result = _orig_generate(*args, **kwargs)
            n_action = 8
            token_ids = result[0, -n_action:].cpu().tolist()
            _self_ref._last_gen_tokens = token_ids
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

        # Action tokenization config.  n_bins must match the value used
        # during LoRA training (default 16).  We build our own bin_centers
        # rather than using the model's built-in 256-entry array.
        pad = getattr(self._model.config, "pad_to_multiple_of", 64)
        cfg_vocab = self._model.config.text_config.vocab_size
        self._eff_vocab = cfg_vocab - pad    # 32000
        self._n_bins = int(os.environ.get("N_BINS", "16"))
        self._custom_bin_centers = np.linspace(-1.0, 1.0, self._n_bins)
        self._act_lo = self._eff_vocab - self._n_bins
        self._act_hi = self._eff_vocab - 1   # 31999
        self._action_logit_processor = ActionTokenLogitsProcessor(
            self._act_lo, self._act_hi,
        )
        logger.info(
            f"Action-token constraint: {self._n_bins} bins, "
            f"tokens [{self._act_lo}, {self._act_hi}]"
        )

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
        self._task_id = task_config.get("id", 0)
        self._goal = tuple(task_config["goal"]) if "goal" in task_config else None
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._model_guided_hops = 0
        self._recent_actions.clear()
        self._recent_dists.clear()
        self._keyframes = []
        self._leg_max_hops = task_config.get("max_hops", self._max_hops)

        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)

        mode = "GPS-free" if self._goal is None else f"goal={self._goal}"
        logger.info(
            f"OpenFlyController reset — task {self._task_id}, "
            f"instruction='{self._instruction}', {mode}, "
            f"scale={self._waypoint_scale}, goal_bias={self._goal_bias}"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1

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
        model_unit = direction / norm

        # GPS-free mode: pure model output, no goal blending
        if self._goal is None:
            offset = model_unit * self._waypoint_scale
            target_x = state.position[0] + offset[0]
            target_y = state.position[1] + offset[1]
            target_z = self._clamp_altitude(state.position[2] + offset[2])

            if self._hop_count <= 3 or self._hop_count % 5 == 0:
                logger.info(
                    f"OpenFly hop {self._hop_count} [GPS-free]: "
                    f"delta=({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}), "
                    f"norm={norm:.4f}, scale={self._waypoint_scale:.1f}, "
                    f"pos=({state.position[0]:.1f}, {state.position[1]:.1f}, {state.position[2]:.1f}), "
                    f"waypoint=({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
                )
            return ControlAction(
                target_position=(target_x, target_y, target_z),
                velocity=self._effective_speed,
            )

        # Goal-directed mode: blend model direction with goal vector
        goal_dist = self._distance_to_goal(state)
        goal_vec = np.array([
            self._goal[0] - state.position[0],
            self._goal[1] - state.position[1],
            self._goal[2] - state.position[2],
        ])
        goal_norm = np.linalg.norm(goal_vec)
        goal_unit = goal_vec / goal_norm if goal_norm > 1e-3 else np.zeros(3)

        adaptive_bias = self._goal_bias
        if self._hop_count >= 3 and len(self._recent_dists) >= 3:
            if self._recent_dists[-1] > self._recent_dists[-3]:
                adaptive_bias = min(0.95, self._goal_bias + 0.4)
        if goal_dist < 20.0:
            adaptive_bias = max(adaptive_bias, 0.8)
        elif goal_dist < 40.0:
            adaptive_bias = max(adaptive_bias, 0.6)

        blended = (1.0 - adaptive_bias) * model_unit + adaptive_bias * goal_unit
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-6:
            blended = blended / blended_norm

        effective_scale = min(self._waypoint_scale, max(2.0, goal_dist * 0.4))
        offset = blended * effective_scale

        target_x = state.position[0] + offset[0]
        target_y = state.position[1] + offset[1]
        target_z = self._clamp_altitude(state.position[2] + offset[2])

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
        effective_max = self._leg_max_hops if hasattr(self, '_leg_max_hops') else self._max_hops

        # GPS-free mode: only hop budget determines leg completion
        if self._goal is None:
            if self._hop_count >= effective_max:
                logger.info(
                    f"Leg complete — {self._hop_count} hops, "
                    f"model guided {self._model_guided_hops}/{self._hop_count}"
                )
                return True
            return False

        # Goal-directed mode
        dist = self._distance_to_goal(state)
        if dist < self._arrival_tolerance * 3.0:
            logger.info(
                f"Goal reached at {dist:.1f}m "
                f"(model guided {self._model_guided_hops}/{self._hop_count} hops)"
            )
            return True

        if self._hop_count >= effective_max:
            logger.info(
                f"Max hops ({effective_max}) reached — dist={dist:.1f}m, "
                f"model guided {self._model_guided_hops}/{self._hop_count}"
            )
            return True

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

        # Fix D: on hops 2, 3, 5 log whether the 3 input images actually
        # differ (confirms keyframe buffer provides temporal diversity)
        if self._hop_count in (2, 3, 5):
            pv = inputs.get("pixel_values")
            if pv is not None:
                pv_f = pv.float()
                n_imgs = min(3, pv_f.shape[0])
                means = [pv_f[i].mean().item() for i in range(n_imgs)]
                all_same = all(abs(m - means[0]) < 1e-4 for m in means)
                logger.info(
                    f"[FIX-D] hop {self._hop_count} keyframe diversity: "
                    f"means=[{', '.join(f'{m:.4f}' for m in means)}], "
                    f"{'IDENTICAL' if all_same else 'DIVERSE'}"
                )

        t0 = time.time()
        with torch.inference_mode():
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            # Generate action tokens and decode with our custom n_bins.
            # We bypass predict_action() because it uses the model's built-in
            # 256-bin decoder which doesn't match our reduced bin count.
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

            predicted_token_ids = generated_ids[0, -n_action:].cpu().numpy()

            discretized = np.clip(
                self._eff_vocab - predicted_token_ids - 1,
                0, self._n_bins - 1,
            )
            normalized = self._custom_bin_centers[discretized]

            stats = self._model.get_action_stats(self._unnorm_key)
            mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
            q99, q01 = np.array(stats["q99"]), np.array(stats["q01"])
            action = np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)

            # Fix C: log raw token IDs from generate() to verify tokenizer fix
            gen_toks = self._last_gen_tokens
            if self._inference_count < 5 or self._inference_count % 10 == 0:
                unique_toks = set(gen_toks)
                logger.info(
                    f"[FIX-C] inference #{self._inference_count}: "
                    f"gen_tokens={gen_toks}, "
                    f"unique={len(unique_toks)}, "
                    f"collapsed={'YES' if len(unique_toks) <= 1 else 'NO'}"
                )

            # Fix B: if generation collapsed (all tokens identical), try
            # direct logit extraction as a fallback
            if gen_toks and len(set(gen_toks)) <= 1:
                logger.warning(
                    f"[FIX-B] Generation collapsed (all tokens={gen_toks[0]}), "
                    f"trying direct logit extraction"
                )
                try:
                    action = self._direct_logit_inference(
                        inputs["input_ids"], inputs["pixel_values"]
                    )
                except Exception as e:
                    logger.warning(f"[FIX-B] Direct logit fallback failed: {e}")

            # Logit analysis: on first inference, check action token distribution
            if self._inference_count == 0:
                try:
                    self._log_action_logits(inputs)
                except Exception as e:
                    logger.debug(f"Logit analysis failed: {e}")

            # Detailed per-dim diagnostics on first 3 inferences
            if self._inference_count < 3:
                try:
                    self._log_per_dim_logits(inputs)
                except Exception as e:
                    logger.debug(f"Per-dim logit analysis failed: {e}")

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
        if self._goal is None:
            # GPS-free: fly straight ahead (north) as a safe default
            return ControlAction(
                target_position=(
                    state.position[0] + self._waypoint_scale,
                    state.position[1],
                    self._clamp_altitude(state.position[2]),
                ),
                velocity=self._effective_speed,
            )
        gx, gy, gz = self._goal
        gz = self._clamp_altitude(gz)
        return ControlAction(
            target_position=(gx, gy, gz),
            velocity=self._effective_speed,
        )

    def _distance_to_goal(self, state: DroneState) -> float:
        if self._goal is None:
            return float("inf")
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

    def _direct_logit_inference(
        self,
        input_ids: "torch.Tensor",
        pixel_values: "torch.Tensor",
    ) -> np.ndarray:
        """Bypass generate() and extract action tokens from raw logits.

        Runs an autoregressive forward pass, picking the argmax token from
        the full logit vector at each step.  This sidesteps any vocabulary-
        size constraint that HuggingFace's generate() pipeline may impose.
        """
        if not torch.all(input_ids[:, -1] == 29871):
            trigger = torch.tensor(
                [[29871]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, trigger], dim=1)

        n_action = self._model.get_action_dim(self._unnorm_key)
        action_token_ids = []  # type: list[int]
        pv = pixel_values  # only needed on first step

        for step in range(n_action):
            outputs = self._model(input_ids=input_ids, pixel_values=pv)
            logits = outputs.logits[:, -1, :]  # [1, vocab_dim]

            # On the very first step of the first inference, dump top-5
            # to diagnose where the model's probability mass sits
            if step == 0 and self._inference_count == 0:
                top5 = torch.topk(logits[0], 5)
                logger.info(
                    f"[FIX-B] top-5 logit IDs: {top5.indices.tolist()}, "
                    f"values: {[f'{v:.3f}' for v in top5.values.tolist()]}"
                )
                # Also log the action-token-range max
                pad = getattr(self._model.config, "pad_to_multiple_of", 64)
                v_size = self._model.config.text_config.vocab_size - pad
                act_lo = v_size - self._model.bin_centers.shape[0]
                act_hi = v_size - 1
                act_logits = logits[0, act_lo:act_hi + 1]
                act_top5 = torch.topk(act_logits, min(5, act_logits.shape[0]))
                logger.info(
                    f"[FIX-B] action-range [{act_lo}-{act_hi}] top-5 bins: "
                    f"{(act_top5.indices + act_lo).tolist()}, "
                    f"values: {[f'{v:.3f}' for v in act_top5.values.tolist()]}"
                )

            token_id = logits.argmax(dim=-1)  # greedy
            action_token_ids.append(token_id.item())
            input_ids = torch.cat(
                [input_ids, token_id.unsqueeze(0)], dim=1
            )
            pv = None  # vision features cached in KV after first step

        logger.info(f"[FIX-B] direct logit token IDs: {action_token_ids}")

        # Decode tokens → continuous actions using our custom bin centers
        token_arr = np.array(action_token_ids)
        discretized = np.clip(
            self._eff_vocab - token_arr - 1, 0, self._n_bins - 1,
        )
        normalized = self._custom_bin_centers[discretized]

        stats = self._model.get_action_stats(self._unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        q99 = np.array(stats["q99"])
        q01 = np.array(stats["q01"])
        return np.where(
            mask,
            0.5 * (normalized + 1) * (q99 - q01) + q01,
            normalized,
        )

    def _log_action_logits(self, inputs: dict) -> None:
        """Log the action-range logit distribution for diagnostics.

        Runs a single forward pass (no generation) and reports where the
        model's probability mass sits relative to the expected action-token
        range (IDs vocab_size-255 through vocab_size-1).
        """
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        if not torch.all(input_ids[:, -1] == 29871):
            trigger = torch.tensor(
                [[29871]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, trigger], dim=1)

        outputs = self._model(input_ids=input_ids, pixel_values=pixel_values)
        logits = outputs.logits[:, -1, :]  # [1, vocab_dim]

        # Overall top-5
        top5 = torch.topk(logits[0], 5)
        logger.info(
            f"[LOGIT-DIAG] overall top-5 IDs: {top5.indices.tolist()}, "
            f"values: {[f'{v:.3f}' for v in top5.values.tolist()]}"
        )

        # Action-range top-5
        pad = getattr(self._model.config, "pad_to_multiple_of", 64)
        v_size = self._model.config.text_config.vocab_size - pad
        n_bins = self._model.bin_centers.shape[0]
        act_lo = v_size - n_bins
        act_hi = v_size - 1
        act_logits = logits[0, act_lo:act_hi + 1]
        act_top5 = torch.topk(act_logits, min(5, act_logits.shape[0]))
        logger.info(
            f"[LOGIT-DIAG] action-range [{act_lo}-{act_hi}] top-5: "
            f"IDs={(act_top5.indices + act_lo).tolist()}, "
            f"values={[f'{v:.3f}' for v in act_top5.values.tolist()]}"
        )

        # Check whether argmax falls inside the action range
        global_argmax = logits[0].argmax().item()
        in_range = act_lo <= global_argmax <= act_hi
        logger.info(
            f"[LOGIT-DIAG] global argmax={global_argmax}, "
            f"in action range: {in_range}"
        )

    def _log_per_dim_logits(self, inputs: dict) -> None:
        """Step-by-step autoregressive diagnostic: for each of the 8 action
        dims, log the top-5 logits WITHIN the action range and OUTSIDE it.

        This tells us definitively:
        - Whether the model prefers action tokens or text tokens at each step
        - How much probability mass is in the action range vs. outside
        - Whether the argmax within the action range varies across dims
        - Whether feeding back predicted action tokens changes anything
        """
        input_ids = inputs["input_ids"].clone()
        pv = inputs["pixel_values"]

        if not torch.all(input_ids[:, -1] == 29871):
            trigger = torch.tensor(
                [[29871]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, trigger], dim=1)

        pad = getattr(self._model.config, "pad_to_multiple_of", 64)
        v_size = self._model.config.text_config.vocab_size - pad
        n_bins = self._model.bin_centers.shape[0]
        act_lo = v_size - n_bins
        act_hi = v_size - 1

        logger.info(f"[PER-DIM] === Autoregressive logit breakdown (inference #{self._inference_count}) ===")

        for step in range(8):
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    pixel_values=pv if step == 0 else None,
                )
            logits = outputs.logits[0, -1]  # (vocab_size,)

            # Global top-5
            global_top5 = torch.topk(logits, 5)
            global_ids = global_top5.indices.tolist()
            global_vals = [f"{v:.2f}" for v in global_top5.values.tolist()]

            # Action-range top-5
            act_logits = logits[act_lo:act_hi + 1]
            act_top5 = torch.topk(act_logits, 5)
            act_ids = (act_top5.indices + act_lo).tolist()
            act_vals = [f"{v:.2f}" for v in act_top5.values.tolist()]

            # Probability mass in action range via softmax
            probs = torch.softmax(logits, dim=-1)
            action_prob_mass = probs[act_lo:act_hi + 1].sum().item()

            global_argmax = logits.argmax().item()
            act_argmax = act_ids[0]

            logger.info(
                f"[PER-DIM] dim{step}: "
                f"global_argmax={global_argmax} "
                f"act_argmax={act_argmax} "
                f"action_prob={action_prob_mass:.4f} "
                f"global_top5={list(zip(global_ids, global_vals))} "
                f"act_top5={list(zip(act_ids, act_vals))}"
            )

            # Feed the action-range argmax back (what the logit processor would pick)
            next_tok = torch.tensor([[act_argmax]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            pv = None

        logger.info(f"[PER-DIM] === End breakdown ===")
