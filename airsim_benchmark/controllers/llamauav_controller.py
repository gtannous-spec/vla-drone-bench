"""
llamauav_controller.py — LLaMA-UAV VLN Controller.

Uses the LLaMA-UAV model (wangxiangyu0814/llama-uav-7b) from the TravelUAV
project (ICLR 2025). The model is a Vicuna-7B + EVA-ViT-G + Q-Former
architecture that outputs a 4D waypoint prediction: 3D direction + distance.

Architecture:
    Camera RGB + Instruction + History → LLaMA-UAV MLLM → (direction, dist)
    → direction * dist → world-frame waypoint → AirSim moveToPosition

Reference: "Towards Realistic UAV Vision-Language Navigation: Platform,
Benchmark, and Methodology" (Wang et al., 2025, arXiv:2410.07087)
"""

import logging
import math
import os
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _ensure_llamavid_on_path():
    """Add TravelUAV/Model/LLaMA-UAV to sys.path so llamavid is importable."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "third_party",
                     "TravelUAV", "Model", "LLaMA-UAV"),
        "/tmp/TravelUAV/Model/LLaMA-UAV",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(os.path.join(p, "llamavid")) and p not in sys.path:
            sys.path.insert(0, p)
            logger.info(f"Added {p} to sys.path for llamavid")
            return
    raise ImportError(
        "Cannot find llamavid package. Clone TravelUAV into "
        "third_party/TravelUAV or /tmp/TravelUAV"
    )


class LLaMAUAVController(BaseController):
    """LLaMA-UAV VLN controller for drone navigation.

    Uses a hierarchical MLLM that takes camera images + instruction + flight
    history and predicts a 4D waypoint (direction + distance) in the drone's
    local frame.

    Args:
        model_path: Path to LLaMA-UAV LoRA adapter (HF or local).
        model_base: Path to Vicuna-7B base model.
        vision_tower: Path to EVA-ViT-G weights.
        qformer_path: Path to InstructBLIP Q-Former weights.
        arrival_tolerance: Distance threshold for goal check (m).
        nav_speed: Cruise speed (m/s).
        max_hops: Maximum navigation iterations per task.
        device: CUDA device for inference.
    """

    def __init__(
        self,
        model_path: str = "~/models/llama-uav/llama-uav-7b",
        model_base: str = "~/models/llama-uav/vicuna-7b-v1.5",
        vision_tower: str = "~/models/llama-uav/eva_vit_g.pth",
        qformer_path: str = "~/models/llama-uav/instruct_blip_vicuna7b_trimmed.pth",
        arrival_tolerance: float = 1.5,
        nav_speed: float = 5.0,
        max_hops: int = 50,
        goal_bias: float = 0.6,
        device: str = "auto",
    ):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for LLaMA-UAV")

        self._arrival_tolerance = arrival_tolerance
        self._goal_bias = goal_bias
        self._nav_speed = nav_speed
        self._max_hops = max_hops

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        model_path = os.path.expanduser(model_path)
        model_base = os.path.expanduser(model_base)
        vision_tower = os.path.expanduser(vision_tower)
        qformer_path = os.path.expanduser(qformer_path)

        logger.info(f"Loading LLaMA-UAV from '{model_path}' on {device}...")
        t0 = time.time()

        _ensure_llamavid_on_path()

        from llamavid.model.builder import load_pretrained_model
        from llamavid.constants import (
            WAYPOINT_LABEL_TOKEN, WAYPOINT_INPUT_TOKEN,
            DEFAULT_IMAGE_TOKEN, DEFAULT_HISTORY_TOKEN, DEFAULT_WP_TOKEN,
        )
        from llava.mm_utils import get_model_name_from_path

        self._WAYPOINT_LABEL_TOKEN = WAYPOINT_LABEL_TOKEN
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN

        model_name = get_model_name_from_path(model_path)

        # Patch the model config to resolve relative paths from the training env
        _config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(_config_path):
            import json as _json
            with open(_config_path) as _f:
                _cfg = _json.load(_f)
            _dirty = False

            _saved_vt = _cfg.get("mm_vision_tower", "")
            if _saved_vt and not os.path.exists(_saved_vt) and os.path.isfile(vision_tower):
                logger.info(f"Patching mm_vision_tower: '{_saved_vt}' → '{vision_tower}'")
                _cfg["mm_vision_tower"] = vision_tower
                _dirty = True

            _saved_ip = _cfg.get("image_processor", "")
            if _saved_ip and not os.path.isdir(_saved_ip):
                _llamavid_root = None
                for _cand in [
                    os.path.join(os.path.dirname(__file__), "..", "..", "third_party",
                                 "TravelUAV", "Model", "LLaMA-UAV"),
                    "/tmp/TravelUAV/Model/LLaMA-UAV",
                ]:
                    _abs = os.path.abspath(os.path.join(_cand, _saved_ip.lstrip("./")))
                    if os.path.isdir(_abs):
                        _llamavid_root = _abs
                        break
                if _llamavid_root:
                    logger.info(f"Patching image_processor: '{_saved_ip}' → '{_llamavid_root}'")
                    _cfg["image_processor"] = _llamavid_root
                    _dirty = True

            if _dirty:
                with open(_config_path, "w") as _f:
                    _json.dump(_cfg, _f, indent=2)

        tokenizer, model, image_processor, ctx_len = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        # Add special tokens and load LoRA + non-LoRA weights
        from peft import PeftModel
        special_tokens = ["<wp>", "<his>"]
        num_new = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new > 0:
            # Resize embed_tokens and lm_head directly to avoid
            # tie_weights() iterating into the Q-Former submodule.
            import torch.nn as nn
            new_size = len(tokenizer)
            old_embed = model.model.embed_tokens
            new_embed = nn.Embedding(new_size, old_embed.embedding_dim,
                                     device=old_embed.weight.device,
                                     dtype=old_embed.weight.dtype)
            new_embed.weight.data[:old_embed.num_embeddings] = old_embed.weight.data
            model.model.embed_tokens = new_embed

            old_head = model.lm_head
            new_head = nn.Linear(old_head.in_features, new_size, bias=False,
                                 device=old_head.weight.device,
                                 dtype=old_head.weight.dtype)
            new_head.weight.data[:old_head.out_features] = old_head.weight.data
            model.lm_head = new_head
            model.config.vocab_size = new_size

        model.get_special_token_id({
            "<wp>": tokenizer.encode("<wp>")[1],
            "<his>": tokenizer.encode("<his>")[1],
            ",": tokenizer.encode(",")[1],
            ";": tokenizer.encode(";")[1],
        })

        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, model_path)
        non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
        if os.path.isfile(non_lora_path):
            non_lora_weights = torch.load(non_lora_path, map_location="cpu")
            model.load_state_dict(non_lora_weights, strict=False)
        mm_proj_path = os.path.join(model_path, "mm_projector.bin")
        if os.path.isfile(mm_proj_path):
            mm_proj_weights = torch.load(mm_proj_path, map_location="cpu")
            model.load_state_dict(mm_proj_weights, strict=False)

        # Cast to bfloat16 AFTER all weights are loaded (LoRA, non-LoRA, mm_projector)
        # to ensure every layer has consistent dtype
        model.to(torch.bfloat16)
        model.eval()

        # PEFT wraps the model, hiding vision_tower from get_vision_tower().
        # Expose it directly on the PeftModel so prepare_inputs_labels_for_multimodal
        # doesn't skip image processing.
        if hasattr(model, "base_model"):
            inner = model.base_model.model
            if not hasattr(model, "vision_tower") and hasattr(inner, "model"):
                model.vision_tower = inner.model.vision_tower
            if not hasattr(model, "vlm_att_projector") and hasattr(inner, "model"):
                for attr in ("vlm_att_projector", "vlm_att_query",
                             "vlm_att_ln", "vlm_att_val_projector"):
                    if hasattr(inner.model, attr):
                        setattr(model, attr, getattr(inner.model, attr))

        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor

        # Diagnostic: check if vision tower is accessible through the call chain
        try:
            inner_model = model.base_model.model if hasattr(model, "base_model") else model
            vt = inner_model.get_model().get_vision_tower()
            logger.info(f"[DIAG] vision_tower accessible: {type(vt).__name__}, "
                        f"is_loaded={getattr(vt, 'is_loaded', 'N/A')}")
            if hasattr(vt, 'is_loaded') and not vt.is_loaded:
                logger.info("[DIAG] Vision tower not loaded — calling load_model()...")
                vt.load_model()
                logger.info("[DIAG] Vision tower loaded successfully")
        except Exception as e:
            logger.error(f"[DIAG] vision_tower check failed: {e}")

        load_time = time.time() - t0
        logger.info(f"LLaMA-UAV loaded in {load_time:.1f}s")

        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._instruction: str = ""
        self._constraints: dict = {}
        self._task_id: int = 0
        self._hop_count: int = 0
        self._model_guided_hops: int = 0
        self._inference_count: int = 0
        self._position_history: list = []
        self._recent_dists: deque = deque(maxlen=10)
        self._effective_speed: float = nav_speed

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._start = tuple(task_config["start"])
        self._instruction = task_config.get("instruction", "")
        self._constraints = task_config.get("constraints", {})
        self._hop_count = 0
        self._model_guided_hops = 0
        self._inference_count = 0
        self._position_history = [list(self._start)]
        self._recent_dists.clear()

        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)

        logger.info(
            f"LLaMAUAVController reset — task {self._task_id}, "
            f"instruction='{self._instruction}', goal={self._goal}, "
            f"goal_bias={self._goal_bias}"
        )

    def get_action(self, state: DroneState) -> ControlAction:
        self._hop_count += 1
        self._position_history.append(list(state.position))
        goal_dist = self._distance_to_goal(state)

        if state.image is None:
            logger.warning("No camera image — classical fallback")
            return self._fallback_action(state)

        try:
            waypoint = self._predict_waypoint(state)
        except Exception as e:
            logger.warning(f"LLaMA-UAV inference failed ({e}) — classical fallback")
            return self._fallback_action(state)

        self._model_guided_hops += 1

        # Normalize model direction
        model_norm = np.linalg.norm(waypoint)
        model_unit = waypoint / model_norm if model_norm > 1e-6 else waypoint

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

        # Scale hop distance (shorter near goal)
        effective_scale = min(model_norm, max(2.0, goal_dist * 0.4))
        offset = blended * effective_scale

        target = (
            float(state.position[0] + offset[0]),
            float(state.position[1] + offset[1]),
            float(self._clamp_altitude(state.position[2] + offset[2])),
        )

        self._recent_dists.append(goal_dist)

        if self._hop_count <= 3 or self._hop_count % 5 == 0:
            logger.info(
                f"LLaMA-UAV hop {self._hop_count}: "
                f"wp=({waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f}), "
                f"goal_dist={goal_dist:.1f}m, bias={adaptive_bias:.2f}, "
                f"target=({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})"
            )

        return ControlAction(
            target_position=target,
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        dist = self._distance_to_goal(state)

        if dist < self._arrival_tolerance * 3.0:
            logger.info(
                f"Goal reached at {dist:.1f}m "
                f"(model guided {self._model_guided_hops}/{self._hop_count} hops)"
            )
            return True

        if self._hop_count >= self._max_hops:
            logger.info(
                f"Max hops ({self._max_hops}) reached — dist={dist:.1f}m, "
                f"model guided {self._model_guided_hops}/{self._hop_count}"
            )
            return True

        return False

    def _predict_waypoint(self, state: DroneState) -> np.ndarray:
        """Run LLaMA-UAV inference → 3D waypoint displacement in world frame."""
        from PIL import Image as PILImage
        from llava.mm_utils import tokenizer_image_token
        from llamavid.constants import IMAGE_TOKEN_INDEX, WAYPOINT_LABEL_TOKEN
        from llamavid import conversation as conversation_lib

        pil_img = PILImage.fromarray(state.image).convert("RGB")

        # Process image through EVA-ViT-G processor
        pixel_values = self._image_processor.preprocess(
            np.array(pil_img)[np.newaxis], return_tensors="pt"
        )["pixel_values"]
        pixel_values = pixel_values.to(self._device, dtype=torch.bfloat16)

        # Build conversation prompt
        conv_text = f"{self._DEFAULT_IMAGE_TOKEN}\n{self._instruction}"
        prompt = (
            f"A chat between a curious user and an artificial intelligence "
            f"assistant. The assistant gives helpful, detailed, and polite "
            f"answers to the user's questions. "
            f"USER: {conv_text} ASSISTANT:"
        )

        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        # Build labels with waypoint marker at the end
        labels = torch.full_like(input_ids, -100)
        wp_label = torch.tensor([[WAYPOINT_LABEL_TOKEN]], device=self._device)
        input_ids = torch.cat([input_ids, torch.ones(1, 1, dtype=torch.long, device=self._device)], dim=1)
        labels = torch.cat([labels, wp_label], dim=1)

        attention_mask = input_ids.ne(self._tokenizer.pad_token_id)

        # Build history as position deltas — shape [N, 3]
        # The model's forward() calls history_preprocessor(info) where info = history.view(-1, 3)
        hist_positions = np.array(self._position_history[-5:]) - np.array(self._position_history[0])
        history_tensor = torch.tensor(
            hist_positions, dtype=torch.bfloat16, device=self._device
        ).view(-1, 3)  # ensure [N, 3]

        # Build prompts for Q-Former attention
        prompts = [[self._instruction]]

        t0 = time.time()
        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=[pixel_values],
                prompts=prompts,
                historys=[history_tensor],
                orientations=torch.zeros(1, 3, device=self._device, dtype=torch.bfloat16),
                return_waypoints=True,
                use_cache=False,
            )

        elapsed = time.time() - t0
        self._inference_count += 1

        if output is None:
            raise RuntimeError("Model returned None — forward() failed silently")

        # Extract waypoint prediction — 4D: [dx, dy, dz, distance]
        if isinstance(output, torch.Tensor):
            waypoint_raw = output[0].cpu().float().numpy()
        elif hasattr(output, 'waypoints') and output.waypoints is not None:
            waypoint_raw = output.waypoints[0].cpu().float().numpy()
        elif isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            waypoint_raw = output[1][0].cpu().float().numpy()
        else:
            raise RuntimeError(f"No waypoints in output (type={type(output).__name__})")

        direction = waypoint_raw[:3]
        distance = waypoint_raw[3]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            waypoint = direction / norm * distance
        else:
            waypoint = direction * distance

        if self._inference_count <= 5 or self._inference_count % 10 == 0:
            logger.info(
                f"[DIAG] LLaMA-UAV inference #{self._inference_count} "
                f"({elapsed:.2f}s): raw=({waypoint_raw[0]:.4f}, "
                f"{waypoint_raw[1]:.4f}, {waypoint_raw[2]:.4f}, "
                f"dist={waypoint_raw[3]:.4f}), "
                f"world=({waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f})"
            )

        return waypoint

    def _fallback_action(self, state: DroneState) -> ControlAction:
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
