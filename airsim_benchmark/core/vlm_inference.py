"""
vlm_inference.py — VLM-based scene understanding for drone navigation.

Uses a Vision-Language Model (InternVL2 or LLaVA) as a scene understanding
module. Instead of raw action deltas, the VLM answers structured spatial
questions ("where is the target?") and returns a NavigationHint with
heading offset, distance estimate, and confidence.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    _HAS_VLM = True
except ImportError as e:
    _HAS_VLM = False
    logger.warning(f"torch/transformers not available — VLM inference disabled ({e})")


@dataclass
class NavigationHint:
    """Structured output from VLM scene analysis."""
    heading_offset_deg: float
    distance_estimate_m: float
    confidence: float
    reasoning: str


_CONFIDENCE_MAP = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3, "NONE": 0.0}

NAVIGATION_PROMPT = """\
You are a drone navigation system. You see this image from the drone's front camera flying over a suburban neighborhood.

Task: {instruction}

Look at the image carefully. Where is the target object in this image?
Answer with EXACTLY these 4 lines:
POSITION: <FAR_LEFT, LEFT, CENTER_LEFT, CENTER, CENTER_RIGHT, RIGHT, FAR_RIGHT, or NOT_VISIBLE>
DISTANCE: <CLOSE, MEDIUM, FAR, or UNKNOWN>
CONFIDENCE: <HIGH, MEDIUM, LOW, or NONE>
REASONING: <one sentence about what you see and where the target is>"""

_POSITION_TO_HEADING = {
    "FAR_LEFT": -60.0,
    "LEFT": -40.0,
    "CENTER_LEFT": -20.0,
    "CENTER": 0.0,
    "CENTER_RIGHT": 20.0,
    "RIGHT": 40.0,
    "FAR_RIGHT": 60.0,
    "NOT_VISIBLE": None,
}

_DISTANCE_TO_METERS = {
    "CLOSE": 10.0,
    "MEDIUM": 30.0,
    "FAR": 60.0,
    "UNKNOWN": 30.0,
}


def _parse_vlm_response(text: str) -> NavigationHint:
    """Parse the VLM's structured response into a NavigationHint."""
    heading = 0.0
    distance = 30.0
    confidence = 0.0
    reasoning = ""

    pos_match = re.search(
        r"POSITION:\s*(FAR_LEFT|LEFT|CENTER_LEFT|CENTER|CENTER_RIGHT|RIGHT|FAR_RIGHT|NOT_VISIBLE)",
        text, re.IGNORECASE
    )
    if pos_match:
        pos_key = pos_match.group(1).upper()
        heading_val = _POSITION_TO_HEADING.get(pos_key)
        if heading_val is not None:
            heading = heading_val
        else:
            confidence = 0.0

    heading_match = re.search(r"HEADING:\s*(-?\d+(?:\.\d+)?)", text)
    if heading_match and not pos_match:
        heading = float(heading_match.group(1))
        heading = max(-180.0, min(180.0, heading))

    dist_match = re.search(
        r"DISTANCE:\s*(CLOSE|MEDIUM|FAR|UNKNOWN)",
        text, re.IGNORECASE
    )
    if dist_match:
        distance = _DISTANCE_TO_METERS.get(dist_match.group(1).upper(), 30.0)
    else:
        num_dist = re.search(r"DISTANCE:\s*(\d+(?:\.\d+)?)", text)
        if num_dist:
            distance = float(num_dist.group(1))

    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW|NONE)", text, re.IGNORECASE)
    if conf_match:
        confidence = _CONFIDENCE_MAP.get(conf_match.group(1).upper(), 0.0)

    reason_match = re.search(r"REASONING:\s*(.+)", text)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    if not pos_match and not heading_match:
        text_lower = text.lower()
        if "not visible" in text_lower or "cannot see" in text_lower or "not seen" in text_lower:
            confidence = 0.0
        elif "far left" in text_lower or "far-left" in text_lower:
            heading = -60.0
        elif "far right" in text_lower or "far-right" in text_lower:
            heading = 60.0
        elif "center-left" in text_lower or "center left" in text_lower or "slightly left" in text_lower:
            heading = -20.0
        elif "center-right" in text_lower or "center right" in text_lower or "slightly right" in text_lower:
            heading = 20.0
        elif "left side" in text_lower or "to the left" in text_lower or "on the left" in text_lower:
            heading = -40.0
        elif "right side" in text_lower or "to the right" in text_lower or "on the right" in text_lower:
            heading = 40.0
        elif "left" in text_lower and "right" not in text_lower:
            heading = -40.0
        elif "right" in text_lower and "left" not in text_lower:
            heading = 40.0
        elif "center" in text_lower or "ahead" in text_lower or "directly ahead" in text_lower or "straight" in text_lower:
            heading = 0.0

        if confidence == 0.0 and not conf_match:
            conf_inferred = any(w in text_lower for w in ["visible", "see", "parked", "ahead", "center", "left", "right"])
            if conf_inferred:
                confidence = 0.6

    if not pos_match and not heading_match and not conf_match:
        text_lower = text.lower()
        if any(w in text_lower for w in ["see", "visible", "parked", "ahead", "car", "house", "roof"]):
            confidence = 0.6
        else:
            confidence = 0.0
            reasoning = f"Failed to parse VLM response: {text[:100]}"

    if pos_match and pos_key == "NOT_VISIBLE":
        confidence = 0.0
        reasoning = f"Target not visible: {reasoning}"

    return NavigationHint(
        heading_offset_deg=heading,
        distance_estimate_m=distance,
        confidence=confidence,
        reasoning=reasoning,
    )


class VLMInference:
    """Vision-Language Model inference for scene understanding.

    Supports InternVL2 and LLaVA model families. Loads the model once
    at construction, then provides fast structured queries.

    Args:
        model_path: HuggingFace model ID or local path.
        device: CUDA device string.
        max_new_tokens: Maximum tokens for VLM response.
    """

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2-8B",
        device: str = "cuda:0",
        max_new_tokens: int = 100,
    ):
        if not _HAS_VLM:
            raise RuntimeError(
                "PyTorch and transformers are required for VLM inference. "
                "Install with: pip install torch transformers accelerate"
            )

        self._device = device
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._last_inference_time: float = 0.0
        self._inference_count: int = 0
        self._model_family: str = "unknown"

        logger.info(f"Loading VLM from '{model_path}' on {device}...")
        t0 = time.time()

        self._detect_model_family(model_path)

        if self._model_family == "internvl":
            self._load_internvl(model_path, device)
        else:
            self._load_llava(model_path, device)

        load_time = time.time() - t0
        logger.info(f"VLM loaded in {load_time:.1f}s (family={self._model_family})")

    def _detect_model_family(self, model_path: str) -> None:
        path_lower = model_path.lower()
        if "internvl" in path_lower:
            self._model_family = "internvl"
        elif "llava" in path_lower:
            self._model_family = "llava"
        else:
            self._model_family = "llava"
            logger.warning(f"Unknown model family for '{model_path}', defaulting to llava")

    def _load_internvl(self, model_path: str, device: str) -> None:
        from transformers import AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self._tokenizer = None
        logger.info(f"Loaded InternVL via native transformers (AutoModelForImageTextToText)")

    def _load_llava(self, model_path: str, device: str) -> None:
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        self._tokenizer = None

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time

    def query(self, image: np.ndarray, instruction: str) -> NavigationHint:
        """Query the VLM about the scene.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.
            instruction: Natural language task instruction.

        Returns:
            NavigationHint with heading, distance, confidence, and reasoning.
        """
        from PIL import Image as PILImage

        prompt = NAVIGATION_PROMPT.format(instruction=instruction)
        pil_img = PILImage.fromarray(image)

        try:
            t0 = time.time()
            if self._model_family == "internvl":
                response_text = self._query_internvl(pil_img, prompt)
            else:
                response_text = self._query_llava(pil_img, prompt)
            self._last_inference_time = time.time() - t0
            self._inference_count += 1

        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            self._last_inference_time = 0.0
            return NavigationHint(
                heading_offset_deg=0.0,
                distance_estimate_m=30.0,
                confidence=0.0,
                reasoning=f"Inference error: {e}",
            )

        hint = _parse_vlm_response(response_text)

        if self._inference_count <= 3 or self._inference_count % 10 == 0:
            logger.debug(
                f"VLM query #{self._inference_count} "
                f"({self._last_inference_time:.2f}s): "
                f"heading={hint.heading_offset_deg:.0f}°, "
                f"dist={hint.distance_estimate_m:.0f}m, "
                f"conf={hint.confidence:.1f}, "
                f"reason='{hint.reasoning[:60]}'"
            )

        return hint

    def _query_internvl(self, pil_img, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            images=pil_img, text=text, return_tensors="pt"
        ).to(self._device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        input_len = inputs["input_ids"].shape[-1]
        response = self._processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        return response.strip()

    # InternVL2 custom preprocessing removed — InternVL3 uses native AutoProcessor

    # _internvl_preprocess removed — replaced by _internvl_load_image + _dynamic_preprocess

    def _query_llava(self, pil_img, prompt: str) -> str:
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text_prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=pil_img, text=text_prompt, return_tensors="pt"
        ).to(self._device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        input_len = inputs["input_ids"].shape[-1]
        response = self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return response
