"""
detection_inference.py — GroundingDINO wrapper for object detection.

Provides a thin interface over GroundingDINO (via HuggingFace transformers)
for zero-shot object detection. Returns structured Detection objects with
bounding box, confidence score, and matched phrase.

Used by the detection-guided controller and data collection quality gates.
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    _HAS_DETECTION = True
except ImportError as e:
    _HAS_DETECTION = False
    logger.warning(f"torch/transformers not available — detection disabled ({e})")


@dataclass
class Detection:
    """A single object detection result."""
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    phrase: str

    @property
    def center_x(self) -> float:
        return (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    def area_ratio(self, img_width: int, img_height: int) -> float:
        """Fraction of image area occupied by this detection."""
        box_area = self.width * self.height
        img_area = img_width * img_height
        if img_area <= 0:
            return 0.0
        return box_area / img_area


def bbox_to_heading_offset(center_x: float, image_width: int, fov_degrees: float) -> float:
    """Convert bbox center x-pixel to heading offset in degrees.

    Left edge → -fov/2, center → 0, right edge → +fov/2.
    """
    return (center_x / image_width - 0.5) * fov_degrees


def bbox_to_distance_estimate(
    area_ratio: float,
    close: float = 10.0,
    medium: float = 30.0,
    far: float = 60.0,
    close_threshold: float = 0.05,
    medium_threshold: float = 0.01,
) -> float:
    """Estimate distance from area ratio using simple thresholds."""
    if area_ratio > close_threshold:
        return close
    elif area_ratio > medium_threshold:
        return medium
    else:
        return far


def check_spatial_proximity(
    detections_a: List[Detection],
    detections_b: List[Detection],
    image_width: int = 640,
    image_height: int = 480,
    max_distance_ratio: float = 0.5,
) -> bool:
    """Check if two sets of detections are spatially close in the image.

    "Close" means the centers of the best detections are within
    max_distance_ratio of the image diagonal.

    Both objects being visible in the same frame from drone altitude
    is already a strong proximity signal.

    Args:
        detections_a: Detections for object A.
        detections_b: Detections for object B.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        max_distance_ratio: Max center distance as fraction of image diagonal.

    Returns:
        True if both objects detected and spatially close.
    """
    if not detections_a or not detections_b:
        return False

    best_a = max(detections_a, key=lambda d: d.score)
    best_b = max(detections_b, key=lambda d: d.score)

    dx = best_a.center_x - best_b.center_x
    dy = best_a.center_y - best_b.center_y
    distance = math.sqrt(dx * dx + dy * dy)

    diagonal = math.sqrt(image_width ** 2 + image_height ** 2)
    return distance <= max_distance_ratio * diagonal


class ObjectDetector:
    """Zero-shot object detector using GroundingDINO.

    Loads the model once at construction, then provides fast inference via
    detect(). Lazy-imports torch/transformers so the module can be imported
    without GPU availability.

    Args:
        model_id: HuggingFace model ID for GroundingDINO.
        device: CUDA device string (e.g. "cuda:0" or "cpu").
        box_threshold: Minimum confidence to keep a detection.
        text_threshold: Minimum text-matching score.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        lora_path: str = None,
    ):
        if not _HAS_DETECTION:
            raise RuntimeError(
                "PyTorch and transformers are required for object detection. "
                "Install with: pip install torch transformers"
            )

        self._model_id = model_id
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

        logger.info(f"Loading GroundingDINO from '{model_id}' on {device}...")
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, trust_remote_code=True
        )

        if lora_path and os.path.isdir(lora_path):
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, lora_path)
                self._model = self._model.merge_and_unload()
                logger.info(f"LoRA adapter merged from {lora_path}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter ({e}), using base model")

        self._model = self._model.to(device).eval()
        logger.info("GroundingDINO loaded successfully")

    def detect(self, image: np.ndarray, text_query: str) -> List[Detection]:
        """Run zero-shot detection on an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.
            text_query: Text prompt describing objects to detect
                        (e.g. "red car . mailbox . house").

        Returns:
            List of Detection objects sorted by score descending.
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        inputs = self._processor(
            images=pil_img, text=text_query, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        phrases = results.get("labels", [text_query] * len(scores))

        return self._postprocess_detections(boxes, scores, phrases)

    def detect_multi(self, image: np.ndarray, queries: List[str]) -> Dict[str, List[Detection]]:
        """Run detection for multiple object queries in a single forward pass.

        GroundingDINO supports period-separated multi-class queries.
        This method runs them all at once and groups results by query.

        Args:
            image: RGB numpy array (H, W, 3), uint8.
            queries: List of detection phrases, e.g. ["blue truck", "parking lot"]

        Returns:
            Dict mapping each query to its List[Detection].
            Example: {"blue truck": [Detection(...)], "parking lot": [Detection(...)]}
        """
        joined_query = " . ".join(queries)
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        inputs = self._processor(
            images=pil_img, text=joined_query, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("labels", [joined_query] * len(scores))

        return self._group_detections(boxes, scores, labels, queries)

    def _group_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: List[str],
        queries: List[str],
    ) -> Dict[str, List[Detection]]:
        """Group raw detections by matched query phrase.

        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes.
            scores: (N,) array of confidence scores.
            labels: List of N labels returned by the model (matched phrases).
            queries: Original query list to use as dict keys.

        Returns:
            Dict mapping each query to its list of Detection objects,
            sorted by score descending. Queries with no matches get empty lists.
        """
        grouped: Dict[str, List[Detection]] = {q: [] for q in queries}

        for i in range(len(scores)):
            if scores[i] < self._box_threshold:
                continue
            label = labels[i].strip()
            matched_query = None
            for q in queries:
                if label == q or label in q or q in label:
                    matched_query = q
                    break
            if matched_query is None:
                continue
            grouped[matched_query].append(Detection(
                bbox_xyxy=tuple(boxes[i].tolist()),
                score=float(scores[i]),
                phrase=matched_query,
            ))

        for q in grouped:
            grouped[q].sort(key=lambda d: d.score, reverse=True)

        return grouped

    def _postprocess_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        phrases: List[str],
    ) -> List[Detection]:
        """Filter and sort detections.

        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes.
            scores: (N,) array of confidence scores.
            phrases: List of N matched text phrases.

        Returns:
            List of Detection objects above box_threshold, sorted by score desc.
        """
        if len(scores) == 0:
            return []

        detections = []
        for i in range(len(scores)):
            if scores[i] >= self._box_threshold:
                detections.append(Detection(
                    bbox_xyxy=tuple(boxes[i].tolist()),
                    score=float(scores[i]),
                    phrase=phrases[i],
                ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
