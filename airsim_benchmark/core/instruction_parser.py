"""Instruction parser for multi-step drone navigation commands.

Decomposes complex natural-language instructions into ordered subtasks
with DINO-friendly detection phrases. Supports both LLM-based parsing
(via VLMInference) and rule-based fallback for simple instructions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from airsim_benchmark.core.target_phrase import extract_target, _find_reference

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    action: str
    detect: str
    nearby: Optional[str] = None
    done_when: str = "close"


_SPLIT_PATTERN = re.compile(
    r",?\s+and\s+then\s+|,?\s+then\s+|;\s*", re.IGNORECASE
)

_ACTION_KEYWORDS = {
    "land": "land",
    "circle": "circle",
    "orbit": "circle",
    "hover": "hover",
    "wait": "hover",
    "inspect": "inspect",
    "examine": "inspect",
}

_LLM_PROMPT_TEMPLATE = """\
You are a drone mission planner. Decompose the instruction into subtasks.

Each subtask must have:
- action: one of "navigate", "land", "hover", "circle", "inspect"
- detect: a SHORT noun phrase (2-4 words max) that an object detector can find in a camera image. Use simple, concrete object names.
- nearby: (optional) another object that should be near the target
- done_when: "close" (arrived near target) or "proximity" (very close, for landing)

Known detectable objects (high confidence): {detectable}
Known undetectable objects (don't use these): rooftop, roof, tree, fence, garage, intersection, structure

If the instruction mentions an undetectable object, rephrase it using a detectable alternative.

Respond with ONLY a JSON array, no other text.

Instruction: {instruction}"""

_DEFAULT_DETECTABLE = (
    "car, red car, vehicle, truck, house, building, white house, "
    "two-story house, stop sign, parking lot, road, street, driveway, "
    "fire hydrant, street light"
)


def parse_instruction(
    instruction: str,
    model=None,
    detectable_objects: Optional[List[str]] = None,
) -> List[Subtask]:
    """Decompose a drone instruction into ordered subtasks.

    Args:
        instruction: Natural-language mission instruction.
        model: Optional VLMInference instance for LLM-based parsing.
        detectable_objects: Objects GroundingDINO can detect (from audit).

    Returns:
        Ordered list of Subtask objects.
    """
    if model is not None:
        return _parse_with_llm(instruction, model, detectable_objects)
    return _parse_rule_based(instruction)


def parse_json_subtasks(response_text: str) -> List[Subtask]:
    """Parse LLM JSON response into Subtask objects.

    Handles clean JSON, markdown-wrapped JSON, and partial output.
    Returns empty list if completely unparseable.
    """
    text = response_text.strip()

    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    subtasks = []
    for item in data:
        if not isinstance(item, dict):
            continue
        action = item.get("action", "navigate")
        detect = item.get("detect", "")
        if not detect:
            continue
        subtasks.append(Subtask(
            action=action,
            detect=detect,
            nearby=item.get("nearby"),
            done_when=item.get("done_when", "close"),
        ))
    return subtasks


def _parse_with_llm(
    instruction: str,
    model,
    detectable_objects: Optional[List[str]],
) -> List[Subtask]:
    """Use VLMInference to parse a complex instruction."""
    from PIL import Image as PILImage

    detectable_str = (
        ", ".join(detectable_objects) if detectable_objects else _DEFAULT_DETECTABLE
    )
    prompt = _LLM_PROMPT_TEMPLATE.format(
        instruction=instruction, detectable=detectable_str
    )

    dummy_img = PILImage.new("RGB", (1, 1), color=(255, 255, 255))

    try:
        if hasattr(model, "_model_family") and model._model_family == "internvl":
            response = model._query_internvl(dummy_img, prompt)
        else:
            response = model._query_llava(dummy_img, prompt)
    except Exception as e:
        logger.warning(f"LLM parsing failed ({e}), falling back to rule-based")
        return _parse_rule_based(instruction)

    subtasks = parse_json_subtasks(response)
    if not subtasks:
        logger.warning("LLM returned unparseable response, falling back to rule-based")
        return _parse_rule_based(instruction)
    return subtasks


def _parse_rule_based(instruction: str) -> List[Subtask]:
    """Rule-based fallback for simple instructions."""
    segments = _SPLIT_PATTERN.split(instruction)
    segments = [s.strip() for s in segments if s.strip()]

    subtasks = []
    for segment in segments:
        action = _detect_action(segment)
        target_info = extract_target(segment)
        detect_phrase = target_info.phrase
        nearby = _find_reference(segment, detect_phrase)

        done_when = "proximity" if action == "land" else "close"

        subtasks.append(Subtask(
            action=action,
            detect=detect_phrase,
            nearby=nearby,
            done_when=done_when,
        ))

    if not subtasks:
        target_info = extract_target(instruction)
        subtasks.append(Subtask(
            action="navigate",
            detect=target_info.phrase,
            nearby=target_info.reference_phrase,
        ))

    return subtasks


def _detect_action(segment: str) -> str:
    """Determine the action type from a text segment."""
    lower = segment.lower()
    for keyword, action in _ACTION_KEYWORDS.items():
        if keyword in lower:
            return action
    return "navigate"
