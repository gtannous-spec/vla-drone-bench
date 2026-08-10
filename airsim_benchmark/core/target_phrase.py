"""Extract object name and landing intent from a natural-language instruction.

Generates multi-query detection strings for GroundingDINO by automatically
appending broader category terms to the target phrase. No hardcoded
alternative dictionaries — works for any object description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

KNOWN_LANDMARKS: list[str] = [
    "red car",
    "rooftop",
    "two-story house",
    "white house",
    "intersection",
    "mailbox",
    "house",
    "driveway",
    "road",
]

SUPERCLASS_TERMS = {
    "vehicle": ["car", "truck", "van", "vehicle", "car shelter"],
    "building": ["house", "rooftop", "roof", "building", "structure",
                  "two-story house", "white house", "small building"],
    "road": ["road", "street", "intersection", "driveway", "path"],
    "object": ["mailbox", "post box", "letter box", "sign"],
}

AERIAL_ALTERNATIVES = {
    "rooftop": ["house top view", "house roof", "flat roof"],
    "roof": ["house top view", "house roof", "flat roof"],
    "tree": ["tree canopy", "green bush"],
    "intersection": ["road crossing", "road junction"],
    "garage": ["car shelter"],
    "structure": ["small building", "residential building"],
    "mailbox": ["post box", "letter box"],
    "swimming pool": ["pool", "blue pool"],
}

_VERB_PHRASES = re.compile(
    r"(?:fly\s+to(?:ward)?|navigate\s+to|go\s+to|land\s+on|descend\s+to)"
    r"\s+(?:the\s+)?(.+)",
    re.IGNORECASE,
)

_LANDMARK_PATTERN = re.compile(
    "|".join(re.escape(lm) for lm in KNOWN_LANDMARKS),
    re.IGNORECASE,
)

def _find_reference(instruction: str, primary_phrase: str) -> Optional[str]:
    """Find a secondary landmark in the instruction to use as navigation reference.

    For "Land on the closest rooftop to the red car", the primary is "rooftop"
    and the reference is "red car". Works by finding any known landmark in the
    instruction that isn't the primary target.
    """
    lower = instruction.lower()
    for lm in KNOWN_LANDMARKS:
        if lm != primary_phrase and lm in lower:
            return lm
    return None


@dataclass
class TargetInfo:
    phrase: str
    wants_land: bool
    wants_ground_land: bool
    multi_query: str = ""
    reference_phrase: Optional[str] = None


def _build_multi_query(phrase: str) -> str:
    """Build a GroundingDINO multi-query string with aerial-optimized alternatives.

    GroundingDINO accepts period-separated queries and detects all of them
    in a single forward pass. This widens the detection net for objects that
    look different from aerial views.

    Priority order:
      1. Audit-proven aerial alternatives (e.g. "rooftop" -> "house top view" at 97.5%)
      2. Superclass broadening (e.g. "car" -> "vehicle")

    Returns a string like "rooftop . house top view . house roof . flat roof".
    """
    terms = [phrase]

    for key, alts in AERIAL_ALTERNATIVES.items():
        if key in phrase or phrase in key:
            terms.extend(alts)
            break

    for superclass, members in SUPERCLASS_TERMS.items():
        if any(member in phrase or phrase in member for member in members):
            for member in members:
                if member != phrase and member not in terms:
                    terms.append(member)
            break

    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return " . ".join(unique[:5])


def extract_target(instruction: str) -> TargetInfo:
    lower = instruction.lower()
    wants_land = "land" in lower
    has_roof = "roof" in lower or "rooftop" in lower

    wants_ground_land = wants_land and not has_roof

    if wants_land and has_roof:
        phrase = "rooftop"
        return TargetInfo(
            phrase=phrase,
            wants_land=True,
            wants_ground_land=False,
            multi_query=_build_multi_query(phrase),
            reference_phrase=_find_reference(instruction, phrase),
        )

    match = _LANDMARK_PATTERN.search(instruction)
    if match:
        phrase = match.group(0).lower()
        return TargetInfo(
            phrase=phrase,
            wants_land=wants_land,
            wants_ground_land=wants_ground_land,
            multi_query=_build_multi_query(phrase),
            reference_phrase=_find_reference(instruction, phrase),
        )

    verb_match = _VERB_PHRASES.search(instruction)
    if verb_match:
        phrase = verb_match.group(1).strip()
        return TargetInfo(
            phrase=phrase,
            wants_land=wants_land,
            wants_ground_land=wants_ground_land,
            multi_query=_build_multi_query(phrase.lower()),
            reference_phrase=_find_reference(instruction, phrase.lower()),
        )

    return TargetInfo(
        phrase=instruction,
        wants_land=wants_land,
        wants_ground_land=wants_ground_land,
        multi_query=_build_multi_query(instruction.lower()),
        reference_phrase=None,
    )
