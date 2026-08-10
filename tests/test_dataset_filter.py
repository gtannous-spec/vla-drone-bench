"""Tests for filter_samples in airsim_dataset.py."""

import pytest

from airsim_benchmark.training.airsim_dataset import (
    APPROACH_INTENTS,
    filter_samples,
)


def _approach_sample(*, geometry: bool, detected: bool) -> dict:
    return {
        "intent": "search_then_approach",
        "instruction": "fly to tower",
        "action": [0] * 8,
        "in_fov_geometry": geometry,
        "in_fov_detected": detected,
    }


def _non_approach_sample() -> dict:
    return {
        "intent": "climb",
        "instruction": "gain altitude",
        "action": [0] * 8,
    }


def _legacy_sample() -> dict:
    """Old manifest entry without detection fields."""
    return {
        "intent": "search_then_approach",
        "instruction": "fly to tower",
        "action": [0] * 8,
        "in_fov": True,
    }


# ── mode="all" ───────────────────────────────────────────────────────

def test_all_mode_keeps_everything():
    samples = [
        _approach_sample(geometry=True, detected=True),
        _approach_sample(geometry=False, detected=False),
        _non_approach_sample(),
    ]
    assert len(filter_samples(samples, mode="all")) == 3


# ── mode="agreement_only" ────────────────────────────────────────────

def test_agreement_only_filters_approach():
    samples = [
        _approach_sample(geometry=True, detected=True),   # kept
        _approach_sample(geometry=True, detected=False),   # filtered
        _non_approach_sample(),                             # kept (non-approach)
    ]
    result = filter_samples(samples, mode="agreement_only")
    assert len(result) == 2
    assert result[0]["in_fov_detected"] is True
    assert result[1]["intent"] == "climb"


# ── mode="detected_only" ─────────────────────────────────────────────

def test_detected_only_filters_approach():
    samples = [
        _approach_sample(geometry=False, detected=True),   # kept
        _approach_sample(geometry=True, detected=False),    # filtered
        _non_approach_sample(),                              # kept
    ]
    result = filter_samples(samples, mode="detected_only")
    assert len(result) == 2
    assert result[0]["in_fov_detected"] is True
    assert result[1]["intent"] == "climb"


# ── backward compatibility ───────────────────────────────────────────

def test_missing_fields_treated_as_all():
    samples = [_legacy_sample()]
    result = filter_samples(samples, mode="agreement_only")
    assert len(result) == 1


# ── edge cases ────────────────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown filter_mode"):
        filter_samples([], mode="bad_mode")


def test_approach_intents_constant():
    assert "search_then_approach" in APPROACH_INTENTS
    assert "land_on_surface" in APPROACH_INTENTS
    assert "climb" not in APPROACH_INTENTS
