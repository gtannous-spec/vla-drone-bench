import math
from pathlib import Path

import pytest

from airsim_benchmark.collection.geometry import wrap_yaw_deg
from airsim_benchmark.collection.landmarks import Landmark, LandmarkCatalog
from airsim_benchmark.collection.scenarios import Scenario, ScenarioCatalog

ROOT = Path(__file__).resolve().parents[1]
LM = ROOT / "airsim_benchmark" / "config" / "landmarks.yaml"
SC = ROOT / "airsim_benchmark" / "config" / "training_scenarios.yaml"


def test_landmarks_contain_red_car_and_roof():
    cat = LandmarkCatalog.load(LM)
    assert "red_car" in cat
    assert "rooftop_near_red_car" in cat
    assert len(cat.safe_spawns) >= 4


def test_missing_landmark_raises():
    cat = LandmarkCatalog.load(LM)
    with pytest.raises(KeyError):
        cat.get("not_a_place")


def test_scenarios_reference_existing_landmarks():
    lm = LandmarkCatalog.load(LM)
    sc = ScenarioCatalog.load(SC)
    assert sc.total_weight > 0
    for s in sc.scenarios:
        if s.landmark_id:
            lm.get(s.landmark_id)


def test_sample_episode_has_instruction_and_intent():
    lm = LandmarkCatalog.load(LM)
    sc = ScenarioCatalog.load(SC)
    ep = sc.sample_episode(lm, rng_seed=0)
    assert ep.intent
    assert len(ep.instruction) > 5
    assert len(ep.start) == 3


def test_no_compass_words_in_templates():
    sc = ScenarioCatalog.load(SC)
    banned = ("north", "east", "south", "west")
    for s in sc.scenarios:
        blob = " ".join(s.templates).lower()
        for w in banned:
            assert w not in blob, f"{s.id} contains compass word {w}"


def test_weights_sum_to_one():
    sc = ScenarioCatalog.load(SC)
    assert sc.total_weight == pytest.approx(1.0, abs=1e-6)


def test_landmark_episodes_annulus_and_yaw_spread():
    lm = LandmarkCatalog.load(LM)
    sc = ScenarioCatalog.load(SC)
    heading_errors = []
    n_landmark = 0
    for seed in range(80):
        ep = sc.sample_episode(lm, rng_seed=seed)
        if not ep.landmark_id:
            continue
        n_landmark += 1
        pos = lm.get(ep.landmark_id).position
        dist = math.hypot(ep.start[0] - pos[0], ep.start[1] - pos[1])
        assert 15.0 <= dist <= 70.0, f"seed={seed} dist={dist}"
        assert -180.0 <= ep.start_yaw <= 180.0
        assert ep.start_yaw == pytest.approx(wrap_yaw_deg(ep.start_yaw))
        bearing = math.degrees(
            math.atan2(pos[1] - ep.start[1], pos[0] - ep.start[0])
        )
        heading_errors.append(wrap_yaw_deg(bearing - ep.start_yaw))
    assert n_landmark >= 10
    assert max(abs(err) for err in heading_errors) > 45.0
    assert max(heading_errors) - min(heading_errors) > 90.0


def test_unknown_intent_raises(tmp_path):
    bad = tmp_path / "bad_scenarios.yaml"
    bad.write_text(
        "\n".join(
            [
                "paraphrase_wrappers:",
                '  - "{instruction}"',
                "scenarios:",
                "  - id: bad",
                "    intent: fly_to_moon",
                "    landmark: null",
                "    weight: 1.0",
                "    max_hops: 10",
                "    min_altitude: 4.0",
                "    max_altitude: 12.0",
                "    templates:",
                '      - "Do a thing"',
                "",
            ]
        )
    )
    with pytest.raises(ValueError, match="fly_to_moon"):
        ScenarioCatalog.load(bad)


def test_landmark_episode_does_not_require_safe_spawns():
    lm = LandmarkCatalog(
        {
            "red_car": Landmark(
                id="red_car",
                name="red car",
                kind="vehicle",
                position=(50.0, 20.0, -1.0),
                radius_m=8.0,
            )
        },
        safe_spawns=[],
    )
    sc = ScenarioCatalog(
        [
            Scenario(
                id="approach_red_car",
                intent="search_then_approach",
                landmark_id="red_car",
                weight=1.0,
                max_hops=30,
                min_altitude=6.0,
                max_altitude=16.0,
                templates=["Go to the red car"],
            )
        ],
        wrappers=["{instruction}"],
    )
    ep = sc.sample_episode(lm, rng_seed=0, paraphrase=False)
    assert ep.landmark_id == "red_car"
    dist = math.hypot(ep.start[0] - 50.0, ep.start[1] - 20.0)
    assert 15.0 <= dist <= 70.0


def test_land_ground_target_alt_is_fixed():
    lm = LandmarkCatalog.load(LM)
    sc = ScenarioCatalog(
        [
            Scenario(
                id="land_ground",
                intent="land_ground",
                landmark_id=None,
                weight=1.0,
                max_hops=20,
                min_altitude=3.0,
                max_altitude=12.0,
                templates=["Land"],
            )
        ],
        wrappers=["{instruction}"],
    )
    ep = sc.sample_episode(lm, rng_seed=0, paraphrase=False)
    assert ep.target_alt_ned == pytest.approx(-1.5)
