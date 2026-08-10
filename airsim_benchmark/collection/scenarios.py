from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from airsim_benchmark.collection.geometry import wrap_yaw_deg
from airsim_benchmark.collection.landmarks import LandmarkCatalog


VALID_INTENTS = {
    "search_then_approach",
    "climb",
    "descend",
    "land_on_surface",
    "land_ground",
    "ego_turn_left",
    "ego_turn_right",
    "ego_turn_around",
}


@dataclass
class Scenario:
    id: str
    intent: str
    landmark_id: Optional[str]
    weight: float
    max_hops: int
    min_altitude: float
    max_altitude: float
    templates: List[str]


@dataclass
class EpisodeSpec:
    scenario_id: str
    intent: str
    instruction: str
    start: Tuple[float, float, float]
    start_yaw: float
    max_hops: int
    min_altitude: float
    max_altitude: float
    landmark_id: Optional[str]
    landmark_position: Optional[Tuple[float, float, float]]
    landmark_radius: float
    surface_z: Optional[float]
    target_alt_ned: float


class ScenarioCatalog:
    def __init__(self, scenarios: List[Scenario], wrappers: List[str]):
        self.scenarios = scenarios
        self.wrappers = wrappers
        self.total_weight = sum(s.weight for s in scenarios)

    @classmethod
    def load(cls, path: Path) -> "ScenarioCatalog":
        data = yaml.safe_load(Path(path).read_text())
        wrappers = data.get("paraphrase_wrappers", ["{instruction}"])
        scenarios = []
        for raw in data["scenarios"]:
            intent = raw["intent"]
            if intent not in VALID_INTENTS:
                raise ValueError(f"Unknown intent {intent}")
            lm = raw.get("landmark")
            if lm == "null":
                lm = None
            scenarios.append(
                Scenario(
                    id=raw["id"],
                    intent=intent,
                    landmark_id=lm,
                    weight=float(raw["weight"]),
                    max_hops=int(raw["max_hops"]),
                    min_altitude=float(raw["min_altitude"]),
                    max_altitude=float(raw["max_altitude"]),
                    templates=list(raw["templates"]),
                )
            )
        return cls(scenarios, wrappers)

    def sample_episode(
        self,
        landmarks: LandmarkCatalog,
        rng_seed: Optional[int] = None,
        paraphrase: bool = True,
    ) -> EpisodeSpec:
        rng = random.Random(rng_seed)
        pick = rng.random() * self.total_weight
        acc = 0.0
        scenario = self.scenarios[-1]
        for s in self.scenarios:
            acc += s.weight
            if pick <= acc:
                scenario = s
                break

        template = rng.choice(scenario.templates)
        instruction = template
        if paraphrase:
            wrapper = rng.choice(self.wrappers)
            instruction = wrapper.format(instruction=template.rstrip(".").lower())

        cruise_z = -rng.uniform(scenario.min_altitude, scenario.max_altitude)
        lm_pos = None
        radius = 8.0
        surface_z = None
        if scenario.landmark_id:
            lm = landmarks.get(scenario.landmark_id)
            lm_pos = lm.position
            radius = lm.radius_m
            surface_z = lm.position[2]
            ang = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(15.0, 70.0)
            spawn = (
                lm.position[0] + dist * math.cos(ang),
                lm.position[1] + dist * math.sin(ang),
            )
            bearing = math.degrees(
                math.atan2(lm.position[1] - spawn[1], lm.position[0] - spawn[0])
            )
            yaw = wrap_yaw_deg(bearing + rng.uniform(-180.0, 180.0))
        else:
            spawn = rng.choice(landmarks.safe_spawns)
            yaw = wrap_yaw_deg(rng.uniform(-180.0, 180.0))

        target_alt = cruise_z
        if scenario.intent == "climb":
            target_alt = -scenario.max_altitude
        elif scenario.intent == "descend":
            target_alt = -scenario.min_altitude
        elif scenario.intent == "land_ground":
            target_alt = -1.5

        return EpisodeSpec(
            scenario_id=scenario.id,
            intent=scenario.intent,
            instruction=instruction,
            start=(float(spawn[0]), float(spawn[1]), cruise_z),
            start_yaw=yaw,
            max_hops=scenario.max_hops,
            min_altitude=scenario.min_altitude,
            max_altitude=scenario.max_altitude,
            landmark_id=scenario.landmark_id,
            landmark_position=lm_pos,
            landmark_radius=radius,
            surface_z=surface_z,
            target_alt_ned=target_alt,
        )
