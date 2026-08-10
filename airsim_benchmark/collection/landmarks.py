from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class Landmark:
    id: str
    name: str
    kind: str
    position: Tuple[float, float, float]
    radius_m: float


class LandmarkCatalog:
    def __init__(self, landmarks: Dict[str, Landmark], safe_spawns: List[Tuple[float, float]]):
        self._landmarks = landmarks
        self.safe_spawns = safe_spawns

    def __contains__(self, key: str) -> bool:
        return key in self._landmarks

    def get(self, landmark_id: str) -> Landmark:
        if landmark_id not in self._landmarks:
            raise KeyError(f"Unknown landmark '{landmark_id}'")
        return self._landmarks[landmark_id]

    @classmethod
    def load(cls, path: Path) -> "LandmarkCatalog":
        data = yaml.safe_load(Path(path).read_text())
        lms = {}
        for lid, raw in data["landmarks"].items():
            pos = tuple(float(x) for x in raw["position"])
            lms[lid] = Landmark(
                id=lid,
                name=raw["name"],
                kind=raw["kind"],
                position=(pos[0], pos[1], pos[2]),
                radius_m=float(raw["radius_m"]),
            )
        spawns = [tuple(float(x) for x in p) for p in data.get("safe_spawns", [])]
        return cls(lms, spawns)
