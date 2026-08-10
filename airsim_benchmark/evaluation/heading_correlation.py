"""
heading_correlation.py — Heading Correlation (HC) metric for VLA navigation.

HC measures the average angular difference between the drone's heading and
the bearing to the ground-truth target across all hops.  HC near 0° means
the drone is consistently navigating toward the target.  HC near 90° means
the instruction has no measurable effect on flight direction.

Coordinate system: NED (+X = North, +Y = East). Yaw 0° faces north.
"""

import math
from typing import List, Tuple


def angular_difference(angle1_deg: float, angle2_deg: float) -> float:
    """Shortest angular distance between two angles in degrees.

    Always returns a non-negative value in [0, 180].
    Handles wrap-around correctly (e.g. 350° vs 10° → 20°).
    """
    return abs((angle1_deg - angle2_deg + 180) % 360 - 180)


def bearing_to_target(position: Tuple[float, float],
                      target: Tuple[float, float]) -> float:
    """Bearing from *position* to *target* in degrees (NED convention).

    0° = North (+X), 90° = East (+Y).
    """
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    return math.degrees(math.atan2(dy, dx))


def heading_correlation(
    positions: List[Tuple[float, float]],
    headings_deg: List[float],
    target_xy: Tuple[float, float],
) -> float:
    """Mean angular error between drone heading and bearing to target.

    Args:
        positions:    (x, y) drone positions at each hop.
        headings_deg: Drone headings in degrees at each hop.
        target_xy:    (x, y) of the ground-truth target.

    Returns:
        Mean angular difference in degrees.  ``float('nan')`` if there
        are no valid hops (empty input or all hops within 1 m of target).
    """
    if len(positions) != len(headings_deg):
        raise ValueError(
            f"positions ({len(positions)}) and headings_deg "
            f"({len(headings_deg)}) must have the same length"
        )

    diffs: List[float] = []
    for pos, hdg in zip(positions, headings_deg):
        dx = target_xy[0] - pos[0]
        dy = target_xy[1] - pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            continue
        brg = bearing_to_target(pos, target_xy)
        diffs.append(angular_difference(hdg, brg))

    if not diffs:
        return float("nan")
    return sum(diffs) / len(diffs)
