from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

GATES = {
    "dim0_stop_nonzero_pct": 8.0,
    "dim2_yawL_nonzero_pct": 25.0,
    "dim3_yawR_nonzero_pct": 25.0,
    "dim4_up_nonzero_pct": 15.0,
    "dim5_down_nonzero_pct": 15.0,
    "unique_instructions": 40,
    "in_fov_pct": 50.0,
}


def evaluate_collection_gates(
    actions: np.ndarray,
    instructions: Sequence[str],
    in_fov: Sequence[bool],
    approach_mask: Sequence[bool] | None = None,
) -> Tuple[bool, dict]:
    n = len(actions)

    def pct(col: int) -> float:
        return float(100.0 * np.count_nonzero(actions[:, col]) / max(n, 1))

    fov_arr = np.asarray(in_fov, dtype=bool)
    if approach_mask is None:
        approach_arr = np.ones(n, dtype=bool)
    else:
        approach_arr = np.asarray(approach_mask, dtype=bool)
    n_approach = int(np.count_nonzero(approach_arr))
    in_fov_pct = float(
        100.0 * np.count_nonzero(fov_arr & approach_arr) / max(n_approach, 1)
    )

    report = {
        "dim0_stop_nonzero_pct": pct(0),
        "dim2_yawL_nonzero_pct": pct(2),
        "dim3_yawR_nonzero_pct": pct(3),
        "dim4_up_nonzero_pct": pct(4),
        "dim5_down_nonzero_pct": pct(5),
        "unique_instructions": len(set(instructions)),
        "in_fov_pct": in_fov_pct,
    }
    ok = (
        report["dim0_stop_nonzero_pct"] >= GATES["dim0_stop_nonzero_pct"]
        and report["dim2_yawL_nonzero_pct"] >= GATES["dim2_yawL_nonzero_pct"]
        and report["dim3_yawR_nonzero_pct"] >= GATES["dim3_yawR_nonzero_pct"]
        and report["dim4_up_nonzero_pct"] >= GATES["dim4_up_nonzero_pct"]
        and report["dim5_down_nonzero_pct"] >= GATES["dim5_down_nonzero_pct"]
        and report["unique_instructions"] >= GATES["unique_instructions"]
        and report["in_fov_pct"] >= GATES["in_fov_pct"]
    )
    return ok, report
