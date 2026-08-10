import numpy as np
from airsim_benchmark.collection.quality import evaluate_collection_gates


def test_gates_fail_on_v3_style_actions():
    n = 100
    actions = np.zeros((n, 8))
    actions[:, 1] = 5.0
    ok, report = evaluate_collection_gates(
        actions, instructions=["fly"] * n, in_fov=[False] * n
    )
    assert ok is False
    assert report["dim5_down_nonzero_pct"] == 0.0


def test_gates_pass_diverse():
    n = 200
    actions = np.zeros((n, 8))
    actions[:40, 0] = 1
    actions[40:100, 1] = 4
    actions[40:90, 2] = 10
    actions[90:140, 3] = 10
    actions[120:160, 4] = 1.5
    actions[160:200, 5] = 1.5
    inst = [f"instr-{i%50}" for i in range(n)]
    fov = [True] * 120 + [False] * 80
    ok, report = evaluate_collection_gates(actions, inst, fov)
    assert ok is True, report


def test_in_fov_gate_uses_approach_mask_only():
    n = 100
    actions = np.zeros((n, 8))
    actions[:20, 0] = 1
    actions[20:50, 2] = 10
    actions[50:80, 3] = 10
    actions[80:90, 4] = 1
    actions[90:100, 5] = 1
    inst = [f"i{i%40}" for i in range(n)]
    # All samples in_fov False except 10 approach samples that are True
    in_fov = [False] * 90 + [True] * 10
    approach = [False] * 90 + [True] * 10
    # 10/10 approach in FOV = 100% — should pass in_fov even if global in_fov is 10%
    ok, report = evaluate_collection_gates(actions, inst, in_fov, approach_mask=approach)
    assert report["in_fov_pct"] == 100.0
