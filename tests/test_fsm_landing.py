"""Landing policy for multi-leg missions: skip street land after leg 0.

Does not instantiate AirSim.
"""

import inspect
from pathlib import Path

from airsim_benchmark.core.drone_fsm import DroneFSM, should_ground_land

_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "airsim_benchmark"
    / "runner"
    / "benchmark_runner.py"
)
_FSM = (
    Path(__file__).resolve().parents[1]
    / "airsim_benchmark"
    / "core"
    / "drone_fsm.py"
)


def test_should_ground_land_plain_land():
    assert should_ground_land("Land") is True


def test_should_ground_land_rooftop_false():
    assert should_ground_land("Land on the closest rooftop to the red car") is False


def test_should_ground_land_fly_false():
    assert should_ground_land("Fly toward the red car") is False


def test_should_ground_land_descend_ground_true():
    assert should_ground_land("Descend and land on the ground") is True


def test_execute_land_at_end_defaults_true():
    sig = inspect.signature(DroneFSM.execute)
    assert "land_at_end" in sig.parameters
    assert sig.parameters["land_at_end"].default is True


def test_execute_only_lands_when_land_at_end():
    src = _FSM.read_text(encoding="utf-8")
    assert "if land_at_end:" in src
    assert "_phase_land()" in src


def test_mission_first_leg_skips_end_land():
    src = _RUNNER.read_text(encoding="utf-8")
    start = src.index("def _run_mission")
    end = src.index("def _compute_path_length")
    mission_src = src[start:end]
    assert "fsm.execute(land_at_end=False)" in mission_src
    assert "fsm._run_navigate_phase()" in mission_src
    assert "should_ground_land" in mission_src
    assert "client.land()" in mission_src


def test_task_mode_keeps_default_execute_land():
    src = _RUNNER.read_text(encoding="utf-8")
    start = src.index("def _run_task")
    end = src.index("def _write_trajectory_csv")
    task_src = src[start:end]
    assert "fsm.execute()" in task_src
    assert "land_at_end=False" not in task_src
