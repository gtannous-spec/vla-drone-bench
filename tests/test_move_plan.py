"""Unit tests for hop execute planning (no AirSim)."""

import math
from pathlib import Path

import pytest

from airsim_benchmark.core.move_plan import (
    is_stuck_hop,
    plan_move_to,
    plan_vertical_velocity,
    should_call_airsim_land,
)


def test_same_pose_is_noop():
    plan = plan_move_to((10.0, 4.0, -10.0), (10.0, 4.0, -10.0), yaw_deg=0.0)
    assert plan["kind"] == "noop"


def test_pure_z_is_vertical():
    plan = plan_move_to((0.0, 0.0, -10.0), (0.0, 0.0, -8.0), yaw_deg=90.0)
    assert plan["kind"] == "vertical"
    assert plan["z"] == -8.0
    assert plan["vz"] == pytest.approx(2.0)
    assert plan["duration"] == pytest.approx(1.0)


def test_vertical_velocity_sign_follows_ned():
    down = plan_vertical_velocity(-10.0, -8.0)
    assert down["vz"] > 0.0
    assert down["duration"] == pytest.approx(1.0)
    up = plan_vertical_velocity(-8.0, -10.0)
    assert up["vz"] < 0.0


def test_airsim_land_only_when_near_ground():
    assert should_call_airsim_land(-1.5) is True
    assert should_call_airsim_land(-10.0) is False


def test_forward_hop_is_xy_max_dof_facing_waypoint():
    heading = 15.0
    fwd = 2.5
    tx = fwd * math.cos(math.radians(heading))
    ty = fwd * math.sin(math.radians(heading))
    plan = plan_move_to((0.0, 0.0, -10.0), (tx, ty, -10.0), yaw_deg=0.0)
    assert plan["kind"] == "xy"
    assert plan["drivetrain"] == "MaxDegreeOfFreedom"
    assert plan["lookahead"] == 0.5
    assert plan["yaw_deg"] == pytest.approx(heading, abs=0.6)


def test_yaw_command_clipped_when_waypoint_is_abeam():
    # +Y is +90° in NED yaw (atan2(y, x)).
    plan = plan_move_to((0.0, 0.0, -10.0), (0.0, 2.5, -10.0), yaw_deg=0.0)
    assert plan["kind"] == "xy"
    assert abs(plan["yaw_deg"]) <= 20.0001


def test_stuck_hop_false_for_planned_stop_and_real_move():
    assert is_stuck_hop((0, 0, -10), (0.01, 0.0, -10), planned_stop=True) is False
    assert is_stuck_hop((0, 0, -10), (2.5, 0.0, -10), planned_stop=False) is False


def test_stuck_hop_true_for_tiny_unplanned_move():
    assert is_stuck_hop((0, 0, -10), (0.05, 0.0, -10), planned_stop=False) is True


def test_collector_aborts_after_three_stuck_hops():
    src = (
        Path(__file__).resolve().parents[1]
        / "airsim_benchmark"
        / "scripts"
        / "collect_trajectories.py"
    ).read_text(encoding="utf-8")
    assert "is_stuck_hop" in src
    assert "stuck_hops >= 3" in src
    assert "should_call_airsim_land" in src
