import math

import numpy as np
import pytest

from airsim_benchmark.controllers.base_controller import DroneState
from airsim_benchmark.controllers.oracle_controller import OracleController


def _quat_yaw(yaw_deg: float):
    r = math.radians(yaw_deg)
    return (math.cos(r / 2), 0.0, 0.0, math.sin(r / 2))


def _state(x, y, z, yaw):
    return DroneState(
        position=(x, y, z),
        velocity=(0, 0, 0),
        orientation=_quat_yaw(yaw),
        image=None,
    )


def test_search_yaws_toward_landmark_behind():
    ctl = OracleController()
    ctl.reset({
        "instruction": "Fly toward the red car",
        "intent": "search_then_approach",
        "start": [0, 0, -10],
        "landmark_position": [50, 0, -1],
        "landmark_radius": 8.0,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 5, "max_altitude": 20},
    })
    ctl.get_action(_state(0, 0, -10, 180.0))
    assert ctl.last_action_vec[2] + ctl.last_action_vec[3] >= 10.0


def test_arrived_emits_stop_vec():
    ctl = OracleController()
    ctl.reset({
        "instruction": "Fly toward the red car",
        "intent": "search_then_approach",
        "start": [50, 0, -10],
        "landmark_position": [50, 0, -1],
        "landmark_radius": 8.0,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 5, "max_altitude": 20},
    })
    ctl.get_action(_state(50, 0, -10, 0.0))
    assert ctl.last_action_vec[0] == 1.0


def test_land_surface_does_not_descend_until_overhead():
    """8 m approach radius is not overhead; descending there hits house walls."""
    ctl = OracleController()
    ctl.reset({
        "instruction": "Land on the rooftop",
        "intent": "land_on_surface",
        "start": [0, 0, -10],
        "landmark_position": [7, 0, -3],
        "landmark_radius": 8.0,
        "surface_z": -3.0,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 4, "max_altitude": 20},
    })
    action = ctl.get_action(_state(0, 0, -10, 0.0))
    assert action.target_position[2] == pytest.approx(-10.0, abs=0.05)
    assert ctl.last_action_vec[5] == 0.0
    assert ctl.last_action_vec[1] > 1.0


def test_land_surface_descends_when_overhead():
    ctl = OracleController()
    ctl.reset({
        "instruction": "Land on the rooftop",
        "intent": "land_on_surface",
        "start": [55, 15, -16],
        "landmark_position": [55, 15, -8],
        "landmark_radius": 6.0,
        "surface_z": -8.0,
        "target_alt_ned": -16.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 4, "max_altitude": 20},
    })
    action = ctl.get_action(_state(55, 15, -16, 0.0))
    assert action.target_position[2] > -16.0  # NED: larger z = lower
    assert ctl.last_action_vec[5] == pytest.approx(2.0, abs=0.1)
    assert ctl.last_action_vec[4] == 0.0


def test_land_ground_descends():
    ctl = OracleController()
    ctl.reset({
        "instruction": "Land on the ground",
        "intent": "land_ground",
        "start": [0, 0, -10],
        "landmark_position": None,
        "landmark_radius": 8.0,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 5, "max_altitude": 20},
    })
    action = ctl.get_action(_state(0, 0, -10, 0.0))
    # NED: +z is down. From -10, hop at most 2 m toward -1.5 → around -8.
    assert action.target_position[2] > -10.0
    assert action.target_position[2] == pytest.approx(-8.0, abs=0.6)
    assert ctl.last_action_vec[5] == pytest.approx(2.0, abs=0.1)
    assert ctl.last_action_vec[4] == 0.0


def test_last_in_fov_false_when_behind():
    ctl = OracleController()
    ctl.reset({
        "instruction": "Fly toward the red car",
        "intent": "search_then_approach",
        "start": [0, 0, -10],
        "landmark_position": [50, 0, -1],
        "landmark_radius": 8.0,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 5, "max_altitude": 20},
    })
    ctl.get_action(_state(0, 0, -10, 180.0))
    assert ctl.last_in_fov is False


def test_ego_turn_left_commands_positive_heading_change():
    ctl = OracleController()
    ctl.reset({
        "instruction": "Turn left",
        "intent": "ego_turn_left",
        "start": [0, 0, -10],
        "landmark_position": None,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "start_yaw": 0.0,
        "constraints": {"min_altitude": 5, "max_altitude": 20},
    })
    ctl.get_action(_state(0, 0, -10, 0.0))
    assert ctl.last_action_vec[2] > 0.0
    assert ctl.last_action_vec[3] == 0.0


def test_vln_norm_q99_aliases_action_space():
    from airsim_benchmark.controllers.oracle_controller import VLN_NORM_Q99
    from airsim_benchmark.core.action_space import VLN_Q99

    assert np.array_equal(VLN_NORM_Q99, VLN_Q99)
    assert VLN_NORM_Q99[5] == 2.0


def _approach_cfg(**overrides):
    cfg = {
        "instruction": "Fly toward the red car",
        "intent": "search_then_approach",
        "start": [50, 0, -10],
        "landmark_position": [50, 0, -1],
        "landmark_radius": 8.0,
        "target_alt_ned": -10.0,
        "max_hops": 20,
        "constraints": {"min_altitude": 5, "max_altitude": 20},
    }
    cfg.update(overrides)
    return cfg


def test_stop_streak_false_after_one_true_after_two():
    ctl = OracleController()
    ctl.reset(_approach_cfg())
    arrived = _state(50, 0, -10, 0.0)
    assert ctl.is_goal_reached(arrived) is False
    ctl.get_action(arrived)
    assert ctl.last_action_vec[0] == 1.0
    assert ctl.is_goal_reached(arrived) is False
    ctl.get_action(arrived)
    assert ctl.last_action_vec[0] == 1.0
    assert ctl.is_goal_reached(arrived) is True


def test_move_resets_stop_streak():
    ctl = OracleController()
    ctl.reset(_approach_cfg())
    arrived = _state(50, 0, -10, 0.0)
    far = _state(0, 0, -10, 0.0)
    ctl.get_action(arrived)
    assert ctl.is_goal_reached(arrived) is False
    ctl.get_action(far)
    assert ctl.last_action_vec[0] != 1.0
    assert ctl.is_goal_reached(far) is False
    ctl.get_action(arrived)
    assert ctl.last_action_vec[0] == 1.0
    assert ctl.is_goal_reached(arrived) is False


def test_hop_budget_terminates():
    ctl = OracleController()
    ctl.reset(_approach_cfg(start=[0, 0, -10], max_hops=2))
    far = _state(0, 0, -10, 0.0)
    ctl.get_action(far)
    assert ctl.is_goal_reached(far) is False
    ctl.get_action(far)
    assert ctl.is_goal_reached(far) is True

