import math

import pytest

from airsim_benchmark.collection.geometry import (
    wrap_yaw_deg,
    clip_heading_error,
    bearing_to_xy,
    landmark_in_fov,
)


def test_wrap_yaw():
    assert wrap_yaw_deg(190) == pytest.approx(-170)
    assert wrap_yaw_deg(-190) == pytest.approx(170)


def test_clip_heading_error():
    assert clip_heading_error(40, 15) == 15
    assert clip_heading_error(-40, 15) == -15
    assert clip_heading_error(5, 15) == 5


def test_bearing_due_east_from_origin_facing_north():
    # Facing +X (north, yaw=0). Landmark at +Y (east) → +90°.
    b = bearing_to_xy((0.0, 0.0), 0.0, (0.0, 50.0))
    assert b == pytest.approx(90.0, abs=1.0)


def test_landmark_in_fov_ahead():
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, (40.0, 0.0, -10.0)) is True


def test_landmark_behind_not_in_fov():
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, (-40.0, 0.0, -10.0)) is False


def test_landmark_wide_angle_not_in_fov():
    # ~90° off heading
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, (5.0, 40.0, -10.0)) is False


def test_bearing_facing_east_landmark_north():
    # Facing +Y (east, yaw=90). Landmark at +X (north) → -90°.
    b = bearing_to_xy((0.0, 0.0), 90.0, (50.0, 0.0))
    assert b == pytest.approx(-90.0, abs=1.0)


def test_landmark_in_fov_on_heading():
    yaw = 30.0
    r = 40.0
    yaw_rad = math.radians(yaw)
    landmark = (r * math.cos(yaw_rad), r * math.sin(yaw_rad), -10.0)
    assert landmark_in_fov((0.0, 0.0, -10.0), yaw, landmark) is True


def test_landmark_50deg_off_heading_not_in_fov():
    yaw = 30.0
    r = 40.0
    off_rad = math.radians(yaw + 50.0)
    landmark = (r * math.cos(off_rad), r * math.sin(off_rad), -10.0)
    assert landmark_in_fov((0.0, 0.0, -10.0), yaw, landmark) is False


def test_landmark_just_inside_half_fov():
    r = 40.0
    ang = math.radians(39.0)
    landmark = (r * math.cos(ang), r * math.sin(ang), -10.0)
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, landmark) is True


def test_landmark_just_outside_half_fov():
    r = 40.0
    ang = math.radians(41.0)
    landmark = (r * math.cos(ang), r * math.sin(ang), -10.0)
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, landmark) is False


def test_landmark_beyond_max_range_not_in_fov():
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, (151.0, 0.0, -10.0)) is False


def test_landmark_too_close_ahead_not_in_fov():
    assert landmark_in_fov((0.0, 0.0, -10.0), 0.0, (0.5, 0.0, -10.0)) is False
