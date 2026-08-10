"""Tests for the Heading Correlation (HC) metric."""

import math

import pytest

from airsim_benchmark.evaluation.heading_correlation import (
    angular_difference,
    bearing_to_target,
    heading_correlation,
)


# ── angular_difference ────────────────────────────────────────────────────


class TestAngularDifference:
    def test_same_angle(self):
        assert angular_difference(45, 45) == 0.0

    def test_opposite(self):
        assert angular_difference(0, 180) == 180.0

    def test_wrap_around(self):
        assert angular_difference(350, 10) == pytest.approx(20.0)

    def test_negative_angles(self):
        assert angular_difference(-10, 10) == pytest.approx(20.0)

    def test_symmetric(self):
        assert angular_difference(30, 90) == angular_difference(90, 30)

    def test_full_circle(self):
        assert angular_difference(0, 360) == pytest.approx(0.0)


# ── bearing_to_target ─────────────────────────────────────────────────────


class TestBearingToTarget:
    def test_north(self):
        assert bearing_to_target((0, 0), (10, 0)) == pytest.approx(0.0)

    def test_east(self):
        assert bearing_to_target((0, 0), (0, 10)) == pytest.approx(90.0)

    def test_south(self):
        assert bearing_to_target((0, 0), (-10, 0)) == pytest.approx(180.0)

    def test_west(self):
        assert bearing_to_target((0, 0), (0, -10)) == pytest.approx(-90.0)

    def test_northeast(self):
        assert bearing_to_target((0, 0), (10, 10)) == pytest.approx(45.0)


# ── heading_correlation ───────────────────────────────────────────────────


class TestHeadingCorrelation:
    def test_perfect_navigation(self):
        """Drone heading matches bearing to target exactly → HC ≈ 0."""
        target = (50, 0)
        positions = [(0, 0), (10, 0), (20, 0)]
        headings = [0.0, 0.0, 0.0]
        assert heading_correlation(positions, headings, target) == pytest.approx(0.0, abs=1e-6)

    def test_random_headings(self):
        """Equally spread headings average out to ~90°."""
        target = (50, 0)
        positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        headings = [0.0, 90.0, 180.0, 270.0]
        assert heading_correlation(positions, headings, target) == pytest.approx(90.0, abs=1.0)

    def test_wrong_direction(self):
        """Drone flies away from target → HC ≈ 180."""
        target = (50, 0)
        positions = [(0, 0), (10, 0), (20, 0)]
        headings = [180.0, 180.0, 180.0]
        assert heading_correlation(positions, headings, target) == pytest.approx(180.0, abs=1.0)

    def test_empty_input(self):
        assert math.isnan(heading_correlation([], [], (50, 0)))

    def test_skips_near_target(self):
        """Hops within 1 m of target are excluded."""
        target = (0, 0)
        positions = [(0.5, 0.0), (0.0, 0.3)]
        headings = [45.0, 90.0]
        assert math.isnan(heading_correlation(positions, headings, target))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            heading_correlation([(0, 0)], [0.0, 90.0], (10, 0))
