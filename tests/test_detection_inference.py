import numpy as np
import pytest

from airsim_benchmark.core.detection_inference import (
    Detection,
    ObjectDetector,
    bbox_to_heading_offset,
    bbox_to_distance_estimate,
    check_spatial_proximity,
)


# --- Detection dataclass ---

class TestDetection:
    def test_fields(self):
        d = Detection(bbox_xyxy=(100.0, 50.0, 300.0, 250.0), score=0.85, phrase="red car")
        assert d.bbox_xyxy == (100.0, 50.0, 300.0, 250.0)
        assert d.score == 0.85
        assert d.phrase == "red car"

    def test_center_x(self):
        d = Detection(bbox_xyxy=(100.0, 50.0, 300.0, 250.0), score=0.9, phrase="car")
        assert d.center_x == pytest.approx(200.0)

    def test_center_y(self):
        d = Detection(bbox_xyxy=(100.0, 50.0, 300.0, 250.0), score=0.9, phrase="car")
        assert d.center_y == pytest.approx(150.0)

    def test_width(self):
        d = Detection(bbox_xyxy=(100.0, 50.0, 300.0, 250.0), score=0.9, phrase="car")
        assert d.width == pytest.approx(200.0)

    def test_height(self):
        d = Detection(bbox_xyxy=(100.0, 50.0, 300.0, 250.0), score=0.9, phrase="car")
        assert d.height == pytest.approx(200.0)

    def test_area_ratio(self):
        d = Detection(bbox_xyxy=(0.0, 0.0, 320.0, 240.0), score=0.9, phrase="house")
        ratio = d.area_ratio(640, 480)
        assert ratio == pytest.approx(0.25)

    def test_area_ratio_full_image(self):
        d = Detection(bbox_xyxy=(0.0, 0.0, 640.0, 480.0), score=0.9, phrase="sky")
        assert d.area_ratio(640, 480) == pytest.approx(1.0)

    def test_area_ratio_zero_image(self):
        d = Detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), score=0.5, phrase="dot")
        assert d.area_ratio(0, 0) == 0.0


# --- bbox_to_heading_offset ---

class TestBboxToHeadingOffset:
    def test_center_gives_zero(self):
        offset = bbox_to_heading_offset(320.0, 640, 90.0)
        assert offset == pytest.approx(0.0)

    def test_left_edge_gives_negative_half_fov(self):
        offset = bbox_to_heading_offset(0.0, 640, 90.0)
        assert offset == pytest.approx(-45.0)

    def test_right_edge_gives_positive_half_fov(self):
        offset = bbox_to_heading_offset(640.0, 640, 90.0)
        assert offset == pytest.approx(45.0)

    def test_quarter_left(self):
        offset = bbox_to_heading_offset(160.0, 640, 90.0)
        assert offset == pytest.approx(-22.5)

    def test_quarter_right(self):
        offset = bbox_to_heading_offset(480.0, 640, 90.0)
        assert offset == pytest.approx(22.5)

    def test_different_fov(self):
        offset = bbox_to_heading_offset(0.0, 640, 120.0)
        assert offset == pytest.approx(-60.0)


# --- bbox_to_distance_estimate ---

class TestBboxToDistanceEstimate:
    def test_large_box_gives_close(self):
        dist = bbox_to_distance_estimate(0.10)
        assert dist == 10.0

    def test_medium_box_gives_medium(self):
        dist = bbox_to_distance_estimate(0.03)
        assert dist == 30.0

    def test_small_box_gives_far(self):
        dist = bbox_to_distance_estimate(0.005)
        assert dist == 60.0

    def test_exactly_close_threshold(self):
        dist = bbox_to_distance_estimate(0.05)
        assert dist == 30.0  # not > threshold, equals it

    def test_above_close_threshold(self):
        dist = bbox_to_distance_estimate(0.051)
        assert dist == 10.0

    def test_custom_thresholds(self):
        dist = bbox_to_distance_estimate(
            0.08, close=5.0, medium=20.0, far=50.0,
            close_threshold=0.10, medium_threshold=0.02
        )
        assert dist == 20.0


# --- ObjectDetector._postprocess_detections ---

class TestPostprocessDetections:
    def _make_detector(self):
        """Create a detector without loading the actual model."""
        det = object.__new__(ObjectDetector)
        det._box_threshold = 0.25
        det._text_threshold = 0.20
        return det

    def test_returns_correct_detections(self):
        det = self._make_detector()
        boxes = np.array([
            [10.0, 20.0, 100.0, 80.0],
            [200.0, 100.0, 400.0, 300.0],
        ])
        scores = np.array([0.80, 0.55])
        phrases = ["red car", "mailbox"]

        results = det._postprocess_detections(boxes, scores, phrases)
        assert len(results) == 2
        assert results[0].phrase == "red car"
        assert results[0].score == pytest.approx(0.80)
        assert results[1].phrase == "mailbox"
        assert results[1].score == pytest.approx(0.55)

    def test_sorted_by_score_descending(self):
        det = self._make_detector()
        boxes = np.array([
            [0.0, 0.0, 50.0, 50.0],
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
        ])
        scores = np.array([0.30, 0.90, 0.60])
        phrases = ["a", "b", "c"]

        results = det._postprocess_detections(boxes, scores, phrases)
        assert results[0].phrase == "b"
        assert results[1].phrase == "c"
        assert results[2].phrase == "a"

    def test_filters_below_threshold(self):
        det = self._make_detector()
        boxes = np.array([
            [0.0, 0.0, 50.0, 50.0],
            [100.0, 100.0, 200.0, 200.0],
        ])
        scores = np.array([0.80, 0.10])
        phrases = ["car", "noise"]

        results = det._postprocess_detections(boxes, scores, phrases)
        assert len(results) == 1
        assert results[0].phrase == "car"

    def test_empty_input_returns_empty(self):
        det = self._make_detector()
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        phrases = []

        results = det._postprocess_detections(boxes, scores, phrases)
        assert results == []

    def test_all_below_threshold_returns_empty(self):
        det = self._make_detector()
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([0.15])
        phrases = ["faint"]

        results = det._postprocess_detections(boxes, scores, phrases)
        assert results == []

    def test_bbox_coordinates_preserved(self):
        det = self._make_detector()
        boxes = np.array([[12.5, 34.7, 256.3, 189.1]])
        scores = np.array([0.72])
        phrases = ["house"]

        results = det._postprocess_detections(boxes, scores, phrases)
        assert len(results) == 1
        assert results[0].bbox_xyxy == pytest.approx((12.5, 34.7, 256.3, 189.1))


# --- ObjectDetector._group_detections (detect_multi helper) ---

class TestDetectMulti:
    def _make_detector(self):
        det = object.__new__(ObjectDetector)
        det._box_threshold = 0.25
        det._text_threshold = 0.20
        return det

    def test_groups_by_query(self):
        det = self._make_detector()
        results = det._group_detections(
            boxes=np.array([[10, 10, 50, 50], [200, 200, 300, 300]]),
            scores=np.array([0.8, 0.7]),
            labels=["blue truck", "parking lot"],
            queries=["blue truck", "parking lot"],
        )
        assert "blue truck" in results
        assert "parking lot" in results
        assert len(results["blue truck"]) == 1
        assert len(results["parking lot"]) == 1

    def test_missing_query_returns_empty_list(self):
        det = self._make_detector()
        results = det._group_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            scores=np.array([0.8]),
            labels=["blue truck"],
            queries=["blue truck", "parking lot"],
        )
        assert results["parking lot"] == []

    def test_multiple_detections_same_query(self):
        det = self._make_detector()
        results = det._group_detections(
            boxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
            scores=np.array([0.8, 0.6]),
            labels=["car", "car"],
            queries=["car"],
        )
        assert len(results["car"]) == 2

    def test_filters_below_threshold(self):
        det = self._make_detector()
        results = det._group_detections(
            boxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
            scores=np.array([0.8, 0.1]),
            labels=["car", "car"],
            queries=["car"],
        )
        assert len(results["car"]) == 1

    def test_sorted_by_score_descending(self):
        det = self._make_detector()
        results = det._group_detections(
            boxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
            scores=np.array([0.5, 0.9]),
            labels=["car", "car"],
            queries=["car"],
        )
        assert results["car"][0].score > results["car"][1].score


# --- check_spatial_proximity ---

class TestSpatialProximity:
    def test_close_objects(self):
        a = [Detection(bbox_xyxy=(100, 100, 200, 200), score=0.8, phrase="a")]
        b = [Detection(bbox_xyxy=(150, 150, 250, 250), score=0.7, phrase="b")]
        assert check_spatial_proximity(a, b) is True

    def test_far_objects(self):
        a = [Detection(bbox_xyxy=(0, 0, 50, 50), score=0.8, phrase="a")]
        b = [Detection(bbox_xyxy=(590, 430, 640, 480), score=0.7, phrase="b")]
        assert check_spatial_proximity(a, b) is False

    def test_empty_detections(self):
        a = [Detection(bbox_xyxy=(100, 100, 200, 200), score=0.8, phrase="a")]
        assert check_spatial_proximity(a, []) is False
        assert check_spatial_proximity([], a) is False

    def test_same_position(self):
        a = [Detection(bbox_xyxy=(100, 100, 200, 200), score=0.8, phrase="a")]
        b = [Detection(bbox_xyxy=(100, 100, 200, 200), score=0.7, phrase="b")]
        assert check_spatial_proximity(a, b) is True

    def test_uses_highest_scoring_detection(self):
        a = [
            Detection(bbox_xyxy=(0, 0, 10, 10), score=0.3, phrase="a"),
            Detection(bbox_xyxy=(100, 100, 200, 200), score=0.9, phrase="a"),
        ]
        b = [Detection(bbox_xyxy=(150, 150, 250, 250), score=0.7, phrase="b")]
        assert check_spatial_proximity(a, b) is True

    def test_custom_threshold(self):
        a = [Detection(bbox_xyxy=(100, 100, 200, 200), score=0.8, phrase="a")]
        b = [Detection(bbox_xyxy=(300, 300, 400, 400), score=0.7, phrase="b")]
        assert check_spatial_proximity(a, b, max_distance_ratio=0.1) is False
        assert check_spatial_proximity(a, b, max_distance_ratio=0.9) is True
