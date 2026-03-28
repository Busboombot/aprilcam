"""Tests for detection pipeline optimization (ticket #002, sprint 011).

Verifies:
- ArUco detector caching on Playfield
- Corner detection throttling via corner_detect_interval
- detect_apriltags accepts an optional gray parameter
"""

from __future__ import annotations

import numpy as np
import cv2 as cv
import pytest

from aprilcam.playfield import Playfield
from aprilcam.aprilcam import AprilCam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bgr_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_gray_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a blank single-channel grayscale frame."""
    return np.zeros((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Playfield ArUco detector caching
# ---------------------------------------------------------------------------

class TestArucoCaching:
    """The ArUco detector should be built once and reused."""

    def test_detector_cached_after_init(self):
        pf = Playfield()
        assert pf._aruco_detector is not None, (
            "_aruco_detector should be created in __post_init__"
        )

    def test_detector_is_aruco_detector(self):
        pf = Playfield()
        assert isinstance(pf._aruco_detector, cv.aruco.ArucoDetector)

    def test_detector_stable_across_updates(self):
        pf = Playfield(corner_detect_interval=1)
        frame = _make_bgr_frame()
        pf.update(frame)
        det_after_first = pf._aruco_detector
        pf.update(frame)
        det_after_second = pf._aruco_detector
        assert det_after_first is det_after_second, (
            "Detector instance must be reused, not rebuilt"
        )


# ---------------------------------------------------------------------------
# Corner detection throttling
# ---------------------------------------------------------------------------

class TestCornerThrottling:
    """Corner re-detection should be throttled by corner_detect_interval."""

    def test_default_interval_is_30(self):
        pf = Playfield()
        assert pf.corner_detect_interval == 30

    def test_custom_interval(self):
        pf = Playfield(corner_detect_interval=10)
        assert pf.corner_detect_interval == 10

    def test_frame_counter_increments(self):
        pf = Playfield(corner_detect_interval=5)
        frame = _make_bgr_frame()
        assert pf._corner_frame_count == 0
        pf.update(frame)
        assert pf._corner_frame_count == 1
        pf.update(frame)
        assert pf._corner_frame_count == 2

    def test_throttle_skips_detection_when_poly_exists(self):
        """When a polygon already exists, non-interval frames should skip detection."""
        pf = Playfield(corner_detect_interval=5)
        # Pre-set a polygon so throttling can kick in
        pf._poly = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        frame = _make_bgr_frame()

        # Frame 1 (count=1): should detect (count-1=0, 0%5==0)
        pf.update(frame)
        assert pf._corner_frame_count == 1

        # Frames 2-5 should be skipped (polygon unchanged from blank frame)
        poly_before = pf._poly.copy()
        for _ in range(4):
            pf.update(frame)
        # Polygon should remain unchanged since detection was skipped
        np.testing.assert_array_equal(pf._poly, poly_before)
        assert pf._corner_frame_count == 5

    def test_no_throttle_without_polygon(self):
        """When no polygon exists yet, detection should run every frame."""
        pf = Playfield(corner_detect_interval=100)
        frame = _make_bgr_frame()
        # Without a polygon, every update should attempt detection
        for i in range(5):
            pf.update(frame)
            assert pf._corner_frame_count == i + 1

    def test_interval_1_detects_every_frame(self):
        """With interval=1, detection should never be throttled."""
        pf = Playfield(corner_detect_interval=1)
        pf._poly = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        frame = _make_bgr_frame()
        for _ in range(5):
            pf.update(frame)
        assert pf._corner_frame_count == 5


# ---------------------------------------------------------------------------
# detect_apriltags accepts gray parameter
# ---------------------------------------------------------------------------

class TestDetectApriltagsGrayParam:
    """AprilCam.detect_apriltags should accept an optional gray parameter."""

    def _make_cam(self) -> AprilCam:
        return AprilCam(
            index=0,
            backend=None,
            speed_alpha=0.3,
            family="36h11",
            proc_width=640,
            headless=True,
            cap=None,
        )

    def test_accepts_gray_none(self):
        cam = self._make_cam()
        frame = _make_bgr_frame()
        # Should work with gray=None (default)
        result = cam.detect_apriltags(frame, gray=None)
        assert isinstance(result, list)

    def test_accepts_gray_image(self):
        cam = self._make_cam()
        frame = _make_bgr_frame()
        gray = _make_gray_frame()
        # Should work with a pre-computed grayscale
        result = cam.detect_apriltags(frame, gray=gray)
        assert isinstance(result, list)

    def test_gray_param_produces_same_result(self):
        """Passing the correct gray should give same results as auto-conversion."""
        cam = self._make_cam()
        frame = _make_bgr_frame()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        result_auto = cam.detect_apriltags(frame)
        result_gray = cam.detect_apriltags(frame, gray=gray)
        assert len(result_auto) == len(result_gray)


# ---------------------------------------------------------------------------
# Playfield.update accepts gray parameter
# ---------------------------------------------------------------------------

class TestPlayfieldUpdateGrayParam:
    """Playfield.update should accept an optional gray parameter."""

    def test_update_with_gray_none(self):
        pf = Playfield(corner_detect_interval=1)
        frame = _make_bgr_frame()
        pf.update(frame, gray=None)
        assert pf._corner_frame_count == 1

    def test_update_with_gray_image(self):
        pf = Playfield(corner_detect_interval=1)
        frame = _make_bgr_frame()
        gray = _make_gray_frame()
        pf.update(frame, gray=gray)
        assert pf._corner_frame_count == 1
