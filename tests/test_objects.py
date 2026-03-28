"""Tests for ObjectRecord, FrameResult, and SquareDetector."""

import dataclasses

import cv2 as cv
import numpy as np
import pytest

from aprilcam.objects import FrameResult, ObjectRecord, SquareDetector


# ---------------------------------------------------------------------------
# ObjectRecord
# ---------------------------------------------------------------------------

def test_object_record_fields():
    rec = ObjectRecord(
        center_px=(10.0, 20.0),
        bbox=(5, 15, 10, 10),
        area_px=100.0,
    )
    assert rec.center_px == (10.0, 20.0)
    assert rec.bbox == (5, 15, 10, 10)
    assert rec.area_px == 100.0
    assert rec.world_xy is None
    assert rec.color == "unknown"
    assert rec.object_type == "cube"
    assert rec.confidence == 1.0


def test_object_record_immutable():
    rec = ObjectRecord(center_px=(1.0, 2.0), bbox=(0, 0, 5, 5), area_px=25.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.color = "red"


# ---------------------------------------------------------------------------
# FrameResult
# ---------------------------------------------------------------------------

def test_frame_result_iter_yields_tags():
    tags = ["tag_a", "tag_b", "tag_c"]
    fr = FrameResult(tags)
    assert list(fr) == tags


def test_frame_result_len_returns_tag_count():
    tags = [1, 2, 3, 4]
    fr = FrameResult(tags)
    assert len(fr) == 4


def test_frame_result_getitem():
    tags = ["x", "y", "z"]
    fr = FrameResult(tags)
    assert fr[0] == "x"
    assert fr[-1] == "z"
    assert fr[1] == "y"


# ---------------------------------------------------------------------------
# SquareDetector helpers
# ---------------------------------------------------------------------------

def _blank(h=480, w=640):
    """Return a black grayscale image."""
    return np.zeros((h, w), dtype=np.uint8)


def _draw_rect(img, cx, cy, half_w, half_h, color=255):
    """Draw a filled rectangle centered at (cx, cy)."""
    cv.rectangle(
        img,
        (cx - half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        color,
        -1,
    )


# ---------------------------------------------------------------------------
# SquareDetector tests
# ---------------------------------------------------------------------------

def test_square_detector_synthetic():
    img = _blank()
    # Three 40x40 white squares, well separated.
    _draw_rect(img, 100, 100, 20, 20)
    _draw_rect(img, 300, 200, 20, 20)
    _draw_rect(img, 500, 350, 20, 20)

    det = SquareDetector(min_area=200, max_area=5000)
    recs = det.detect(img)
    assert len(recs) == 3


def test_square_detector_excludes_tags():
    img = _blank()
    _draw_rect(img, 200, 200, 20, 20)  # 40x40 square

    # Tag polygon that covers the square center.
    tag_poly = np.array([[180, 180], [220, 180], [220, 220], [180, 220]], dtype=np.float32)
    det = SquareDetector(min_area=200, max_area=5000)
    recs = det.detect(img, tag_corners=[tag_poly])
    assert len(recs) == 0


def test_square_detector_exclusion_radius():
    img = _blank()
    _draw_rect(img, 100, 100, 20, 20)

    det = SquareDetector(min_area=200, max_area=5000)
    recs = det.detect(img, exclusion_point=(100, 100), exclusion_radius=60)
    assert len(recs) == 0


def test_square_detector_aspect_ratio_filter():
    img = _blank()
    # 100x30 rectangle — aspect ratio ~3.3, should be rejected.
    _draw_rect(img, 300, 200, 50, 15)

    det = SquareDetector(min_area=200, max_area=5000)
    recs = det.detect(img)
    assert len(recs) == 0


def test_square_detector_area_bounds():
    img = _blank()
    # 5x5 square = 25 px area — too small for default min_area=200.
    _draw_rect(img, 100, 100, 2, 2)
    # 200x200 square = 40000 px area — too big for default max_area=5000.
    _draw_rect(img, 400, 300, 100, 100)

    det = SquareDetector(min_area=200, max_area=5000)
    recs = det.detect(img)
    assert len(recs) == 0


def test_square_detector_homography():
    img = _blank()
    _draw_rect(img, 300, 200, 20, 20)

    det = SquareDetector(min_area=200, max_area=5000)
    recs = det.detect(img, homography=np.eye(3))
    assert len(recs) >= 1
    rec = recs[0]
    assert rec.world_xy is not None
    # With identity homography, world coords should match pixel center.
    assert abs(rec.world_xy[0] - rec.center_px[0]) < 1.0
    assert abs(rec.world_xy[1] - rec.center_px[1]) < 1.0
