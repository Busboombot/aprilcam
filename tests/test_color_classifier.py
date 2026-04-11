"""Tests for ColorClassifier HSV color detection."""

import numpy as np
import pytest

from aprilcam.color_classifier import ColorClassifier
from aprilcam.objects import ObjectRecord


def _make_image(width=200, height=200):
    """Return a black BGR image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_square(img, cx, cy, size, bgr):
    """Draw a filled square at (cx, cy) with given BGR color."""
    half = size // 2
    img[cy - half : cy + half, cx - half : cx + half] = bgr


# --- classify tests ---


def test_classify_red_patch():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (0, 0, 255))
    cc = ColorClassifier()
    results = cc.classify(img)
    assert len(results) >= 1
    colors = {r.color for r in results}
    assert "red" in colors


def test_classify_green_patch():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (0, 255, 0))
    cc = ColorClassifier()
    results = cc.classify(img)
    assert len(results) >= 1
    colors = {r.color for r in results}
    assert "green" in colors


def test_classify_blue_patch():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (255, 0, 0))
    cc = ColorClassifier()
    results = cc.classify(img)
    assert len(results) >= 1
    colors = {r.color for r in results}
    assert "blue" in colors


def test_classify_yellow_patch():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (0, 255, 255))
    cc = ColorClassifier()
    results = cc.classify(img)
    assert len(results) >= 1
    colors = {r.color for r in results}
    assert "yellow" in colors


def test_classify_multiple_colors():
    img = _make_image(400, 200)
    _draw_square(img, 80, 100, 50, (0, 0, 255))  # red
    _draw_square(img, 300, 100, 50, (255, 0, 0))  # blue
    cc = ColorClassifier()
    results = cc.classify(img)
    colors = {r.color for r in results}
    assert "red" in colors
    assert "blue" in colors


def test_classify_returns_object_records():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (0, 0, 255))
    cc = ColorClassifier()
    results = cc.classify(img)
    assert len(results) >= 1
    for r in results:
        assert isinstance(r, ObjectRecord)


def test_classify_with_homography():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (0, 255, 0))
    H = np.eye(3, dtype=np.float64)
    cc = ColorClassifier()
    results = cc.classify(img, homography=H)
    assert len(results) >= 1
    for r in results:
        assert r.world_xy is not None
        assert isinstance(r.world_xy, tuple)
        assert len(r.world_xy) == 2


def test_classify_area_filter():
    """A tiny 5x5 dot should be below the default min_area=200."""
    img = _make_image()
    _draw_square(img, 100, 100, 5, (0, 0, 255))
    cc = ColorClassifier()
    results = cc.classify(img)
    assert len(results) == 0


def test_custom_color_ranges():
    """Pass only red ranges — blue patch should not be detected."""
    img = _make_image(400, 200)
    _draw_square(img, 80, 100, 50, (0, 0, 255))  # red
    _draw_square(img, 300, 100, 50, (255, 0, 0))  # blue
    red_only = {
        "red": [((0, 50, 50), (12, 255, 255)), ((165, 50, 50), (180, 255, 255))]
    }
    cc = ColorClassifier(color_ranges=red_only)
    results = cc.classify(img)
    colors = {r.color for r in results}
    assert "red" in colors
    assert "blue" not in colors


# --- classify_at_point tests ---


def test_classify_at_point_red():
    img = _make_image()
    _draw_square(img, 100, 100, 50, (0, 0, 255))
    cc = ColorClassifier()
    assert cc.classify_at_point(img, 100, 100) == "red"


def test_classify_at_point_unknown():
    """A gray patch should not match any color range."""
    img = _make_image()
    _draw_square(img, 100, 100, 50, (128, 128, 128))
    cc = ColorClassifier()
    assert cc.classify_at_point(img, 100, 100) == "unknown"


def test_classify_at_point_out_of_bounds():
    """Querying near corner with radius should not crash."""
    img = _make_image()
    cc = ColorClassifier()
    result = cc.classify_at_point(img, 0, 0, radius=20)
    assert isinstance(result, str)
