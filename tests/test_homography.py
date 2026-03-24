"""Tests for aprilcam.homography — calibrate_from_corners and FieldSpec."""

import numpy as np
import pytest

from aprilcam.homography import FieldSpec, calibrate_from_corners


def test_calibrate_identity():
    """Corners at (0,0)-(100,0)-(0,100)-(100,100) with 100x100cm field -> near-identity H."""
    corners = {
        "upper_left": (0.0, 0.0),
        "upper_right": (100.0, 0.0),
        "lower_left": (0.0, 100.0),
        "lower_right": (100.0, 100.0),
    }
    field = FieldSpec(100.0, 100.0, "cm")
    H, _, _ = calibrate_from_corners(corners, field)
    assert H.shape == (3, 3)
    np.testing.assert_allclose(H / H[2, 2], np.eye(3), atol=1e-6)


def test_calibrate_scaled():
    """Known pixel corners with inch measurements -> correct world mapping."""
    corners = {
        "upper_left": (100.0, 50.0),
        "upper_right": (500.0, 50.0),
        "lower_left": (100.0, 400.0),
        "lower_right": (500.0, 400.0),
    }
    field = FieldSpec(40.0, 35.0, "inch")
    H, pixel_pts, world_pts = calibrate_from_corners(corners, field)
    # Verify each corner maps correctly
    for i in range(4):
        pt = np.array([pixel_pts[i][0], pixel_pts[i][1], 1.0])
        mapped = H @ pt
        mapped = mapped[:2] / mapped[2]
        np.testing.assert_allclose(mapped, world_pts[i], atol=0.1)


def test_calibrate_returns_points():
    """calibrate_from_corners returns pixel_pts and world_pts arrays."""
    corners = {
        "upper_left": (10.0, 20.0),
        "upper_right": (310.0, 20.0),
        "lower_left": (10.0, 220.0),
        "lower_right": (310.0, 220.0),
    }
    field = FieldSpec(50.0, 30.0, "cm")
    H, pixel_pts, world_pts = calibrate_from_corners(corners, field)
    assert pixel_pts.shape == (4, 2)
    assert world_pts.shape == (4, 2)
    np.testing.assert_allclose(pixel_pts[0], [10.0, 20.0])
    np.testing.assert_allclose(world_pts[1], [50.0, 0.0])


def test_field_spec_inch_to_cm():
    f = FieldSpec(40.0, 35.0, "inch")
    assert abs(f.width_cm - 101.6) < 0.01
    assert abs(f.height_cm - 88.9) < 0.01


def test_field_spec_cm():
    f = FieldSpec(100.0, 80.0, "cm")
    assert f.width_cm == 100.0
    assert f.height_cm == 80.0
