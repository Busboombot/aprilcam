import numpy as np
import pytest
import cv2 as cv
from aprilcam.playfield import Playfield


def test_constructor_with_polygon():
    poly = np.array([[100, 50], [500, 50], [500, 400], [100, 400]], dtype=np.float32)
    pf = Playfield(polygon=poly)
    result = pf.get_polygon()
    assert result is not None
    np.testing.assert_array_equal(result, poly)


def test_constructor_without_polygon():
    pf = Playfield()
    assert pf.get_polygon() is None


def test_update_noop_with_injected_polygon():
    poly = np.array([[100, 50], [500, 50], [500, 400], [100, 400]], dtype=np.float32)
    pf = Playfield(polygon=poly)
    # Create a dummy frame - update should be a no-op
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    pf.update(dummy)
    result = pf.get_polygon()
    np.testing.assert_array_equal(result, poly)


def test_order_poly_canonical():
    """_order_poly produces UL, UR, LR, LL regardless of input ID mapping."""
    pf = Playfield()
    # IDs scrambled: 0=LR, 1=UL, 2=UR, 3=LL
    corners = {
        0: (500.0, 400.0),  # LR
        1: (100.0, 50.0),   # UL
        2: (500.0, 50.0),   # UR
        3: (100.0, 400.0),  # LL
    }
    result = pf._order_poly(corners)
    assert result is not None
    # Expected: UL, UR, LR, LL
    expected = np.array([[100, 50], [500, 50], [500, 400], [100, 400]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1.0)


def test_deskew_masks_outside_polygon():
    """Deskew should mask out content outside the playfield polygon."""
    # Create image filled with red (outside content)
    img = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)
    # Paint green inside the polygon region
    poly = np.array([[100, 50], [500, 50], [500, 400], [100, 400]], dtype=np.int32)
    cv.fillConvexPoly(img, poly, (0, 255, 0))
    pf = Playfield(polygon=poly.astype(np.float32))
    result = pf.deskew(img)
    # The output should contain NO red pixels (outside content was masked)
    red_channel = result[:, :, 2]  # BGR -> R
    green_channel = result[:, :, 1]  # BGR -> G
    # Check that no pixel is pure red (0,0,255)
    is_red = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 255)
    assert not np.any(is_red), "Red (outside) pixels found in deskewed output"
    # Most of the output should be green (inside content)
    is_green = (result[:, :, 0] == 0) & (result[:, :, 1] == 255) & (result[:, :, 2] == 0)
    green_ratio = np.sum(is_green) / (result.shape[0] * result.shape[1])
    assert green_ratio > 0.9, f"Expected >90% green pixels, got {green_ratio:.1%}"


def test_deskew_output_dimensions():
    """Deskew produces correct output dimensions from injected polygon."""
    poly = np.array([[100, 50], [500, 50], [500, 400], [100, 400]], dtype=np.float32)
    pf = Playfield(polygon=poly)
    # Create a test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pf.deskew(img)
    # Expected: width=400 (500-100), height=350 (400-50)
    assert result.shape[1] == 400  # width
    assert result.shape[0] == 350  # height
