"""Unit tests for aprilcam.models — AprilTag family field."""

from __future__ import annotations

import numpy as np

from aprilcam.models import AprilTag


def _make_corners() -> np.ndarray:
    """Return a simple 4x2 corner array for testing."""
    return np.array(
        [[90.0, 190.0], [110.0, 190.0], [110.0, 210.0], [90.0, 210.0]],
        dtype=np.float32,
    )


def test_apriltag_family_default():
    """from_corners() without explicit family defaults to '36h11'."""
    tag = AprilTag.from_corners(tag_id=1, corners_px=_make_corners())
    assert tag.family == "36h11"


def test_apriltag_from_corners_with_family():
    """from_corners() with family='25h9' stores the family correctly."""
    tag = AprilTag.from_corners(
        tag_id=2, corners_px=_make_corners(), family="25h9"
    )
    assert tag.family == "25h9"


def test_apriltag_clone_preserves_family():
    """clone() preserves the family field."""
    tag = AprilTag.from_corners(
        tag_id=3, corners_px=_make_corners(), family="16h5"
    )
    cloned = tag.clone()
    assert cloned.family == "16h5"
    assert cloned.id == tag.id
    assert cloned.family == tag.family
