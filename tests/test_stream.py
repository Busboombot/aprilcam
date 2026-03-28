"""Tests for the detect_tags generator API (aprilcam.stream)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aprilcam.stream import detect_tags


def test_detect_tags_import():
    """Verify detect_tags is importable from the top-level package."""
    from aprilcam import detect_tags as dt
    assert callable(dt)


def test_detect_tags_with_mock_camera(monkeypatch):
    """Mock cv.VideoCapture to return synthetic frames and verify yields."""
    # Create a simple 480x640 BGR frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640  # CAP_PROP_FRAME_WIDTH / HEIGHT
    # Return 3 frames then stop
    mock_cap.read.side_effect = [
        (True, frame.copy()),
        (True, frame.copy()),
        (True, frame.copy()),
        (False, None),
    ]

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        gen = detect_tags(camera=0, homography=None, proc_width=0)
        results = list(gen)

    assert len(results) == 3
    # Each result should be a list (possibly empty if no tags in blank frame)
    for r in results:
        assert isinstance(r, list)

    # Camera should have been released
    mock_cap.release.assert_called()


def test_detect_tags_cleanup(monkeypatch):
    """Verify camera is released when generator is closed early."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640
    # Infinite frames
    mock_cap.read.return_value = (True, frame)

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        gen = detect_tags(camera=0, homography=None, proc_width=0)
        # Consume one frame then close
        first = next(gen)
        assert isinstance(first, list)
        gen.close()

    mock_cap.release.assert_called()


def test_detect_tags_camera_pattern(monkeypatch):
    """Verify string camera argument resolves via list_cameras."""
    from aprilcam.camutil import CameraInfo

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640
    mock_cap.read.side_effect = [(True, frame), (False, None)]

    fake_cams = [CameraInfo(index=2, name="Brio 501 (AVFOUNDATION)")]
    monkeypatch.setattr("aprilcam.stream.list_cameras", lambda **kw: fake_cams)

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap) as mock_vc:
        results = list(detect_tags(camera="brio", homography=None))

    # Should have opened camera index 2 (matched by pattern)
    mock_vc.assert_called_once_with(2)
    assert len(results) == 1


def test_detect_tags_camera_not_found(monkeypatch):
    """Verify ValueError when no camera matches pattern."""
    monkeypatch.setattr("aprilcam.stream.list_cameras", lambda **kw: [])

    with pytest.raises(ValueError, match="No camera matching"):
        gen = detect_tags(camera="nonexistent", homography=None)
        next(gen)


def test_detect_tags_auto_homography(monkeypatch, tmp_path):
    """Verify auto homography discovery and loading."""
    import json

    # Create a fake homography file
    H = np.eye(3).tolist()
    hfile = tmp_path / "homography-test-cam-640x480.json"
    hfile.write_text(json.dumps({"homography": H}))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    def mock_get(prop):
        if prop == 3:  # CAP_PROP_FRAME_WIDTH
            return 640
        if prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return 480
        return 0

    mock_cap.get.side_effect = mock_get
    mock_cap.read.side_effect = [(True, frame), (False, None)]

    monkeypatch.setattr(
        "aprilcam.stream.get_device_name",
        lambda idx: "test cam",
    )
    monkeypatch.setattr(
        "aprilcam.stream.discover_homography",
        lambda name, w, h, d: hfile,
    )

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        results = list(detect_tags(camera=0, homography="auto", data_dir=str(tmp_path)))

    assert len(results) == 1


def test_all_exports():
    """Verify all names in __all__ are importable from aprilcam."""
    import aprilcam

    for name in aprilcam.__all__:
        assert hasattr(aprilcam, name), f"{name} listed in __all__ but not importable"
