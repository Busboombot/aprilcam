"""Tests for ObjectFuser and ColorCameraThread."""

import time
from unittest.mock import MagicMock, patch

from aprilcam.vision.objects import ColorCameraThread, ObjectFuser, ObjectRecord


def _make_obj(world_xy=None, color="unknown"):
    """Helper to create an ObjectRecord for testing."""
    return ObjectRecord(
        center_px=(100.0, 200.0),
        bbox=(90, 190, 20, 20),
        area_px=400.0,
        world_xy=world_xy,
        color=color,
    )


def test_fuser_update_and_fuse_basic():
    fuser = ObjectFuser()
    color_obj = _make_obj(world_xy=(10.0, 20.0), color="red")
    fuser.update_colors([color_obj])

    bw_obj = _make_obj(world_xy=(10.0, 20.0), color="unknown")
    result = fuser.fuse([bw_obj])
    assert len(result) == 1
    assert result[0].color == "red"


def test_fuser_match_radius():
    fuser = ObjectFuser(match_radius=5.0)
    color_obj = _make_obj(world_xy=(10.3, 20.2), color="green")
    fuser.update_colors([color_obj])

    bw_obj = _make_obj(world_xy=(10.0, 20.0), color="unknown")
    result = fuser.fuse([bw_obj])
    assert len(result) == 1
    assert result[0].color == "green"


def test_fuser_no_match_outside_radius():
    fuser = ObjectFuser(match_radius=5.0)
    color_obj = _make_obj(world_xy=(20.0, 30.0), color="red")
    fuser.update_colors([color_obj])

    bw_obj = _make_obj(world_xy=(10.0, 20.0), color="unknown")
    result = fuser.fuse([bw_obj])
    assert len(result) == 1
    assert result[0].color == "unknown"


def test_fuser_persistence():
    fuser = ObjectFuser()
    color_obj = _make_obj(world_xy=(10.0, 20.0), color="red")
    fuser.update_colors([color_obj])

    for _ in range(3):
        bw_obj = _make_obj(world_xy=(10.0, 20.0), color="unknown")
        result = fuser.fuse([bw_obj])
        assert result[0].color == "red"


def test_fuser_overwrite():
    fuser = ObjectFuser()
    fuser.update_colors([_make_obj(world_xy=(10.0, 20.0), color="red")])
    fuser.update_colors([_make_obj(world_xy=(10.0, 20.0), color="blue")])

    result = fuser.fuse([_make_obj(world_xy=(10.0, 20.0), color="unknown")])
    assert result[0].color == "blue"


def test_fuser_clear_stale():
    fuser = ObjectFuser()

    with patch("aprilcam.vision.objects.time") as mock_time:
        mock_time.time.return_value = 1000.0
        fuser.update_colors([_make_obj(world_xy=(10.0, 20.0), color="red")])
        assert len(fuser._color_map) == 1

        mock_time.time.return_value = 1010.0
        fuser.clear_stale(max_age_seconds=5.0)
        assert len(fuser._color_map) == 0


def test_fuser_no_world_xy():
    fuser = ObjectFuser()
    fuser.update_colors([_make_obj(world_xy=(10.0, 20.0), color="red")])

    bw_obj = _make_obj(world_xy=None, color="unknown")
    result = fuser.fuse([bw_obj])
    assert len(result) == 1
    assert result[0].color == "unknown"
    assert result[0].world_xy is None


def test_fuser_quantization():
    fuser = ObjectFuser()
    # (10.04, 20.06) quantizes to (10.0, 20.1)
    key1 = ObjectFuser._quantize(10.04, 20.06)
    # (10.0, 20.1) quantizes to (10.0, 20.1)
    key2 = ObjectFuser._quantize(10.0, 20.1)
    assert key1 == key2


def test_color_camera_thread_lifecycle():
    fuser = ObjectFuser()
    classifier = MagicMock()
    classifier.classify.return_value = [
        _make_obj(world_xy=(5.0, 5.0), color="yellow")
    ]

    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, MagicMock())

    with patch("aprilcam.vision.objects.cv.VideoCapture", return_value=mock_cap):
        thread = ColorCameraThread(
            camera_index=0, fuser=fuser, classifier=classifier, fps=100.0
        )
        thread.start()
        # Give the thread time to run at least one iteration
        time.sleep(0.2)
        thread.stop()

    assert classifier.classify.call_count >= 1
    assert len(fuser._color_map) >= 1


def test_color_camera_thread_is_daemon():
    fuser = ObjectFuser()
    classifier = MagicMock()
    classifier.classify.return_value = []

    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    with patch("aprilcam.vision.objects.cv.VideoCapture", return_value=mock_cap):
        thread = ColorCameraThread(
            camera_index=0, fuser=fuser, classifier=classifier, fps=100.0
        )
        thread.start()
        assert thread._thread.daemon is True
        thread.stop()
