"""Tests for object detection components and detect_tags backward compat."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aprilcam.objects import FrameResult, ObjectRecord, SquareDetector
from aprilcam.stream import detect_tags


def _make_mock_cap(frames):
    """Create a mock VideoCapture that yields *frames* then stops."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640
    side = [(True, f.copy()) for f in frames] + [(False, None)]
    mock_cap.read.side_effect = side
    return mock_cap


def test_detect_tags_yields_tag_lists():
    """detect_tags yields plain list[TagRecord] per frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([frame, frame])

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        results = list(detect_tags(camera=0, homography=None, proc_width=0))

    assert len(results) == 2
    for r in results:
        assert isinstance(r, list)


def test_detect_tags_iteration():
    """for tags in detect_tags(...): for tag in tags: -- works."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([frame, frame, frame])

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        gen = detect_tags(camera=0, homography=None, proc_width=0)
        count = 0
        for tags in gen:
            for _tag in tags:
                pass
            count += 1

    assert count == 3


def test_square_detector_on_synthetic():
    """SquareDetector finds a white square on a black image."""
    frame = np.zeros((480, 640), dtype=np.uint8)
    frame[200:230, 300:330] = 255  # 30x30 white square

    det = SquareDetector(min_area=50, max_area=2000, threshold=100)
    objects = det.detect(frame)
    assert len(objects) > 0
    for obj in objects:
        assert isinstance(obj, ObjectRecord)


def test_frame_result_len_and_index():
    """FrameResult supports len() and indexing over tags."""
    fr = FrameResult(tags=["a", "b", "c"], objects=["x"])
    assert len(fr) == 3
    assert fr[0] == "a"
    assert fr[2] == "c"
    assert list(fr) == ["a", "b", "c"]


def test_frame_result_attributes():
    """FrameResult exposes timestamp, frame_index, and objects."""
    fr = FrameResult(tags=[], objects=["obj"], timestamp=1.0, frame_index=5)
    assert fr.timestamp == 1.0
    assert fr.frame_index == 5
    assert fr.objects == ["obj"]
