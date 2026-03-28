"""Tests for detect_tags object detection integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aprilcam.objects import FrameResult, ObjectRecord
from aprilcam.stream import detect_tags


def _make_mock_cap(frames):
    """Create a mock VideoCapture that yields *frames* then stops."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640
    side = [(True, f.copy()) for f in frames] + [(False, None)]
    mock_cap.read.side_effect = side
    return mock_cap


def test_detect_tags_yields_frame_result():
    """Default detect_tags yields FrameResult (backward-compat wrapper)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([frame, frame])

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        results = list(detect_tags(camera=0, homography=None, proc_width=0))

    assert len(results) == 2
    for r in results:
        assert isinstance(r, FrameResult)
        # Tags list (likely empty on a blank frame)
        assert isinstance(r.tags, list)
        # Objects list should be empty when detect_objects=False
        assert r.objects == []


def test_detect_tags_with_objects():
    """detect_tags with detect_objects=True detects squares in the frame."""
    # Create a black frame with a white square (should be detected).
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a filled white square at (300, 200) with size 30x30.
    frame[200:230, 300:330, :] = 255

    mock_cap = _make_mock_cap([frame])

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        results = list(
            detect_tags(
                camera=0,
                homography=None,
                proc_width=0,
                detect_objects=True,
            )
        )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, FrameResult)
    # The white square should be detected as an object.
    assert len(r.objects) > 0
    for obj in r.objects:
        assert isinstance(obj, ObjectRecord)


def test_backward_compat_iteration():
    """for tags in detect_tags(...): for tag in tags: -- still works."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap([frame, frame, frame])

    with patch("aprilcam.stream.cv.VideoCapture", return_value=mock_cap):
        gen = detect_tags(camera=0, homography=None, proc_width=0)
        count = 0
        for tags in gen:
            # FrameResult iterates over tags (list of TagRecord)
            for _tag in tags:
                pass  # would run if tags were detected
            count += 1

    assert count == 3


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
