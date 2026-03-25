"""Tests verifying the detection-tool wrapper refactor routes through the frame registry.

Each refactored tool (detect_lines, detect_circles, detect_contours,
detect_qr_codes) now creates a transient FrameEntry, processes via the
existing processing function, records results on the entry, and releases
the entry. These tests confirm:

1. Backward compatibility -- same response format as before.
2. Frame tracking -- frame_registry.add() is called during tool execution.
"""

import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from aprilcam.mcp_server import (
    detect_circles,
    detect_contours,
    detect_lines,
    detect_qr_codes,
    frame_registry,
    registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in that returns a fixed frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def read(self):
        return True, self._frame.copy()

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Ensure registries and frame state are clean before and after each test."""
    frame_registry.clear()
    yield
    frame_registry.clear()
    for cid in list(registry._cameras):
        try:
            registry.close(cid)
        except Exception:
            pass


@pytest.fixture()
def camera_with_shapes() -> str:
    """Register a fake camera with a synthetic image containing lines, circles, and rectangles."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img, (50, 240), (590, 240), (255, 255, 255), 2)
    cv2.line(img, (320, 50), (320, 430), (255, 255, 255), 2)
    cv2.circle(img, (320, 240), 80, (255, 255, 255), 2)
    cv2.rectangle(img, (400, 50), (550, 150), (255, 255, 255), -1)
    cap = FakeCapture(img)
    handle = registry.open(cap)
    return handle


@pytest.fixture()
def blank_camera() -> str:
    """Register a fake camera with a blank (black) image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = FakeCapture(img)
    handle = registry.open(cap)
    return handle


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_lines_backward_compat(camera_with_shapes):
    """detect_lines returns same format: {source_id, lines}."""
    result = await detect_lines(camera_with_shapes, threshold=50, min_length=30, max_gap=10)
    data = json.loads(result[0].text)
    assert "source_id" in data
    assert data["source_id"] == camera_with_shapes
    assert "lines" in data
    assert isinstance(data["lines"], list)


@pytest.mark.asyncio
async def test_detect_lines_empty_backward_compat(blank_camera):
    """detect_lines on blank image returns empty lines list."""
    result = await detect_lines(blank_camera)
    data = json.loads(result[0].text)
    assert data["lines"] == []
    assert data["source_id"] == blank_camera


@pytest.mark.asyncio
async def test_detect_circles_backward_compat():
    """detect_circles returns same format: {source_id, circles}."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 100, (255, 255, 255), 3)
    cap = FakeCapture(img)
    cam_id = registry.open(cap)

    result = await detect_circles(cam_id, param1=100.0, param2=20.0)
    data = json.loads(result[0].text)
    assert "source_id" in data
    assert data["source_id"] == cam_id
    assert "circles" in data
    assert isinstance(data["circles"], list)


@pytest.mark.asyncio
async def test_detect_contours_backward_compat(camera_with_shapes):
    """detect_contours returns same format: {source_id, contours}."""
    result = await detect_contours(camera_with_shapes, min_area=100.0)
    data = json.loads(result[0].text)
    assert "source_id" in data
    assert data["source_id"] == camera_with_shapes
    assert "contours" in data
    assert isinstance(data["contours"], list)
    for contour in data["contours"]:
        assert "area" in contour
        assert "bbox" in contour


@pytest.mark.asyncio
async def test_detect_qr_codes_backward_compat(camera_with_shapes):
    """detect_qr_codes returns same format: {source_id, qr_codes}."""
    result = await detect_qr_codes(camera_with_shapes)
    data = json.loads(result[0].text)
    assert "source_id" in data
    assert data["source_id"] == camera_with_shapes
    assert "qr_codes" in data
    assert isinstance(data["qr_codes"], list)


# ---------------------------------------------------------------------------
# Frame tracking tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrapper_creates_transient_frame_detect_lines(camera_with_shapes):
    """Verify frame_registry.add() is called during detect_lines execution."""
    original_add = frame_registry.add
    add_called = []

    def tracking_add(raw, source, timestamp=None):
        entry = original_add(raw, source, timestamp)
        add_called.append(entry.frame_id)
        return entry

    with patch.object(frame_registry, "add", side_effect=tracking_add):
        await detect_lines(camera_with_shapes)

    # add was called exactly once
    assert len(add_called) == 1
    # The transient frame was released (no longer in registry)
    assert len(frame_registry) == 0


@pytest.mark.asyncio
async def test_wrapper_creates_transient_frame_detect_circles():
    """Verify frame_registry.add() is called during detect_circles execution."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 100, (255, 255, 255), 3)
    cap = FakeCapture(img)
    cam_id = registry.open(cap)

    original_add = frame_registry.add
    add_called = []

    def tracking_add(raw, source, timestamp=None):
        entry = original_add(raw, source, timestamp)
        add_called.append(entry.frame_id)
        return entry

    with patch.object(frame_registry, "add", side_effect=tracking_add):
        await detect_circles(cam_id, param1=100.0, param2=20.0)

    assert len(add_called) == 1
    assert len(frame_registry) == 0


@pytest.mark.asyncio
async def test_wrapper_creates_transient_frame_detect_contours(camera_with_shapes):
    """Verify frame_registry.add() is called during detect_contours execution."""
    original_add = frame_registry.add
    add_called = []

    def tracking_add(raw, source, timestamp=None):
        entry = original_add(raw, source, timestamp)
        add_called.append(entry.frame_id)
        return entry

    with patch.object(frame_registry, "add", side_effect=tracking_add):
        await detect_contours(camera_with_shapes)

    assert len(add_called) == 1
    assert len(frame_registry) == 0


@pytest.mark.asyncio
async def test_wrapper_creates_transient_frame_detect_qr(camera_with_shapes):
    """Verify frame_registry.add() is called during detect_qr_codes execution."""
    original_add = frame_registry.add
    add_called = []

    def tracking_add(raw, source, timestamp=None):
        entry = original_add(raw, source, timestamp)
        add_called.append(entry.frame_id)
        return entry

    with patch.object(frame_registry, "add", side_effect=tracking_add):
        await detect_qr_codes(camera_with_shapes)

    assert len(add_called) == 1
    assert len(frame_registry) == 0


@pytest.mark.asyncio
async def test_error_does_not_leak_frame():
    """Verify that if source_id is invalid, no frame leaks into the registry."""
    result = await detect_lines("nonexistent-id")
    data = json.loads(result[0].text)
    assert "error" in data
    assert len(frame_registry) == 0
