"""Integration tests for image processing MCP tools."""

import json

import cv2
import numpy as np
import pytest

from aprilcam.mcp_server import (
    _motion_prev_frames,
    apply_transform,
    crop_region,
    detect_circles,
    detect_contours,
    detect_lines,
    detect_motion,
    detect_qr_codes,
    frame_registry,
    get_frame,
    playfield_registry,
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


class AlternatingCapture:
    """Fake capture that alternates between a blank and a white-rectangle frame."""

    def __init__(self):
        self._count = 0

    def read(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        if self._count % 2 == 1:
            cv2.rectangle(img, (200, 200), (400, 400), (255, 255, 255), -1)
        self._count += 1
        return True, img

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Ensure registries and motion state are empty before and after each test."""
    yield
    _motion_prev_frames.clear()
    frame_registry.clear()
    for pid in list(playfield_registry._playfields):
        playfield_registry.remove(pid)
    for cid in list(registry._cameras):
        try:
            registry.close(cid)
        except Exception:
            pass


@pytest.fixture()
def camera_with_shapes() -> str:
    """Register a fake camera with a synthetic image containing lines, circles, and rectangles."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a horizontal line
    cv2.line(img, (50, 240), (590, 240), (255, 255, 255), 2)
    # Draw a vertical line
    cv2.line(img, (320, 50), (320, 430), (255, 255, 255), 2)
    # Draw a circle
    cv2.circle(img, (320, 240), 80, (255, 255, 255), 2)
    # Draw a filled rectangle
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
# get_frame tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_frame_base64(camera_with_shapes):
    result = await get_frame(camera_with_shapes)
    assert len(result) > 0
    assert result[0].type == "image"


@pytest.mark.asyncio
async def test_get_frame_file(camera_with_shapes):
    result = await get_frame(camera_with_shapes, format="file")
    assert len(result) > 0
    data = json.loads(result[0].text)
    assert "path" in data


@pytest.mark.asyncio
async def test_get_frame_invalid_source():
    result = await get_frame("nonexistent-camera-id")
    data = json.loads(result[0].text)
    assert "error" in data


# ---------------------------------------------------------------------------
# crop_region tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crop_region_success(camera_with_shapes):
    result = await crop_region(camera_with_shapes, x=100, y=100, w=200, h=200)
    assert len(result) > 0
    assert result[0].type == "image"


@pytest.mark.asyncio
async def test_crop_region_out_of_bounds(camera_with_shapes):
    result = await crop_region(camera_with_shapes, x=9999, y=9999, w=100, h=100)
    data = json.loads(result[0].text)
    assert "error" in data


# ---------------------------------------------------------------------------
# detect_lines tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_lines_finds_lines(camera_with_shapes):
    result = await detect_lines(camera_with_shapes, threshold=50, min_length=30, max_gap=10)
    data = json.loads(result[0].text)
    assert "lines" in data
    assert len(data["lines"]) > 0


@pytest.mark.asyncio
async def test_detect_lines_empty(blank_camera):
    result = await detect_lines(blank_camera)
    data = json.loads(result[0].text)
    assert "lines" in data
    assert data["lines"] == []


# ---------------------------------------------------------------------------
# detect_circles tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_circles_finds_circles():
    # Create a dedicated image with a thick, well-defined circle for reliable detection
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 100, (255, 255, 255), 3)
    cap = FakeCapture(img)
    cam_id = registry.open(cap)
    result = await detect_circles(cam_id, param1=100.0, param2=20.0)
    data = json.loads(result[0].text)
    assert "circles" in data
    assert len(data["circles"]) > 0


# ---------------------------------------------------------------------------
# detect_contours tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_contours_finds_shapes(camera_with_shapes):
    result = await detect_contours(camera_with_shapes, min_area=100.0)
    data = json.loads(result[0].text)
    assert "contours" in data
    assert len(data["contours"]) > 0
    for contour in data["contours"]:
        assert "area" in contour
        assert "bbox" in contour


# ---------------------------------------------------------------------------
# detect_motion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_motion_baseline(camera_with_shapes):
    result = await detect_motion(camera_with_shapes)
    data = json.loads(result[0].text)
    assert data["is_baseline"] is True


@pytest.mark.asyncio
async def test_detect_motion_with_change():
    cap = AlternatingCapture()
    cam_id = registry.open(cap)
    # First call: baseline
    await detect_motion(cam_id)
    # Second call: different frame, should detect motion
    result = await detect_motion(cam_id)
    data = json.loads(result[0].text)
    assert data["is_baseline"] is False
    assert len(data["motion_regions"]) > 0


# ---------------------------------------------------------------------------
# detect_qr_codes tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_qr_codes_empty(camera_with_shapes):
    result = await detect_qr_codes(camera_with_shapes)
    data = json.loads(result[0].text)
    assert "qr_codes" in data
    assert data["qr_codes"] == []


# ---------------------------------------------------------------------------
# apply_transform tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_transform_canny(camera_with_shapes):
    result = await apply_transform(
        camera_with_shapes, "canny", params='{"low": 50, "high": 150}'
    )
    assert len(result) > 0
    assert result[0].type == "image"


@pytest.mark.asyncio
async def test_apply_transform_blur(camera_with_shapes):
    result = await apply_transform(camera_with_shapes, "blur")
    assert len(result) > 0
    assert result[0].type == "image"


@pytest.mark.asyncio
async def test_apply_transform_unknown(camera_with_shapes):
    result = await apply_transform(camera_with_shapes, "foo")
    data = json.loads(result[0].text)
    assert "error" in data
