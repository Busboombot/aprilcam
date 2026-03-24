"""Tests for image processing functions and MCP helpers."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from mcp.types import ImageContent, TextContent

from aprilcam.image_processing import (
    process_apply_transform,
    process_detect_circles,
    process_detect_contours,
    process_detect_lines,
    process_detect_motion,
    process_detect_qr_codes,
)
from aprilcam.mcp_server import (
    PlayfieldEntry,
    format_image_output,
    playfield_registry,
    registry,
    resolve_source,
)
from aprilcam.playfield import Playfield


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


@pytest.fixture(autouse=True)
def _clean_registries():
    """Ensure both registries are empty before and after each test."""
    yield
    # Tear-down: remove anything the test added
    for pid in list(playfield_registry._playfields):
        playfield_registry.remove(pid)
    for cid in list(registry._cameras):
        try:
            registry.close(cid)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# resolve_source tests
# ---------------------------------------------------------------------------


class TestResolveSourceCamera:
    def test_returns_frame_from_camera(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = FakeCapture(frame)
        cam_id = registry.open(cap)

        result = resolve_source(cam_id)
        assert result.shape == (480, 640, 3)


class TestResolveSourcePlayfield:
    def test_returns_deskewed_frame(self):
        # Create a 640x480 white frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cap = FakeCapture(frame)
        cam_id = registry.open(cap)

        # Playfield with a known polygon — deskew output size depends on polygon
        poly = np.array(
            [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32
        )
        pf = Playfield(polygon=poly)
        pf_id = "pf_test"
        entry = PlayfieldEntry(
            playfield_id=pf_id,
            camera_id=cam_id,
            playfield=pf,
        )
        playfield_registry.register(entry)

        result = resolve_source(pf_id)
        # Deskewed shape is derived from polygon extents (400x300)
        assert result.shape != (480, 640, 3), "Frame should be deskewed to a different size"
        assert result.shape[0] == 300  # height = 400-100
        assert result.shape[1] == 400  # width = 500-100


class TestResolveSourceInvalid:
    def test_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown source_id"):
            resolve_source("no-such-source")


# ---------------------------------------------------------------------------
# format_image_output tests
# ---------------------------------------------------------------------------


class TestFormatImageBase64:
    def test_returns_image_content(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = format_image_output(frame)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].type == "image"
        assert result[0].mimeType == "image/jpeg"
        assert len(result[0].data) > 0


class TestFormatImageFile:
    def test_returns_text_content_with_path(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = format_image_output(frame, format="file")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        data = json.loads(result[0].text)
        assert "path" in data
        assert data["path"].endswith(".jpg")
        assert os.path.isfile(data["path"])

        # Clean up temp file
        os.unlink(data["path"])


# ---------------------------------------------------------------------------
# process_detect_lines tests
# ---------------------------------------------------------------------------


class TestDetectLines:
    def test_detect_lines_synthetic(self):
        """White lines on a black image should be detected."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.line(img, (10, 100), (190, 100), (255, 255, 255), 2)
        cv2.line(img, (100, 10), (100, 190), (255, 255, 255), 2)
        lines = process_detect_lines(img, threshold=30, min_length=30, max_gap=10)
        assert len(lines) >= 1
        for l in lines:
            assert "x1" in l and "y1" in l and "x2" in l and "y2" in l

    def test_detect_lines_empty(self):
        """Blank image should yield no lines."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        lines = process_detect_lines(img)
        assert lines == []


# ---------------------------------------------------------------------------
# process_detect_circles tests
# ---------------------------------------------------------------------------


class TestDetectCircles:
    def test_detect_circles_synthetic(self):
        """White circle on a black image should be detected."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 40, (255, 255, 255), 2)
        circles = process_detect_circles(img, param1=100.0, param2=20.0)
        assert len(circles) >= 1
        c = circles[0]
        assert "center" in c and "radius" in c
        assert "x" in c["center"] and "y" in c["center"]

    def test_detect_circles_empty(self):
        """Blank image should yield no circles."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        circles = process_detect_circles(img)
        assert circles == []


# ---------------------------------------------------------------------------
# process_detect_contours tests
# ---------------------------------------------------------------------------


class TestDetectContours:
    def test_detect_contours_synthetic(self):
        """White rectangle on a black image should produce a contour."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (170, 170), (255, 255, 255), -1)
        contours = process_detect_contours(img, min_area=100.0)
        assert len(contours) >= 1
        c = contours[0]
        assert "area" in c
        assert "bbox" in c
        assert c["area"] > 100

    def test_detect_contours_min_area(self):
        """Small contour should be filtered out by min_area."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (90, 90), (95, 95), (255, 255, 255), -1)
        contours = process_detect_contours(img, min_area=1000.0)
        assert contours == []


# ---------------------------------------------------------------------------
# process_detect_motion tests
# ---------------------------------------------------------------------------


class TestDetectMotion:
    def test_detect_motion_no_prev(self):
        """None prev_frame should return empty list."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        regions = process_detect_motion(img, None)
        assert regions == []

    def test_detect_motion_with_change(self):
        """Two different frames should produce motion regions."""
        frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
        frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (20, 20), (180, 180), (255, 255, 255), -1)
        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        regions = process_detect_motion(frame2, prev_gray)
        assert len(regions) >= 1
        assert "bbox" in regions[0]
        assert "area" in regions[0]

    def test_detect_motion_no_change(self):
        """Same frame twice should produce no motion regions."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions = process_detect_motion(img, gray)
        assert regions == []


# ---------------------------------------------------------------------------
# process_detect_qr_codes tests
# ---------------------------------------------------------------------------


class TestDetectQRCodes:
    def test_detect_qr_codes_empty(self):
        """Image with no QR codes should return empty list."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        codes = process_detect_qr_codes(img)
        assert codes == []


# ---------------------------------------------------------------------------
# process_apply_transform tests
# ---------------------------------------------------------------------------


class TestApplyTransform:
    def test_rotate(self):
        """Rotated image should keep the same dimensions."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = process_apply_transform(img, "rotate", {"angle": 90})
        assert result.shape[:2] == (100, 200)

    def test_scale(self):
        """Scaled image should have proportionally adjusted dimensions."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = process_apply_transform(img, "scale", {"factor": 0.5})
        assert result.shape[0] == 50
        assert result.shape[1] == 100

    def test_threshold(self):
        """Threshold output should be binary (0 or 255 only)."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = process_apply_transform(img, "threshold", {"value": 127})
        # Convert to grayscale to check values
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        unique = set(np.unique(gray))
        assert unique <= {0, 255}

    def test_canny(self):
        """Canny transform should produce a 3-channel output."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = process_apply_transform(img, "canny")
        assert result.shape == (100, 100, 3)

    def test_blur(self):
        """Blur should produce output with same shape."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = process_apply_transform(img, "blur", {"kernel_size": 5})
        assert result.shape == (100, 100, 3)

    def test_unknown_raises(self):
        """Unknown operation should raise ValueError."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown operation"):
            process_apply_transform(img, "nonexistent")
