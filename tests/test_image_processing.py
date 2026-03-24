"""Tests for resolve_source and format_image_output utilities."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from mcp.types import ImageContent, TextContent

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
