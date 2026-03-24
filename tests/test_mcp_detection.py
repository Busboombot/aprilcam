"""Tests for start_detection, stop_detection, get_tags, get_tag_history MCP tools."""

import asyncio
import json
import time
from pathlib import Path

import cv2
import pytest

from aprilcam.mcp_server import (
    detection_registry,
    get_tag_history,
    get_tags,
    playfield_registry,
    registry,
    start_detection,
    stop_detection,
)

TEST_DATA = Path(__file__).parent / "data"


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _parse(result):
    """Extract JSON dict from MCP tool result."""
    return json.loads(result[0].text)


class FakeCapture:
    """Mock camera that reads from a static image with a small delay."""

    def __init__(self, image_path: str):
        self._img = cv2.imread(str(image_path))
        assert self._img is not None, f"Failed to load {image_path}"

    def read(self):
        time.sleep(0.01)
        return True, self._img.copy()

    def release(self):
        pass


@pytest.fixture(autouse=True)
def clean_registries():
    """Reset all registries before/after each test."""
    detection_registry.clear()
    playfield_registry._playfields.clear()
    registry._cameras.clear()
    yield
    # Stop any lingering detection loops
    for entry in list(detection_registry.values()):
        try:
            entry.loop.stop()
        except Exception:
            pass
    detection_registry.clear()
    playfield_registry._playfields.clear()
    registry._cameras.clear()


def _open_fake_camera() -> str:
    """Register a FakeCapture and return its camera_id."""
    cap = FakeCapture(str(TEST_DATA / "playfield_cam3_moved.jpg"))
    return registry.open(cap)


class TestStartDetection:
    def test_start_detection_with_camera(self):
        camera_id = _open_fake_camera()
        result = _parse(_run(start_detection(camera_id)))
        assert result["status"] == "started"
        assert result["source_id"] == camera_id
        # Cleanup
        _run(stop_detection(camera_id))

    def test_start_detection_invalid_source(self):
        result = _parse(_run(start_detection("nonexistent-id")))
        assert "error" in result

    def test_start_detection_duplicate(self):
        camera_id = _open_fake_camera()
        _run(start_detection(camera_id))
        result = _parse(_run(start_detection(camera_id)))
        assert "error" in result
        # Cleanup
        _run(stop_detection(camera_id))


class TestStopDetection:
    def test_stop_detection(self):
        camera_id = _open_fake_camera()
        _run(start_detection(camera_id))
        result = _parse(_run(stop_detection(camera_id)))
        assert result["status"] == "stopped"
        assert result["source_id"] == camera_id

    def test_stop_detection_invalid(self):
        result = _parse(_run(stop_detection("nonexistent-id")))
        assert "error" in result


class TestGetTags:
    def test_get_tags_active_loop(self):
        camera_id = _open_fake_camera()
        _run(start_detection(camera_id))
        time.sleep(0.5)
        result = _parse(_run(get_tags(camera_id)))
        assert "source_id" in result
        assert "tags" in result
        _run(stop_detection(camera_id))

    def test_get_tags_no_loop(self):
        result = _parse(_run(get_tags("nonexistent-id")))
        assert "error" in result


class TestGetTagHistory:
    def test_get_tag_history_default(self):
        camera_id = _open_fake_camera()
        _run(start_detection(camera_id))
        time.sleep(0.5)
        result = _parse(_run(get_tag_history(camera_id)))
        assert "frames" in result
        assert "source_id" in result
        _run(stop_detection(camera_id))

    def test_get_tag_history_custom_n(self):
        camera_id = _open_fake_camera()
        _run(start_detection(camera_id))
        time.sleep(0.5)
        result = _parse(_run(get_tag_history(camera_id, num_frames=3)))
        assert "frames" in result
        assert len(result["frames"]) <= 3
        _run(stop_detection(camera_id))

    def test_get_tag_history_no_loop(self):
        result = _parse(_run(get_tag_history("nonexistent-id")))
        assert "error" in result
