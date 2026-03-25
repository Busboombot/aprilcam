"""Tests for stream_tags and stop_stream MCP tools."""

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
    stop_stream,
    stream_tags,
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


class TestStreamTags:
    def test_stream_tags_start_stop(self):
        """Start a stream, verify it's running, stop it."""
        camera_id = _open_fake_camera()
        start_res = _parse(_run(stream_tags(camera_id)))
        assert start_res["status"] == "started"
        assert start_res["stream_id"] == camera_id
        assert "operations" in start_res

        # Verify entry exists in detection_registry
        assert camera_id in detection_registry

        stop_res = _parse(_run(stop_stream(camera_id)))
        assert stop_res["status"] == "stopped"
        assert stop_res["stream_id"] == camera_id
        assert camera_id not in detection_registry

    def test_stream_tags_default_operations(self):
        """Verify default operations is ["detect_tags"]."""
        camera_id = _open_fake_camera()
        result = _parse(_run(stream_tags(camera_id)))
        assert result["operations"] == ["detect_tags"]

        # Also verify stored in the registry entry
        entry = detection_registry[camera_id]
        assert entry.operations == ["detect_tags"]

        _run(stop_stream(camera_id))

    def test_stream_tags_custom_operations(self):
        """Verify custom operations are stored."""
        camera_id = _open_fake_camera()
        ops = ["detect_tags", "detect_lines", "detect_circles"]
        result = _parse(_run(stream_tags(camera_id, operations=ops)))
        assert result["operations"] == ops

        entry = detection_registry[camera_id]
        assert entry.operations == ops

        _run(stop_stream(camera_id))

    def test_stream_tags_get_tags_works(self):
        """Start stream, verify get_tags returns data."""
        camera_id = _open_fake_camera()
        _run(stream_tags(camera_id))
        time.sleep(0.5)

        result = _parse(_run(get_tags(camera_id)))
        assert "source_id" in result
        assert "tags" in result

        _run(stop_stream(camera_id))

    def test_stream_tags_get_tag_history_works(self):
        """Start stream, verify get_tag_history returns data."""
        camera_id = _open_fake_camera()
        _run(stream_tags(camera_id))
        time.sleep(0.5)

        result = _parse(_run(get_tag_history(camera_id)))
        assert "source_id" in result
        assert "frames" in result

        _run(stop_stream(camera_id))

    def test_stream_tags_duplicate(self):
        """Starting a stream twice on the same source returns an error."""
        camera_id = _open_fake_camera()
        _run(stream_tags(camera_id))
        result = _parse(_run(stream_tags(camera_id)))
        assert "error" in result
        _run(stop_stream(camera_id))


class TestStopStream:
    def test_stop_stream_unknown(self):
        """Verify error on unknown source_id."""
        result = _parse(_run(stop_stream("nonexistent-id")))
        assert "error" in result
        assert "nonexistent-id" in result["error"]

    def test_stop_stream_cleans_up(self):
        """Stopping a stream removes it from the detection registry."""
        camera_id = _open_fake_camera()
        _run(stream_tags(camera_id))
        assert camera_id in detection_registry
        _run(stop_stream(camera_id))
        assert camera_id not in detection_registry
