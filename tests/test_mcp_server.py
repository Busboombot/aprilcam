"""Tests for the MCP server: CameraRegistry and tool handlers."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy
import pytest

from aprilcam.mcp_server import CameraRegistry, registry

# ---------------------------------------------------------------------------
# CameraRegistry tests
# ---------------------------------------------------------------------------


class TestCameraRegistry:
    def setup_method(self):
        self.reg = CameraRegistry()
        self.cap = MagicMock()

    def test_open_returns_uuid_string(self):
        handle = self.reg.open(self.cap)
        assert isinstance(handle, str)
        # UUID4 format: 8-4-4-4-12 hex chars
        parts = handle.split("-")
        assert len(parts) == 5

    def test_open_stores_capture(self):
        handle = self.reg.open(self.cap)
        assert self.reg.get(handle) is self.cap

    def test_get_returns_stored_capture(self):
        handle = self.reg.open(self.cap)
        assert self.reg.get(handle) is self.cap

    def test_get_raises_keyerror_for_invalid_handle(self):
        with pytest.raises(KeyError):
            self.reg.get("nonexistent-handle")

    def test_close_calls_release_and_removes(self):
        handle = self.reg.open(self.cap)
        self.reg.close(handle)
        self.cap.release.assert_called_once()
        with pytest.raises(KeyError):
            self.reg.get(handle)

    def test_close_raises_keyerror_for_invalid_handle(self):
        with pytest.raises(KeyError):
            self.reg.close("nonexistent-handle")

    def test_close_raises_keyerror_for_already_closed(self):
        handle = self.reg.open(self.cap)
        self.reg.close(handle)
        with pytest.raises(KeyError):
            self.reg.close(handle)

    def test_close_all_releases_all_and_clears(self):
        caps = [MagicMock() for _ in range(3)]
        for c in caps:
            self.reg.open(c)
        self.reg.close_all()
        for c in caps:
            c.release.assert_called_once()
        assert self.reg.list_open() == []

    def test_list_open_returns_correct_handles(self):
        h1 = self.reg.open(MagicMock())
        h2 = self.reg.open(MagicMock())
        handles = self.reg.list_open()
        assert set(handles) == {h1, h2}

    def test_multiple_cameras_simultaneously(self):
        caps = [MagicMock() for _ in range(5)]
        handles = [self.reg.open(c) for c in caps]
        # All unique
        assert len(set(handles)) == 5
        # All retrievable
        for h, c in zip(handles, caps):
            assert self.reg.get(h) is c
        # Close one, others still accessible
        self.reg.close(handles[2])
        assert len(self.reg.list_open()) == 4
        for i, (h, c) in enumerate(zip(handles, caps)):
            if i == 2:
                continue
            assert self.reg.get(h) is c


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


class TestListCameras:
    def test_returns_structured_json(self):
        cam = MagicMock()
        cam.index = 0
        cam.name = "FaceTime"
        cam.backend = "AVFOUNDATION"

        with patch("aprilcam.mcp_server.list_cameras") as mock_tool:
            # We need to patch the inner import instead
            pass

        # Patch the inner import used inside the tool function
        from aprilcam.mcp_server import list_cameras as tool_fn

        mock_cam = MagicMock()
        mock_cam.index = 0
        mock_cam.name = "FaceTime"
        mock_cam.backend = "AVFOUNDATION"

        with patch("aprilcam.camutil.list_cameras", return_value=[mock_cam]):
            result = _run(tool_fn())

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["index"] == 0
        assert data[0]["name"] == "FaceTime"
        assert data[0]["backend"] == "AVFOUNDATION"

    def test_returns_empty_array_when_no_cameras(self):
        from aprilcam.mcp_server import list_cameras as tool_fn

        with patch("aprilcam.camutil.list_cameras", return_value=[]):
            result = _run(tool_fn())

        data = json.loads(result[0].text)
        assert data == []


class TestOpenCamera:
    def test_with_index_returns_handle(self):
        from aprilcam.mcp_server import open_camera, registry as reg

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())

        with patch("cv2.VideoCapture", return_value=mock_cap):
            result = _run(open_camera(index=0))

        data = json.loads(result[0].text)
        assert "camera_id" in data
        # Clean up
        handle = data["camera_id"]
        try:
            reg.close(handle)
        except KeyError:
            pass

    def test_returns_error_when_camera_fails_to_open(self):
        from aprilcam.mcp_server import open_camera

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            result = _run(open_camera(index=99))

        data = json.loads(result[0].text)
        assert "error" in data
        mock_cap.release.assert_called_once()


class TestCaptureFrame:
    def _make_registry_with_cap(self):
        """Create a mock capture in the module-level registry and return (handle, cap)."""
        from aprilcam.mcp_server import registry as reg

        cap = MagicMock()
        frame = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        cap.read.return_value = (True, frame)
        handle = reg.open(cap)
        return handle, cap

    def test_returns_image_content_with_base64(self):
        from aprilcam.mcp_server import capture_frame

        handle, cap = self._make_registry_with_cap()
        try:
            result = _run(capture_frame(camera_id=handle))
            # Default format is base64, returns ImageContent
            assert len(result) == 1
            assert result[0].type == "image"
            assert result[0].mimeType == "image/jpeg"
            assert len(result[0].data) > 0
        finally:
            from aprilcam.mcp_server import registry as reg
            try:
                reg.close(handle)
            except KeyError:
                pass

    def test_with_format_file_returns_path(self):
        from aprilcam.mcp_server import capture_frame

        handle, cap = self._make_registry_with_cap()
        try:
            result = _run(capture_frame(camera_id=handle, format="file"))
            assert len(result) == 1
            data = json.loads(result[0].text)
            assert "path" in data
            assert data["path"].endswith(".jpg")
        finally:
            from aprilcam.mcp_server import registry as reg
            try:
                reg.close(handle)
            except KeyError:
                pass

    def test_returns_error_for_invalid_camera_id(self):
        from aprilcam.mcp_server import capture_frame

        result = _run(capture_frame(camera_id="invalid-id"))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "invalid-id" in data["error"]


class TestCloseCamera:
    def test_returns_status_closed(self):
        from aprilcam.mcp_server import close_camera, registry as reg

        cap = MagicMock()
        handle = reg.open(cap)
        result = _run(close_camera(camera_id=handle))
        data = json.loads(result[0].text)
        assert data == {"status": "closed"}
        cap.release.assert_called_once()

    def test_returns_error_for_invalid_handle(self):
        from aprilcam.mcp_server import close_camera

        result = _run(close_camera(camera_id="bogus-handle"))
        data = json.loads(result[0].text)
        assert "error" in data
