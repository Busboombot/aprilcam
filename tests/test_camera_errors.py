"""Tests for camera error reporting and diagnostics."""

from __future__ import annotations

from unittest.mock import patch, MagicMock, PropertyMock
import subprocess

import pytest

from aprilcam.errors import (
    CameraError,
    CameraNotFoundError,
    CameraInUseError,
    CameraPermissionError,
)
from aprilcam.camutil import diagnose_camera_failure


# ---------- Exception hierarchy ----------

class TestCameraErrorHierarchy:
    def test_camera_not_found_is_camera_error(self):
        assert issubclass(CameraNotFoundError, CameraError)

    def test_camera_in_use_is_camera_error(self):
        assert issubclass(CameraInUseError, CameraError)

    def test_camera_permission_is_camera_error(self):
        assert issubclass(CameraPermissionError, CameraError)

    def test_all_are_exceptions(self):
        for cls in (CameraError, CameraNotFoundError, CameraInUseError, CameraPermissionError):
            assert issubclass(cls, Exception)


# ---------- CameraInUseError attributes ----------

class TestCameraInUseErrorAttributes:
    def test_has_pid_and_process_name(self):
        err = CameraInUseError("Camera busy", pid=1234, process_name="python3")
        assert err.pid == 1234
        assert err.process_name == "python3"
        assert "Camera busy" in str(err)

    def test_defaults_to_none(self):
        err = CameraInUseError("Camera busy")
        assert err.pid is None
        assert err.process_name is None

    def test_catchable_as_camera_error(self):
        with pytest.raises(CameraError):
            raise CameraInUseError("busy", pid=1, process_name="ffmpeg")


# ---------- diagnose_camera_failure — live (safe) ----------

class TestDiagnoseCameraFailureLive:
    def test_returns_dict_with_expected_keys(self):
        result = diagnose_camera_failure(99)
        assert isinstance(result, dict)
        assert "exists" in result
        assert "blocking_processes" in result
        assert isinstance(result["blocking_processes"], list)

    def test_does_not_raise(self):
        # Should never raise, even for an absurd index
        result = diagnose_camera_failure(9999)
        assert isinstance(result, dict)


# ---------- diagnose_camera_failure — mocked ----------

class TestDiagnoseCameraFailureMocked:
    @patch("aprilcam.camutil.sys")
    @patch("aprilcam.camutil._macos_avfoundation_device_names")
    def test_macos_camera_not_found(self, mock_av_names, mock_sys):
        mock_sys.platform = "darwin"
        mock_av_names.return_value = {0: "FaceTime HD Camera"}

        result = diagnose_camera_failure(5)
        assert result["exists"] is False

    @patch("aprilcam.camutil.sys")
    @patch("aprilcam.camutil._macos_avfoundation_device_names")
    @patch("aprilcam.camutil.subprocess")
    def test_macos_blocking_processes(self, mock_subprocess, mock_av_names, mock_sys):
        mock_sys.platform = "darwin"
        mock_av_names.return_value = {0: "FaceTime HD Camera"}

        mock_proc = MagicMock()
        mock_proc.stdout = (
            "zoom      1234  user  txt  REG  1,4  123456  /Library/AppleCamera\n"
            "python3   5678  user  txt  REG  1,4  789012  /some/camera/path\n"
        )
        mock_subprocess.run.return_value = mock_proc

        result = diagnose_camera_failure(0)
        assert result["exists"] is True
        assert len(result["blocking_processes"]) >= 1
        pids = [p["pid"] for p in result["blocking_processes"]]
        assert 1234 in pids

    @patch("aprilcam.camutil.sys")
    @patch("aprilcam.camutil.os.path.exists")
    def test_linux_device_not_found(self, mock_exists, mock_sys):
        mock_sys.platform = "linux"
        mock_exists.return_value = False

        result = diagnose_camera_failure(99)
        assert result["exists"] is False

    @patch("aprilcam.camutil.sys")
    @patch("aprilcam.camutil.os.path.exists")
    @patch("aprilcam.camutil.subprocess")
    @patch("aprilcam.camutil._get_process_name")
    def test_linux_blocking_processes(self, mock_get_name, mock_subprocess, mock_exists, mock_sys):
        mock_sys.platform = "linux"
        mock_exists.return_value = True
        mock_get_name.return_value = "ffmpeg"

        mock_proc = MagicMock()
        mock_proc.stdout = ""
        mock_proc.stderr = "/dev/video0:  4321"
        mock_subprocess.run.return_value = mock_proc

        result = diagnose_camera_failure(0)
        assert result["exists"] is True
        assert len(result["blocking_processes"]) == 1
        assert result["blocking_processes"][0]["pid"] == 4321
        assert result["blocking_processes"][0]["name"] == "ffmpeg"

    @patch("aprilcam.camutil.sys")
    @patch("aprilcam.camutil._macos_avfoundation_device_names")
    def test_subprocess_failure_no_crash(self, mock_av_names, mock_sys):
        mock_sys.platform = "darwin"
        mock_av_names.side_effect = OSError("ffmpeg not found")

        # Should not raise
        result = diagnose_camera_failure(0)
        assert isinstance(result, dict)
        assert "exists" in result
