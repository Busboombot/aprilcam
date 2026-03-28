"""Exception classes for camera errors."""

from __future__ import annotations


class CameraError(Exception):
    """Base exception for camera errors."""


class CameraNotFoundError(CameraError):
    """Camera index does not exist."""


class CameraInUseError(CameraError):
    """Camera is in use by another process."""

    def __init__(self, message: str, pid: int | None = None, process_name: str | None = None):
        super().__init__(message)
        self.pid = pid
        self.process_name = process_name


class CameraPermissionError(CameraError):
    """Insufficient permissions to access camera."""
