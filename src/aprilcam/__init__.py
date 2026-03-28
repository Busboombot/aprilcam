from pathlib import Path as _Path

from aprilcam.stream import detect_tags
from aprilcam.detection import TagRecord
from aprilcam.aprilcam import AprilCam
from aprilcam.models import AprilTag
from aprilcam.playfield import Playfield
from aprilcam.errors import (
    CameraError,
    CameraInUseError,
    CameraNotFoundError,
    CameraPermissionError,
)

__all__ = [
    "__version__",
    "help",
    "detect_tags",
    "TagRecord",
    "AprilCam",
    "AprilTag",
    "Playfield",
    "CameraError",
    "CameraInUseError",
    "CameraNotFoundError",
    "CameraPermissionError",
]
__version__ = "0.1.0"

_AGENT_GUIDE = _Path(__file__).parent / "AGENT_GUIDE.md"


def help() -> str:
    """Return the AprilCam agent guide as a markdown string.

    This guide explains how to use AprilCam as a library and via MCP,
    including available tools, common workflows, and tips for AI agents.
    """
    return _AGENT_GUIDE.read_text()
