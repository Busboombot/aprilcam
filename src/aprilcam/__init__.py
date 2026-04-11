from pathlib import Path as _Path

from aprilcam.stream import detect_tags, detect_objects, calibrate
from aprilcam.core.detection import TagRecord
from aprilcam.core.aprilcam import AprilCam
from aprilcam.core.models import AprilTag
from aprilcam.core.playfield import Playfield
from aprilcam.vision.objects import ObjectRecord
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
    "detect_objects",
    "calibrate",
    "TagRecord",
    "AprilCam",
    "AprilTag",
    "Playfield",
    "ObjectRecord",
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
