"""aprilcam.client — typed client-side objects for the AprilCam daemon.

Application code imports from this package instead of touching
proto-generated types directly.
"""

from aprilcam.client.models import (
    CameraInfo,
    ImageFrame,
    PathRecord,
    StreamEndpoint,
    TagFrame,
    TagRecord,
)

__all__ = [
    "CameraInfo",
    "ImageFrame",
    "PathRecord",
    "StreamEndpoint",
    "TagFrame",
    "TagRecord",
]
