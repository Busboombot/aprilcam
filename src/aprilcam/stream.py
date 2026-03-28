"""Generator-based tag detection API.

Provides :func:`detect_tags`, the primary library interface for opening
a camera, loading homography, and yielding tag records per frame.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np

from .aprilcam import AprilCam
from .camutil import list_cameras, get_device_name, select_camera_by_pattern
from .config import AppConfig
from .detection import TagRecord
from .homography import discover_homography


def _resolve_camera_index(camera: int | str) -> int:
    """Resolve a camera argument to an integer index.

    If *camera* is already an int, return it directly.  If it is a
    string, enumerate cameras with detailed names and match by pattern.
    """
    if isinstance(camera, int):
        return camera
    cams = list_cameras(detailed_names=True)
    idx = select_camera_by_pattern(camera, cams)
    if idx is not None:
        return idx
    raise ValueError(f"No camera matching pattern {camera!r}")


def _load_homography_matrix(
    homography: str | Path | None,
    cap: cv.VideoCapture,
    camera_index: int,
    data_dir: str | Path,
) -> np.ndarray | None:
    """Load a 3x3 homography matrix based on the *homography* parameter."""
    if homography is None:
        return None

    data_path = Path(data_dir)

    if homography == "auto":
        # Try to discover per-camera homography file
        device_name = get_device_name(camera_index)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        found = discover_homography(device_name, width, height, data_path)
        if found is None:
            return None
        hpath = found
    else:
        hpath = Path(homography)

    if not hpath.exists():
        return None

    data = json.loads(hpath.read_text())
    H = np.array(data["homography"], dtype=float)
    if H.shape != (3, 3):
        return None
    return H


def detect_tags(
    camera: int | str = 0,
    homography: str | Path | None = "auto",
    family: str = "36h11",
    data_dir: str | Path = "data",
    proc_width: int = 0,
) -> Generator[list[TagRecord], None, None]:
    """Open a camera, auto-load homography, and yield tag records per frame.

    Args:
        camera: Camera index (int) or device name pattern (str).
        homography: ``"auto"`` to discover per-camera file from *data_dir*,
            a path to a specific file, or ``None`` for pixel-only mode.
        family: AprilTag family (default ``"36h11"``).
        data_dir: Directory containing homography files.
        proc_width: Processing width in pixels (0 = native resolution).

    Yields:
        ``list[TagRecord]`` per frame -- each record includes tag ID, pixel
        center, world coordinates (if calibrated), orientation, velocity.
    """
    index = _resolve_camera_index(camera)
    cap = cv.VideoCapture(index)

    try:
        if not cap.isOpened():
            from .errors import CameraError
            raise CameraError(f"Failed to open camera {index}")

        H = _load_homography_matrix(homography, cap, index, data_dir)

        cam = AprilCam(
            index=index,
            backend=None,
            speed_alpha=0.3,
            family=family,
            proc_width=proc_width,
            cap=cap,
            homography=H,
            headless=True,
        )
        cam.reset_state()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tag_records = cam.process_frame(frame, time.monotonic())
            yield tag_records
    finally:
        if cap.isOpened():
            cap.release()
