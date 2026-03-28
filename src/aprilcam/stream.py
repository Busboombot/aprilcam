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
from .objects import FrameResult, ObjectFuser, SquareDetector


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
    detect_objects: bool = False,
    color_camera: int | str | None = None,
) -> Generator[FrameResult, None, None]:
    """Open a camera, auto-load homography, and yield tag records per frame.

    Args:
        camera: Camera index (int) or device name pattern (str).
        homography: ``"auto"`` to discover per-camera file from *data_dir*,
            a path to a specific file, or ``None`` for pixel-only mode.
        family: AprilTag family (default ``"36h11"``).
        data_dir: Directory containing homography files.
        proc_width: Processing width in pixels (0 = native resolution).
        detect_objects: If ``True``, run square object detection each frame.
        color_camera: Camera index or pattern for a secondary color camera.
            When provided alongside ``detect_objects=True``, a background
            thread classifies object colors from the color camera feed.

    Yields:
        :class:`~aprilcam.objects.FrameResult` per frame.  The result is
        backward-compatible with ``list[TagRecord]`` (supports iteration,
        ``len()``, and indexing over tags).
    """
    index = _resolve_camera_index(camera)
    cap = cv.VideoCapture(index)
    color_thread = None

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

        # Set up object detection pipeline when requested.
        square_detector: SquareDetector | None = None
        fuser: ObjectFuser | None = None
        if detect_objects:
            square_detector = SquareDetector()
            fuser = ObjectFuser()

            if color_camera is not None:
                from .color_classifier import ColorClassifier
                from .objects import ColorCameraThread

                color_index = _resolve_camera_index(color_camera)
                color_H = _load_homography_matrix(
                    homography, cap, color_index, data_dir
                )
                classifier = ColorClassifier()
                color_thread = ColorCameraThread(
                    camera_index=color_index,
                    fuser=fuser,
                    classifier=classifier,
                    homography=color_H,
                )
                color_thread.start()

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tag_records = cam.process_frame(frame, time.monotonic())

            objects = []
            if square_detector is not None and fuser is not None:
                tag_corners = [
                    np.array(t.corners_px, dtype=np.float32)
                    for t in tag_records
                ]
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                objects = square_detector.detect(
                    gray, homography=H, tag_corners=tag_corners
                )
                objects = fuser.fuse(objects)

            yield FrameResult(
                tags=tag_records,
                objects=objects,
                timestamp=time.monotonic(),
                frame_index=frame_index,
            )
            frame_index += 1
    finally:
        if color_thread is not None:
            color_thread.stop()
        if cap.isOpened():
            cap.release()
