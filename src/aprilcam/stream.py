"""Generator-based tag detection API.

Provides :func:`detect_tags`, the primary library interface for opening
a camera, loading homography, and yielding tag records per frame.
Also provides :func:`detect_objects` for standalone object detection.
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
from .detection import TagRecord
from .homography import discover_homography


def _resolve_camera_index(camera: int | str) -> int:
    """Resolve a camera argument to an integer index."""
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
        device_name = get_device_name(camera_index)

        from .homography import load_calibration_for_camera
        cal = load_calibration_for_camera(device_name, data_path)
        if cal is not None:
            return cal.homography

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
    if data.get("type") == "playfield" and "cameras" in data:
        device_name = get_device_name(camera_index)
        for _key, cam_data in data["cameras"].items():
            if cam_data.get("device_name") == device_name:
                H = np.array(cam_data["homography"], dtype=float)
                if H.shape == (3, 3):
                    return H
        return None

    H = np.array(data.get("homography", []), dtype=float)
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
) -> Generator[list[TagRecord], None, None]:
    """Open a camera, auto-load homography, and yield tag records per frame.

    Each yielded list includes all tags seen within the last ~1 second.
    Tags seen on the current frame have ``age=0.0``.  Stale tags (not
    seen this frame but seen recently) have ``age > 0``.

    When *detect_objects* is True, each yielded list has an ``objects``
    attribute containing detected :class:`~aprilcam.objects.ObjectRecord`
    items with color labels.  This runs square detection on the B&W
    frame and (if *color_camera* is provided) color classification on
    the color camera.

    Args:
        camera: Camera index (int) or device name pattern (str).
        homography: ``"auto"`` to discover from *data_dir*, a path, or ``None``.
        family: AprilTag family (default ``"36h11"``).
        data_dir: Directory containing calibration files.
        proc_width: Processing width in pixels (0 = native resolution).
        detect_objects: If True, detect colored cubes alongside tags.
        color_camera: Camera index for color classification (requires
            *detect_objects*).

    Yields:
        ``list[TagRecord]`` per frame.  When *detect_objects* is True,
        the list has an ``objects`` attribute with detected objects.
    """
    index = _resolve_camera_index(camera)
    cap = cv.VideoCapture(index)
    color_cap = None

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

        # Set up object detection if requested
        sq_detector = None
        color_cal = None
        if detect_objects:
            from .objects import SquareDetector
            sq_detector = SquareDetector()

            if color_camera is not None:
                color_idx = _resolve_camera_index(color_camera)
                color_cap = cv.VideoCapture(color_idx)
                if not color_cap.isOpened():
                    color_cap = None
                else:
                    try:
                        from .homography import load_calibration
                        all_cals = load_calibration(data_dir)
                        for _name, cal in all_cals.items():
                            if cal.dist_coeffs is not None or cal.resolution[0] > 1280:
                                color_cal = cal
                                break
                    except Exception:
                        pass

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tag_records = cam.process_frame(frame, time.monotonic())

            # Object detection
            objects = []
            if sq_detector is not None:
                sq_detector.update_known_tags(tag_records)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                tag_corners = [
                    np.array(t.corners_px, dtype=np.float32)
                    for t in tag_records if t.age == 0.0
                ]
                pf_poly = cam.playfield.get_polygon()
                objects = sq_detector.detect(
                    gray, homography=H, tag_corners=tag_corners,
                    playfield_polygon=pf_poly,
                )

                # Color classify via color camera
                if color_cap is not None and color_cal is not None and objects:
                    try:
                        from .color_classifier import ColorClassifier
                        from dataclasses import replace
                        ret_c, color_frame = color_cap.read()
                        if ret_c and color_frame is not None:
                            color_frame = color_cal.undistort(color_frame)
                            classifier = ColorClassifier()
                            H_inv = np.linalg.inv(color_cal.homography)
                            colored = []
                            for obj in objects:
                                if obj.world_xy:
                                    vec = H_inv @ np.array([obj.world_xy[0], obj.world_xy[1], 1.0])
                                    cpx, cpy = vec[0]/vec[2], vec[1]/vec[2]
                                    color = classifier.classify_at_point(color_frame, cpx, cpy, radius=25)
                                    colored.append(replace(obj, color=color))
                                else:
                                    colored.append(obj)
                            objects = colored
                    except Exception:
                        pass

            # Attach objects to the tag list as an attribute
            result = _TagListWithObjects(tag_records, objects)
            yield result
    finally:
        if color_cap is not None:
            try:
                color_cap.release()
            except Exception:
                pass
        if cap.isOpened():
            cap.release()


class _TagListWithObjects(list):
    """A list[TagRecord] with an extra ``objects`` attribute."""

    def __init__(self, tags, objects=None):
        super().__init__(tags)
        self.objects = objects or []


def detect_objects(
    camera: int | str = 0,
    color_camera: int | str | None = None,
    homography: str | Path | None = "auto",
    data_dir: str | Path = "data",
) -> list:
    """One-shot object detection: find colored cubes on the playfield.

    Opens the B&W camera, detects bright squares, optionally classifies
    colors from the color camera, and returns a list of
    :class:`~aprilcam.objects.ObjectRecord`.

    Args:
        camera: B&W camera index or name.
        color_camera: Color camera index or name (optional).
        homography: Homography source (``"auto"`` recommended).
        data_dir: Directory containing calibration files.

    Returns:
        List of :class:`~aprilcam.objects.ObjectRecord` with world
        positions and color labels.
    """
    from .objects import SquareDetector, ObjectRecord

    index = _resolve_camera_index(camera)
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        from .errors import CameraError
        raise CameraError(f"Failed to open camera {index}")

    try:
        H = _load_homography_matrix(homography, cap, index, data_dir)

        cam = AprilCam(
            index=index, backend=None, speed_alpha=0.3, family="36h11",
            proc_width=0, cap=cap, homography=H, headless=True,
        )
        cam.reset_state()

        # Accumulate tag positions for exclusion
        det = SquareDetector()
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                tags = cam.process_frame(frame, time.monotonic())
                det.update_known_tags(tags)

        ret, frame = cap.read()
        if not ret:
            return []
        tags = cam.process_frame(frame, time.monotonic())
        det.update_known_tags(tags)
        cap.release()
        cap = None

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        tag_corners = [np.array(t.corners_px, dtype=np.float32) for t in tags if t.age == 0.0]
        pf_poly = cam.playfield.get_polygon()
        objects = det.detect(gray, homography=H, tag_corners=tag_corners, playfield_polygon=pf_poly)

        # Color classify
        if color_camera is not None and objects:
            try:
                from .color_classifier import ColorClassifier
                from .homography import load_calibration
                from dataclasses import replace

                color_idx = _resolve_camera_index(color_camera)
                cc = cv.VideoCapture(color_idx)
                if cc.isOpened():
                    color_cal = None
                    try:
                        all_cals = load_calibration(data_dir)
                        for _name, cal in all_cals.items():
                            if cal.dist_coeffs is not None or cal.resolution[0] > 1280:
                                color_cal = cal
                                break
                    except Exception:
                        pass

                    for _ in range(3):
                        cc.read()
                    ret_c, color_frame = cc.read()
                    cc.release()

                    if ret_c and color_frame is not None and color_cal is not None:
                        color_frame = color_cal.undistort(color_frame)
                        classifier = ColorClassifier()
                        H_inv = np.linalg.inv(color_cal.homography)
                        colored = []
                        for obj in objects:
                            if obj.world_xy:
                                vec = H_inv @ np.array([obj.world_xy[0], obj.world_xy[1], 1.0])
                                cpx, cpy = vec[0]/vec[2], vec[1]/vec[2]
                                color = classifier.classify_at_point(color_frame, cpx, cpy, radius=25)
                                colored.append(replace(obj, color=color))
                            else:
                                colored.append(obj)
                        objects = colored
            except Exception:
                pass

        return objects
    finally:
        if cap is not None and cap.isOpened():
            cap.release()


def calibrate(
    bw_camera: int | str = 3,
    color_camera: int | str = 2,
    field_width_cm: float = 101.0,
    field_height_cm: float = 89.0,
    data_dir: str | Path = "data",
    num_frames: int = 30,
) -> Path:
    """Run joint calibration on two cameras and save to data/calibration.json.

    Opens both cameras, detects all ArUco + AprilTag markers as shared
    reference points, computes homography for both cameras (with barrel
    distortion correction on the color camera if enough points), and
    saves everything to a single calibration file.

    Args:
        bw_camera: B&W camera index or name pattern.
        color_camera: Color camera index or name pattern.
        field_width_cm: Playfield width between ArUco corners.
        field_height_cm: Playfield height between ArUco corners.
        data_dir: Directory to save calibration.json.
        num_frames: Frames to accumulate for tag detection.

    Returns:
        Path to the saved calibration file.
    """
    from .homography import calibrate_joint, save_calibration

    bw_index = _resolve_camera_index(bw_camera)
    color_index = _resolve_camera_index(color_camera)

    bw_cap = cv.VideoCapture(bw_index)
    color_cap = cv.VideoCapture(color_index)

    try:
        # Warm up
        for _ in range(10):
            bw_cap.read()
        for _ in range(5):
            color_cap.read()

        bw_cal, color_cal = calibrate_joint(
            bw_cap, color_cap,
            field_width_cm=field_width_cm,
            field_height_cm=field_height_cm,
            num_frames=num_frames,
            bw_index=bw_index,
            color_index=color_index,
        )
    finally:
        bw_cap.release()
        color_cap.release()

    path = save_calibration(
        [bw_cal, color_cal],
        data_dir=data_dir,
        field_width_cm=field_width_cm,
        field_height_cm=field_height_cm,
    )

    print(f"Calibration saved to {path}")
    print(f"  B&W:   {bw_cal.device_name} {bw_cal.resolution}, {bw_cal.tags_used} tags, RMS {bw_cal.rms_error:.2f}cm")
    print(f"  Color: {color_cal.device_name} {color_cal.resolution}, {color_cal.tags_used} tags, RMS {color_cal.rms_error:.2f}")
    if color_cal.dist_coeffs is not None:
        print(f"  Barrel distortion correction: yes")

    return path
