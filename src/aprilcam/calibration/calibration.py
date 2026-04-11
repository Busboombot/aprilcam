"""Camera calibration workflow and data types.

This module owns the calibration workflow (detecting corners, computing
homography, saving/loading results).  Pure homography math lives in
:mod:`aprilcam.calibration.homography`.
"""

from __future__ import annotations

from .homography import (
    FieldSpec,
    CameraCalibration,
    calibrate_single,
    calibrate_from_corners,
    calibration_path,
    save_calibration,
    load_calibration,
    load_calibration_for_camera,
)


def calibrate(
    camera,
    *,
    width_cm: float = 101.0,
    height_cm: float = 89.0,
    frames: int = 30,
    output=None,
):
    """Calibrate a camera for playfield homography.

    High-level entry point. Opens the camera if needed, detects ArUco
    corner markers, computes homography, and saves the result.

    Args:
        camera: A Camera instance (or anything with a read() method).
        width_cm: Playfield width between ArUco corners in cm.
        height_cm: Playfield height between ArUco corners in cm.
        frames: Number of frames to accumulate for stable detection.
        output: Optional output path for the calibration JSON.

    Returns:
        CameraCalibration instance.
    """
    if hasattr(camera, "is_open") and not camera.is_open:
        camera.open()

    cal = calibrate_single(
        camera,
        field_width_cm=width_cm,
        field_height_cm=height_cm,
        num_frames=frames,
        camera_index=getattr(camera, "index", 0),
    )

    from ..iohelpers import get_data_dir

    data_dir = output or get_data_dir()
    save_calibration(
        [cal],
        data_dir=data_dir,
        field_width_cm=width_cm,
        field_height_cm=height_cm,
    )

    return cal


__all__ = [
    "calibrate",
    "FieldSpec",
    "CameraCalibration",
    "calibrate_single",
    "calibrate_from_corners",
    "calibration_path",
    "save_calibration",
    "load_calibration",
    "load_calibration_for_camera",
]
