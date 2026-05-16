"""Camera calibration types and workflow functions.

Contains :class:`FieldSpec`, :class:`CameraCalibration`, and all
functions that drive the calibration workflow.  Pure homography math
lives in :mod:`aprilcam.calibration.homography`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    width_in: float  # left->right
    height_in: float  # upper->lower
    units: str  # "inch" or "cm"

    @property
    def width_cm(self) -> float:
        if self.units == "inch":
            return self.width_in * 2.54
        return self.width_in

    @property
    def height_cm(self) -> float:
        if self.units == "inch":
            return self.height_in * 2.54
        return self.height_in


@dataclass
class CameraCalibration:
    """Stores everything needed to undistort + homography-transform a frame.

    For cameras without barrel distortion, *camera_matrix* and
    *dist_coeffs* are ``None`` and ``undistort()`` is a no-op.

    The optional *settings* dict stores hardware control values to apply
    when the camera is opened.  Expected shape::

        {
            "program": "uvc-util",
            "controls": {"exposure-time-abs": "10", "gain": "1", ...}
        }
    """

    device_name: str
    resolution: Tuple[int, int]  # (width, height)
    homography: np.ndarray  # 3x3 pixel->world
    camera_matrix: Optional[np.ndarray] = None  # 3x3 intrinsics
    dist_coeffs: Optional[np.ndarray] = None  # (k1,k2,p1,p2,k3)
    tags_used: int = 0
    rms_error: float = 0.0
    settings: Optional[Dict] = None  # hardware control settings
    pipeline: Optional[Dict] = None  # DetectorConfig overrides

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Remove barrel distortion if calibration data is available."""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return frame

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Map a pixel coordinate to world (cm) coordinates."""
        vec = self.homography @ np.array([u, v, 1.0])
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d: dict = {
            "device_name": self.device_name,
            "resolution": list(self.resolution),
            "homography": self.homography.tolist(),
            "tags_used": self.tags_used,
            "rms_error": self.rms_error,
        }
        if self.camera_matrix is not None:
            d["camera_matrix"] = self.camera_matrix.tolist()
        if self.dist_coeffs is not None:
            d["dist_coeffs"] = self.dist_coeffs.tolist()
        if self.settings is not None:
            d["settings"] = self.settings
        if self.pipeline is not None:
            d["pipeline"] = self.pipeline
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        """Deserialize from a JSON-compatible dict."""
        cm = np.array(d["camera_matrix"], dtype=float) if "camera_matrix" in d else None
        dc = np.array(d["dist_coeffs"], dtype=float) if "dist_coeffs" in d else None
        return cls(
            device_name=d["device_name"],
            resolution=tuple(d["resolution"]),
            homography=np.array(d["homography"], dtype=float),
            camera_matrix=cm,
            dist_coeffs=dc,
            tags_used=d.get("tags_used", 0),
            rms_error=d.get("rms_error", 0.0),
            settings=d.get("settings"),
            pipeline=d.get("pipeline"),
        )


# ---------------------------------------------------------------------------
# Per-camera calibration directory helpers (new scheme)
# ---------------------------------------------------------------------------


def device_name_slug(device_name: str) -> str:
    """Slugify a camera device name for use as a filename component.

    ``"Arducam OV9782 USB Camera"`` → ``"arducam-ov9782-usb-camera"``
    """
    import re
    return re.sub(r"[^a-z0-9]+", "-", device_name.lower()).strip("-")


def calibration_file_for_camera(
    calibration_dir: str | Path,
    device_name: str,
) -> Path:
    """Return the path to the per-camera calibration file.

    The filename is ``<device-slug>.json`` inside *calibration_dir*.
    """
    return Path(calibration_dir) / f"{device_name_slug(device_name)}.json"


def load_calibration_from_dir(
    device_name: str,
    calibration_dir: str | Path,
) -> Optional["CameraCalibration"]:
    """Load calibration for *device_name* from a per-camera directory.

    Reads ``<calibration_dir>/<device-slug>.json``.  Returns ``None``
    if the file does not exist or cannot be parsed.

    The file must contain at minimum ``"homography"``.  Optional fields
    ``"field_width_cm"`` and ``"field_height_cm"`` are ignored here but
    present in the file for operator reference.
    """
    cal_file = calibration_file_for_camera(calibration_dir, device_name)
    if not cal_file.exists():
        return None
    try:
        data = json.loads(cal_file.read_text())
        # Per-camera file has the camera data at the top level (no "cameras" nesting).
        data.setdefault("device_name", device_name)
        return CameraCalibration.from_dict(data)
    except Exception:
        return None


def save_calibration_for_camera(
    cal: "CameraCalibration",
    calibration_dir: str | Path,
    field_width_cm: float,
    field_height_cm: float,
) -> Path:
    """Write a per-camera calibration file to *calibration_dir*.

    Creates ``<calibration_dir>/<device-slug>.json`` with:
    - ``field_width_cm``, ``field_height_cm`` — playfield dimensions
    - all fields from *cal* (homography, resolution, camera_matrix, etc.)

    Returns the path written.
    """
    cal_file = calibration_file_for_camera(calibration_dir, cal.device_name)
    cal_file.parent.mkdir(parents=True, exist_ok=True)
    data = cal.to_dict()
    data["field_width_cm"] = field_width_cm
    data["field_height_cm"] = field_height_cm
    cal_file.write_text(json.dumps(data, indent=2))
    return cal_file


def load_field_dimensions_from_dir(
    device_name: str,
    calibration_dir: str | Path,
) -> Optional[tuple]:
    """Return ``(width_cm, height_cm)`` from a per-camera calibration file.

    Returns ``None`` if the file is missing or dimensions are absent.
    """
    cal_file = calibration_file_for_camera(calibration_dir, device_name)
    if not cal_file.exists():
        return None
    try:
        data = json.loads(cal_file.read_text())
        w = data.get("field_width_cm")
        h = data.get("field_height_cm")
        if w is not None and h is not None:
            return (float(w), float(h))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# File paths (legacy unified-file scheme)
# ---------------------------------------------------------------------------


def calibration_path(data_dir: str | Path = "data") -> Path:
    """Return the path to the unified calibration file."""
    return Path(data_dir) / "calibration.json"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_field_dimensions(
    data_dir: str | Path = "data",
) -> Optional[tuple[float, float]]:
    """Return ``(width_cm, height_cm)`` from the top-level calibration file.

    Returns ``None`` if the file is missing or the keys are absent.
    """
    cal_file = calibration_path(data_dir)
    if not cal_file.exists():
        return None
    try:
        data = json.loads(cal_file.read_text())
        w = data.get("field_width_cm")
        h = data.get("field_height_cm")
        if w is not None and h is not None:
            return (float(w), float(h))
    except Exception:
        pass
    return None


def load_calibration_for_camera(
    device_name: str,
    data_dir: str | Path = "data",
) -> Optional[CameraCalibration]:
    """Load calibration for a specific camera from the unified file.

    Looks up by device_name in ``data/calibration.json``.  Returns
    ``None`` if the file doesn't exist or the camera isn't in it.
    """
    cal_file = calibration_path(data_dir)
    if not cal_file.exists():
        return None
    try:
        data = json.loads(cal_file.read_text())
        cameras = data.get("cameras", {})
        for _key, cam_data in cameras.items():
            if cam_data.get("device_name") == device_name:
                return CameraCalibration.from_dict(cam_data)
    except Exception:
        pass
    return None


def save_calibration(
    calibrations: List[CameraCalibration],
    data_dir: str | Path = "data",
    field_width_cm: float = 101.0,
    field_height_cm: float = 89.0,
) -> Path:
    """Save calibration for all cameras to ``data/calibration.json``.

    Each camera is keyed by its device_name.  Overwrites any existing
    file.  Returns the path written.
    """
    cameras = {}
    for cal in calibrations:
        cameras[cal.device_name] = cal.to_dict()

    data = {
        "type": "playfield",
        "field_width_cm": field_width_cm,
        "field_height_cm": field_height_cm,
        "cameras": cameras,
    }
    path = calibration_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def load_calibration(
    data_dir: str | Path = "data",
) -> Dict[str, CameraCalibration]:
    """Load all camera calibrations from ``data/calibration.json``.

    Returns:
        Dict mapping device_name -> CameraCalibration.
    """
    path = calibration_path(data_dir)
    data = json.loads(path.read_text())
    return {
        name: CameraCalibration.from_dict(cam_data)
        for name, cam_data in data.get("cameras", {}).items()
    }


# ---------------------------------------------------------------------------
# Calibration workflow
# ---------------------------------------------------------------------------


def _reprojection_rms(
    H: np.ndarray, pixel_pts: np.ndarray, world_pts: np.ndarray
) -> float:
    """Compute RMS reprojection error for a homography."""
    errors = []
    for px, wp in zip(pixel_pts, world_pts):
        vec = H @ np.array([px[0], px[1], 1.0])
        pred = np.array([vec[0] / vec[2], vec[1] / vec[2]])
        errors.append(np.linalg.norm(pred - wp))
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def calibrate_from_corners(
    pixel_corners: Dict[str, Tuple[float, float]],
    field_spec: FieldSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute homography from four corner positions and a field spec.

    Args:
        pixel_corners: Dict with keys 'upper_left', 'upper_right',
            'lower_left', 'lower_right', each a (x, y) tuple.
        field_spec: FieldSpec with physical dimensions.

    Returns:
        Tuple of (H, pixel_pts, world_pts_cm) where H is the 3x3
        homography matrix mapping [u,v,1] pixels to [X,Y,W] world cm,
        pixel_pts is the 4x2 array of pixel coordinates, and
        world_pts_cm is the 4x2 array of world coordinates in cm.
    """
    # Import compute_homography lazily to avoid circular import at module load.
    # homography.py re-exports from this module; this module calls into
    # homography.py only at runtime.
    from .homography import compute_homography

    world_pts_cm = np.array([
        [0.0, 0.0],
        [field_spec.width_cm, 0.0],
        [0.0, field_spec.height_cm],
        [field_spec.width_cm, field_spec.height_cm],
    ], dtype=np.float32)
    pixel_pts = np.array([
        pixel_corners["upper_left"],
        pixel_corners["upper_right"],
        pixel_corners["lower_left"],
        pixel_corners["lower_right"],
    ], dtype=np.float32)
    H = compute_homography(pixel_pts, world_pts_cm)
    return H, pixel_pts, world_pts_cm


def calibrate_single(
    cap: cv.VideoCapture,
    field_width_cm: float = 101.0,
    field_height_cm: float = 89.0,
    num_frames: int = 30,
    correct_distortion: bool = True,
    camera_index: int = 0,
) -> CameraCalibration:
    """Calibrate a single camera using ArUco corners and AprilTags.

    Detects ArUco 4x4 corner markers (known world positions) and all
    AprilTags visible in the frame.  Computes homography from the 4
    ArUco corners, then refines using all detected tags.  Optionally
    estimates lens distortion if enough points are available.

    Args:
        cap: Open VideoCapture for the camera.
        field_width_cm: Playfield width between ArUco corners in cm.
        field_height_cm: Playfield height between ArUco corners in cm.
        num_frames: Frames to accumulate for tag detection.
        correct_distortion: Attempt barrel distortion correction.
        camera_index: Camera index (for device name lookup).

    Returns:
        CameraCalibration for the camera.
    """
    from .homography import compute_homography, detect_all_tags

    corner_world = {
        -1: (0.0, 0.0),
        -2: (field_width_cm, 0.0),
        -3: (0.0, field_height_cm),
        -4: (field_width_cm, field_height_cm),
    }

    tags = detect_all_tags(cap, num_frames)

    cam_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Homography from ArUco corners
    corner_pixels = []
    corner_worlds = []
    for neg_id, world_xy in corner_world.items():
        if neg_id in tags:
            corner_pixels.append(tags[neg_id])
            corner_worlds.append(world_xy)

    if len(corner_pixels) < 4:
        raise RuntimeError(
            f"Camera: only {len(corner_pixels)} ArUco corners found, need 4"
        )

    pixel_pts = np.array(corner_pixels, dtype=np.float32)
    world_pts = np.array(corner_worlds, dtype=np.float32)
    H = compute_homography(pixel_pts, world_pts)

    # Compute world positions for AprilTags using corner-based homography
    all_px = list(corner_pixels)
    all_wp = list(corner_worlds)
    for tid, px in tags.items():
        if tid > 0:  # AprilTag (positive ID)
            vec = H @ np.array([px[0], px[1], 1.0])
            world_xy = (float(vec[0] / vec[2]), float(vec[1] / vec[2]))
            all_px.append(px)
            all_wp.append(world_xy)

    all_pixel = np.array(all_px, dtype=np.float32)
    all_world = np.array(all_wp, dtype=np.float32)
    n_pts = len(all_px)

    # Optionally correct barrel distortion
    camera_matrix = None
    dist_coeffs = None

    if correct_distortion and n_pts >= 6:
        obj_pts_3d = np.zeros((n_pts, 1, 3), dtype=np.float32)
        obj_pts_3d[:, 0, :2] = all_world
        img_pts = all_pixel.reshape(n_pts, 1, 2)

        _rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv.calibrateCamera(
            [obj_pts_3d], [img_pts], (cam_w, cam_h), None, None
        )
        dist_coeffs = dist_coeffs.flatten()

        undist_pts = cv.undistortPoints(
            all_pixel.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
        ).reshape(-1, 2)
        H = compute_homography(undist_pts, all_world)
    elif n_pts > 4:
        # Recompute with all points for better accuracy
        H = compute_homography(all_pixel, all_world)

    rms = _reprojection_rms(H, all_pixel, all_world)

    from ..camera.camutil import get_device_name

    return CameraCalibration(
        device_name=get_device_name(camera_index),
        resolution=(cam_w, cam_h),
        homography=H,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        tags_used=n_pts,
        rms_error=rms,
    )


def calibrate_joint(
    bw_cap: cv.VideoCapture,
    color_cap: cv.VideoCapture,
    field_width_cm: float = 101.0,
    field_height_cm: float = 89.0,
    num_frames: int = 30,
    correct_distortion: bool = True,
    bw_index: int = 3,
    color_index: int = 2,
) -> Tuple[CameraCalibration, CameraCalibration]:
    """Run joint multi-tag calibration on two cameras.

    Uses ArUco 4x4 corner markers (known world positions) and AprilTags
    (world positions computed from the B&W camera's homography) as
    shared reference points.  When *correct_distortion* is True and
    enough points are available (>=6), estimates lens distortion
    coefficients for the color camera.

    Args:
        bw_cap: Open VideoCapture for the B&W (primary) camera.
        color_cap: Open VideoCapture for the color (secondary) camera.
        field_width_cm: Playfield width between ArUco corners in cm.
        field_height_cm: Playfield height between ArUco corners in cm.
        num_frames: Frames to accumulate for tag detection.
        correct_distortion: Attempt barrel distortion correction on color.

    Returns:
        Tuple of (bw_calibration, color_calibration).
    """
    from .homography import compute_homography, detect_all_tags

    # Update corner world positions with actual field dimensions
    corner_world = {
        -1: (0.0, 0.0),
        -2: (field_width_cm, 0.0),
        -3: (0.0, field_height_cm),
        -4: (field_width_cm, field_height_cm),
    }

    # Step 1: Detect tags on both cameras
    bw_tags = detect_all_tags(bw_cap, num_frames)
    color_tags = detect_all_tags(color_cap, num_frames)

    bw_w = int(bw_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    bw_h = int(bw_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    color_w = int(color_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    color_h = int(color_cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Step 2: B&W camera homography from ArUco corners
    bw_corner_pixels = []
    bw_corner_world = []
    for neg_id, world_xy in corner_world.items():
        if neg_id in bw_tags:
            bw_corner_pixels.append(bw_tags[neg_id])
            bw_corner_world.append(world_xy)

    if len(bw_corner_pixels) < 4:
        raise RuntimeError(
            f"B&W camera: only {len(bw_corner_pixels)} ArUco corners found, need 4"
        )

    bw_pixel_pts = np.array(bw_corner_pixels, dtype=np.float32)
    bw_world_pts = np.array(bw_corner_world, dtype=np.float32)
    bw_H = compute_homography(bw_pixel_pts, bw_world_pts)

    # Step 3: Compute world positions for ALL AprilTags using B&W homography
    tag_world_positions: Dict[int, Tuple[float, float]] = dict(corner_world)
    for tid, px in bw_tags.items():
        if tid > 0:  # AprilTag (positive ID)
            vec = bw_H @ np.array([px[0], px[1], 1.0])
            tag_world_positions[tid] = (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    # Step 4: Build color camera correspondences from shared tags
    color_pixel_pts = []
    color_world_pts = []
    for tid, world_xy in tag_world_positions.items():
        if tid in color_tags:
            color_pixel_pts.append(color_tags[tid])
            color_world_pts.append(world_xy)

    n_color_pts = len(color_pixel_pts)
    if n_color_pts < 4:
        raise RuntimeError(
            f"Color camera: only {n_color_pts} shared tags found, need >= 4"
        )

    color_px = np.array(color_pixel_pts, dtype=np.float32)
    color_wp = np.array(color_world_pts, dtype=np.float32)

    # Step 5: Color camera calibration
    color_cm = None
    color_dc = None
    color_rms = 0.0

    if correct_distortion and n_color_pts >= 6:
        # Use cv.calibrateCamera for distortion + intrinsics.
        # It needs 3D object points (add z=0 for planar).
        obj_pts_3d = np.zeros((n_color_pts, 1, 3), dtype=np.float32)
        obj_pts_3d[:, 0, :2] = color_wp
        img_pts = color_px.reshape(n_color_pts, 1, 2)

        color_rms, color_cm, color_dc, _rvecs, _tvecs = cv.calibrateCamera(
            [obj_pts_3d], [img_pts], (color_w, color_h), None, None
        )
        color_dc = color_dc.flatten()

        # Undistort the pixel points and recompute homography
        undist_pts = cv.undistortPoints(
            color_px.reshape(-1, 1, 2), color_cm, color_dc, P=color_cm
        ).reshape(-1, 2)
        color_H = compute_homography(undist_pts, color_wp)
    else:
        # Not enough points for distortion -- plain homography
        color_H = compute_homography(color_px, color_wp)

    # Compute B&W RMS error
    bw_all_px = []
    bw_all_wp = []
    for tid, world_xy in tag_world_positions.items():
        if tid in bw_tags:
            bw_all_px.append(bw_tags[tid])
            bw_all_wp.append(world_xy)
    if len(bw_all_px) > 4:
        # Recompute B&W homography with ALL points for better accuracy
        bw_all_pixel = np.array(bw_all_px, dtype=np.float32)
        bw_all_world = np.array(bw_all_wp, dtype=np.float32)
        bw_H = compute_homography(bw_all_pixel, bw_all_world)

    # Compute RMS reprojection errors
    bw_rms = _reprojection_rms(bw_H, bw_all_pixel, bw_all_world)

    from ..camera.camutil import get_device_name

    bw_cal = CameraCalibration(
        device_name=get_device_name(bw_index),
        resolution=(bw_w, bw_h),
        homography=bw_H,
        tags_used=len(bw_all_px),
        rms_error=bw_rms,
    )
    color_cal = CameraCalibration(
        device_name=get_device_name(color_index),
        resolution=(color_w, color_h),
        homography=color_H,
        camera_matrix=color_cm,
        dist_coeffs=color_dc,
        tags_used=n_color_pts,
        rms_error=color_rms,
    )
    return bw_cal, color_cal


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def calibrate(
    camera: "cv.VideoCapture | int",
    *,
    width_cm: float = 101.0,
    height_cm: float = 89.0,
    frames: int = 30,
    output: "str | Path | None" = None,
) -> CameraCalibration:
    """Calibrate a camera for playfield homography.

    Args:
        camera: An open :class:`cv.VideoCapture` or a camera index.
        width_cm: Playfield width between ArUco corners in cm.
        height_cm: Playfield height between ArUco corners in cm.
        frames: Frames to accumulate for tag detection.
        output: Optional path to save the calibration file.  If given,
            the result is merged into (or created as) a
            ``calibration.json`` at that path.

    Returns:
        :class:`CameraCalibration` for the camera.
    """
    own_cap = False
    if isinstance(camera, int):
        cap = cv.VideoCapture(camera)
        cam_index = camera
        own_cap = True
    else:
        cap = camera
        cam_index = 0

    try:
        cal = calibrate_single(
            cap,
            field_width_cm=width_cm,
            field_height_cm=height_cm,
            num_frames=frames,
            camera_index=cam_index,
        )
    finally:
        if own_cap:
            cap.release()

    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            try:
                import json as _json
                cal_data = _json.loads(out_path.read_text())
            except Exception:
                cal_data = {}
        else:
            cal_data = {}
        cal_data.setdefault("type", "playfield")
        cal_data.setdefault("field_width_cm", width_cm)
        cal_data.setdefault("field_height_cm", height_cm)
        if "cameras" not in cal_data:
            cal_data["cameras"] = {}
        cal_data["cameras"][cal.device_name] = cal.to_dict()
        import json as _json
        out_path.write_text(_json.dumps(cal_data, indent=2))

    return cal


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------


def save_joint_calibration(
    bw_cal: CameraCalibration,
    color_cal: CameraCalibration,
    path: Path,
    field_width_cm: float = 101.0,
    field_height_cm: float = 89.0,
) -> None:
    """Save calibration (legacy -- prefers :func:`save_calibration`)."""
    save_calibration(
        [bw_cal, color_cal],
        data_dir=path.parent,
        field_width_cm=field_width_cm,
        field_height_cm=field_height_cm,
    )


def load_joint_calibration(
    path: Path,
) -> Tuple[CameraCalibration, CameraCalibration]:
    """Load calibration (legacy -- prefers :func:`load_calibration`)."""
    data = json.loads(path.read_text())
    cams = list(data.get("cameras", {}).values())
    if len(cams) < 2:
        raise ValueError(f"Expected at least 2 cameras in {path}")
    return (
        CameraCalibration.from_dict(cams[0]),
        CameraCalibration.from_dict(cams[1]),
    )
