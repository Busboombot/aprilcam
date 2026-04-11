from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from ..camera.camutil import camera_slug
from ..config import AppConfig
from ..camera.screencap import ScreenCaptureMSS


# Mapping of special 4x4 ArUco IDs to board corners
# These IDs were used earlier when generating markers with labels.
CORNER_ID_MAP = {
    0: "upper_left",
    1: "upper_right",
    2: "lower_left",
    3: "lower_right",
}


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


def detect_aruco_4x4(gray: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    params = cv.aruco.DetectorParameters()
    det = cv.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = det.detectMarkers(gray)
    results: List[Tuple[np.ndarray, int]] = []
    if ids is not None and len(ids) > 0:
        for c, i in zip(corners, ids.flatten()):
            pts = np.array(c, dtype=np.float32).reshape(-1, 2)
            results.append((pts, int(i)))
    return results


def choose_corner_point(pts: np.ndarray) -> np.ndarray:
    """Return the tag center for homography correspondence.
    Heights are measured from the center of the tags per spec.
    """
    return pts.mean(axis=0)


def homography_path(slug: str, data_dir: str | Path = "data") -> Path:
    """Return the per-camera homography file path for a given slug."""
    return Path(data_dir) / f"homography-{slug}.json"


def calibration_path(data_dir: str | Path = "data") -> Path:
    """Return the path to the unified calibration file."""
    return Path(data_dir) / "calibration.json"


def load_calibration_for_camera(
    device_name: str,
    data_dir: str | Path = "data",
) -> Optional["CameraCalibration"]:
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


def discover_homography(
    device_name: str,
    width: int,
    height: int,
    data_dir: str | Path = "data",
) -> Path | None:
    """Find the best homography file for a specific camera.

    Checks in order:
    1. ``data/calibration.json`` (unified playfield calibration)
    2. ``data/homography-<slug>.json`` (legacy per-camera file)
    3. ``data/homography.json`` (legacy global fallback)

    Returns the path to the file, or ``None`` if nothing found.
    """
    # Prefer unified calibration file
    cal_file = calibration_path(data_dir)
    if cal_file.exists():
        try:
            data = json.loads(cal_file.read_text())
            cameras = data.get("cameras", {})
            for _key, cam_data in cameras.items():
                if cam_data.get("device_name") == device_name:
                    return cal_file
        except Exception:
            pass

    # Legacy per-camera file
    slug = camera_slug(device_name, width, height)
    per_camera = homography_path(slug, data_dir)
    if per_camera.exists():
        return per_camera

    # Legacy global fallback
    fallback = Path(data_dir) / "homography.json"
    if fallback.exists():
        return fallback
    return None


def compute_homography(pixel_pts: np.ndarray, world_pts_cm: np.ndarray) -> np.ndarray:
    # Use Direct Linear Transform via OpenCV. We require 4 points.
    H, mask = cv.findHomography(pixel_pts, world_pts_cm, method=0)
    if H is None:
        raise RuntimeError("Homography computation failed")
    return H


# ---------------------------------------------------------------------------
# Joint multi-tag calibration
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """Stores everything needed to undistort + homography-transform a frame.

    For cameras without barrel distortion, *camera_matrix* and
    *dist_coeffs* are ``None`` and ``undistort()`` is a no-op.
    """

    device_name: str
    resolution: Tuple[int, int]  # (width, height)
    homography: np.ndarray  # 3x3 pixel→world
    camera_matrix: Optional[np.ndarray] = None  # 3x3 intrinsics
    dist_coeffs: Optional[np.ndarray] = None  # (k1,k2,p1,p2,k3)
    tags_used: int = 0
    rms_error: float = 0.0

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
        )


def detect_all_tags(
    cap: cv.VideoCapture,
    num_frames: int = 30,
) -> Dict[int, np.ndarray]:
    """Detect AprilTags and ArUco 4x4 markers, return averaged centers.

    Accumulates detections over *num_frames* and averages pixel positions
    for stability.  ArUco 4x4 IDs are stored as negative numbers
    (-1, -2, -3, -4 for IDs 0-3) to avoid collision with AprilTag IDs.

    Returns:
        Dict mapping tag_id → (cx, cy) averaged pixel center.
    """
    d36 = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36H11)
    p36 = cv.aruco.DetectorParameters()
    det36 = cv.aruco.ArucoDetector(d36, p36)

    d4 = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    p4 = cv.aruco.DetectorParameters()
    det4 = cv.aruco.ArucoDetector(d4, p4)

    accum: Dict[int, List[np.ndarray]] = {}

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        corners36, ids36, _ = det36.detectMarkers(gray)
        if ids36 is not None:
            for c, tid in zip(corners36, ids36.flatten()):
                center = c.reshape(-1, 2).mean(axis=0)
                accum.setdefault(int(tid), []).append(center)

        corners4, ids4, _ = det4.detectMarkers(gray)
        if ids4 is not None:
            for c, tid in zip(corners4, ids4.flatten()):
                center = c.reshape(-1, 2).mean(axis=0)
                # Negative IDs for ArUco 4x4 to avoid collision
                accum.setdefault(-(int(tid) + 1), []).append(center)

    result: Dict[int, np.ndarray] = {}
    for tid, pts_list in accum.items():
        result[tid] = np.array(pts_list).mean(axis=0)
    return result


# ArUco 4x4 corner world positions (cm).
# Stored as negative IDs: -1=ArUco0=UL, -2=ArUco1=UR, -3=ArUco2=LL, -4=ArUco3=LR.
ARUCO_CORNER_WORLD: Dict[int, Tuple[float, float]] = {
    -1: (0.0, 0.0),      # ArUco 0 = upper-left
    -2: (101.0, 0.0),     # ArUco 1 = upper-right
    -3: (0.0, 89.0),      # ArUco 2 = lower-left
    -4: (101.0, 89.0),    # ArUco 3 = lower-right
}


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
        # Not enough points for distortion — plain homography
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
        Dict mapping device_name → CameraCalibration.
    """
    path = calibration_path(data_dir)
    data = json.loads(path.read_text())
    return {
        name: CameraCalibration.from_dict(cam_data)
        for name, cam_data in data.get("cameras", {}).items()
    }


# Legacy aliases for backward compatibility
def save_joint_calibration(
    bw_cal: CameraCalibration,
    color_cal: CameraCalibration,
    path: Path,
    field_width_cm: float = 101.0,
    field_height_cm: float = 89.0,
) -> None:
    """Save calibration (legacy — prefers :func:`save_calibration`)."""
    save_calibration(
        [bw_cal, color_cal],
        data_dir=path.parent,
        field_width_cm=field_width_cm,
        field_height_cm=field_height_cm,
    )


def load_joint_calibration(
    path: Path,
) -> Tuple[CameraCalibration, CameraCalibration]:
    """Load calibration (legacy — prefers :func:`load_calibration`)."""
    data = json.loads(path.read_text())
    cams = list(data.get("cameras", {}).values())
    if len(cams) < 2:
        raise ValueError(f"Expected at least 2 cameras in {path}")
    return (
        CameraCalibration.from_dict(cams[0]),
        CameraCalibration.from_dict(cams[1]),
    )


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


def run_once(cap: cv.VideoCapture) -> Optional[Dict[str, Tuple[float, float]]]:
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dets = detect_aruco_4x4(gray)
    corner_centers: Dict[str, Tuple[float, float]] = {}
    for pts, tid in dets:
        if tid in CORNER_ID_MAP:
            key = CORNER_ID_MAP[tid]
            c = choose_corner_point(pts)
            corner_centers[key] = (float(c[0]), float(c[1]))
    return corner_centers


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="homocal",
        description=(
            "Select input (camera or screen), compute pixel->field homography from four 4x4 ArUco corner tags, "
            "and write JSON with source metadata plus a PNG snapshot."
        ),
    )
    # Field spec
    parser.add_argument("--width", type=float, default=40.0, help="Field width (left->right). Default 40 (inches)")
    parser.add_argument("--height", type=float, default=35.0, help="Field height (upper->lower). Default 35 (inches)")
    parser.add_argument("--units", choices=["inch", "cm"], default="inch", help="Units for width/height input (default inch)")
    # Input selection (camera or screen)
    parser.add_argument("--screen", action="store_true", help="Use screen capture (primary display by default)")
    parser.add_argument("--screen-monitor", type=int, default=1, help="mss monitor index (1=primary, 0=all/virtual)")
    parser.add_argument("--screen-fps", type=float, default=30.0, help="Target FPS for screen capture")
    parser.add_argument("--screen-region", type=str, help="Optional region x,y,w,h within the chosen monitor")
    parser.add_argument("--camera", type=int, help="Camera index (if not using --screen)")
    parser.add_argument("--backend", type=str, choices=["auto", "avfoundation", "v4l2", "msmf", "dshow"], default="auto")
    parser.add_argument("--max-cams", type=int, default=10, help="Max camera indices to probe when resolving a camera")
    parser.add_argument("--cap-width", type=int, help="Optional camera capture width")
    parser.add_argument("--cap-height", type=int, help="Optional camera capture height")
    # Detection behavior baked into file (so aprilcam needn't carry flags)
    parser.add_argument("--detect-inverted", action="store_true", default=True, help="Enable detecting inverted tags (default on)")
    parser.add_argument("--no-detect-inverted", dest="detect_inverted", action="store_false")
    # Homography file
    parser.add_argument("--homography", type=str, default="homography.json", help="Output homography JSON filename (in data dir unless absolute)")
    parser.add_argument("--frames", type=int, default=30, help="Max frames to search for all four tags before failing")
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help=(
            "Delay in seconds before starting capture; prints a countdown. "
            "Defaults to 3s when --screen is used, otherwise 0."
        ),
    )
    args = parser.parse_args(argv)

    # Config and data dir
    cfg = AppConfig.load()

    # Resolve input
    cap = None
    source_meta: Dict[str, object]
    if args.screen:
        region = None
        if args.screen_region:
            try:
                parts = [int(p.strip()) for p in str(args.screen_region).split(",")]
                if len(parts) != 4:
                    raise ValueError
                region = (parts[0], parts[1], parts[2], parts[3])
            except Exception:
                print("Invalid --screen-region. Use x,y,w,h")
                return 2
        try:
            cap = ScreenCaptureMSS(monitor=int(args.screen_monitor), fps=float(args.screen_fps), region=region)
        except Exception as e:
            print(f"Failed to initialize screen capture: {e}")
            return 2
        source_meta = {
            "type": "screen",
            "monitor": int(args.screen_monitor),
            "fps": float(args.screen_fps),
            "region": list(region) if region else None,
        }
    else:
        # Camera path using AppConfig helpers
        cap = cfg.get_camera(arg=args.camera, backend=args.backend, max_cams=int(args.max_cams), quiet=True)
        if not cap or not cap.isOpened():
            print("Failed to open camera")
            return 2
        if args.cap_width:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, int(args.cap_width))
        if args.cap_height:
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(args.cap_height))
        source_meta = {
            "type": "camera",
            "index": int(args.camera) if args.camera is not None else None,
            "backend": str(args.backend),
            "cap_width": int(args.cap_width) if args.cap_width else None,
            "cap_height": int(args.cap_height) if args.cap_height else None,
        }

    try:
        # Optional start delay with countdown
        sleep_secs: float = 3.0 if (args.sleep is None and args.screen) else (float(args.sleep) if args.sleep is not None else 0.0)
        if sleep_secs > 0:
            total = int(math.floor(sleep_secs))
            frac = max(0.0, float(sleep_secs) - float(total))
            print(f"Starting calibration in {sleep_secs:.1f}s...")
            for i in range(total, 0, -1):
                print(f"{i}...")
                time.sleep(1.0)
            if frac > 0:
                time.sleep(frac)

        # Resolve output paths and prepare snapshot path
        out_path = Path(args.homography)
        if not out_path.is_absolute():
            out_path = cfg.data_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path = out_path.with_suffix('.png')
        # Capture an initial snapshot (raw). We'll overwrite with annotations later once detections exist.
        snap = None
        ok_snap = False
        try:
            ok_snap, snap = cap.read()
            if ok_snap and snap is not None:
                cv.imwrite(str(snap_path), snap)
                print(f"Saved snapshot to {snap_path}")
        except Exception:
            pass

        # Accumulate corner detections across frames
        found: Dict[str, Tuple[float, float]] = {}
        for _ in range(max(1, int(args.frames))):
            obs = run_once(cap) or {}
            found.update(obs)
            missing = [k for k in ("upper_left", "upper_right", "lower_left", "lower_right") if k not in found]
            if not missing:
                break
        if len(found) < 4:
            print("Did not observe all four corner tags. Move camera or ensure IDs 0..3 are visible.")
            return 3

        # Build correspondences. Pixel: detected centers. World: field coordinates in cm.
        field = FieldSpec(width_in=float(args.width), height_in=float(args.height), units=str(args.units))
        H, pixel_pts, world_pts_cm = calibrate_from_corners(found, field)

        # Save JSON with homography and source metadata
        out = {
            "units": "cm",
            "width_cm": field.width_cm,
            "height_cm": field.height_cm,
            "pixel_points": pixel_pts.tolist(),
            "world_points_cm": world_pts_cm.tolist(),
            "homography": H.tolist(),
            "note": "Maps [u,v,1]^T pixels to [X,Y,W]^T; use X/W,Y/W in centimeters.",
            "source": source_meta,
            "detect_inverted": bool(args.detect_inverted),
        }
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote homography to {out_path}")

        # Also save a per-camera named file when source is a camera
        if source_meta.get("type") == "camera":
            try:
                from ..camera.camutil import macos_avfoundation_device_names
                cam_idx = source_meta.get("index")
                cap_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                cap_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                av_names = macos_avfoundation_device_names()
                dev_name = av_names.get(cam_idx, f"camera-{cam_idx}") if cam_idx is not None else None
                if dev_name and cap_w and cap_h:
                    slug = camera_slug(dev_name, cap_w, cap_h)
                    per_cam_path = homography_path(slug, cfg.data_dir)
                    per_cam_path.write_text(json.dumps(out, indent=2))
                    print(f"Wrote per-camera homography to {per_cam_path}")
            except Exception as e:
                print(f"Warning: could not save per-camera homography: {e}")

        # Draw annotations on snapshot: fiducial centers, fiducial bounding boxes, and playfield boundary
        draw_img = None
        try:
            if ok_snap and snap is not None:
                draw_img = snap.copy()
            else:
                ok2, frm = cap.read()
                if ok2 and frm is not None:
                    draw_img = frm.copy()
        except Exception:
            draw_img = None

        if draw_img is not None:
            try:
                # Attempt detection on the draw frame to get boxes
                gray2 = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
                dets2 = detect_aruco_4x4(gray2)
                for pts, tid in dets2:
                    if tid in CORNER_ID_MAP:
                        pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
                        cv.polylines(draw_img, [pts_i], isClosed=True, color=(255, 0, 0), thickness=2)
                        c = pts.mean(axis=0)
                        cv.circle(draw_img, (int(c[0]), int(c[1])), 6, (0, 255, 0), -1)

                # Always draw centerpoints from the consolidated 'found' dict
                for name in ("upper_left", "upper_right", "lower_right", "lower_left"):
                    if name in found:
                        x, y = found[name]
                        cv.circle(draw_img, (int(x), int(y)), 5, (0, 200, 0), -1)

                # Draw playfield boundary polygon (UL -> UR -> LR -> LL)
                ul = found["upper_left"]
                ur = found["upper_right"]
                lr = found["lower_right"]
                ll = found["lower_left"]
                poly = np.array([ul, ur, lr, ll], dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(draw_img, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

                cv.imwrite(str(snap_path), draw_img)
                print(f"Updated annotated snapshot at {snap_path}")
            except Exception as e:
                # If annotation fails, keep the raw snapshot
                print(f"Warning: failed to annotate snapshot: {e}")

        # Headless: no preview windows
        return 0
    finally:
        try:
            cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
