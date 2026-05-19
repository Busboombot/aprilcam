"""CLI command: aprilcam tags — detect and list all tags on a camera."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _load_homography(cam_name: str, data_dir: Path) -> Optional[np.ndarray]:
    """Try to load a 3x3 homography matrix from the per-camera calibration.json."""
    cal_file = data_dir / "cameras" / cam_name / "calibration.json"
    if not cal_file.exists():
        return None
    try:
        data = json.loads(cal_file.read_text())
        h = data.get("homography")
        if h is not None:
            mat = np.array(h, dtype=float)
            if mat.shape == (3, 3):
                return mat
    except Exception:
        pass
    return None


def _px_to_world(
    px_points: List[Tuple[float, float]],
    H: np.ndarray,
) -> List[Tuple[float, float]]:
    """Apply homography H to a list of pixel (x, y) points → world (x, y)."""
    import cv2 as cv
    pts = np.array([[p] for p in px_points], dtype=np.float32)
    world = cv.perspectiveTransform(pts, H)
    return [(float(w[0][0]), float(w[0][1])) for w in world]


def _print_table(
    title: str,
    rows: List[dict],
    has_world: bool,
) -> None:
    """Print a formatted table for one tag family."""
    print(title)

    if not rows:
        print("  (none detected)\n")
        return

    # Column headers
    if has_world:
        hdr = f"  {'ID':>5}  {'px_x':>8}  {'px_y':>8}  {'wx_cm':>8}  {'wy_cm':>8}"
        sep = "  " + "-" * 5 + "  " + "-" * 8 + "  " + "-" * 8 + "  " + "-" * 8 + "  " + "-" * 8
    else:
        hdr = f"  {'ID':>5}  {'px_x':>8}  {'px_y':>8}"
        sep = "  " + "-" * 5 + "  " + "-" * 8 + "  " + "-" * 8

    print(hdr)
    print(sep)
    for r in rows:
        if has_world and r.get("wx") is not None:
            print(
                f"  {r['label']:>5}  {r['px'][0]:8.1f}  {r['px'][1]:8.1f}"
                f"  {r['wx']:8.1f}  {r['wy']:8.1f}"
            )
        else:
            wx_col = "  {:>8}  {:>8}".format("—", "—") if has_world else ""
            print(f"  {r['label']:>5}  {r['px'][0]:8.1f}  {r['px'][1]:8.1f}{wx_col}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam tags",
        description="Detect ArUco and AprilTag markers on a camera and print locations.",
    )
    parser.add_argument(
        "camera", type=int,
        help="Camera index to use",
    )
    parser.add_argument(
        "--frames", type=int, default=30,
        help="Number of frames to accumulate detections over (default: 30)",
    )
    args = parser.parse_args(argv)

    import cv2 as cv
    from ..calibration.homography import detect_all_tags
    from ..camera.camutil import get_device_name
    from ..calibration.calibration import device_name_slug
    from ..config import Config

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}")
        return 1

    time.sleep(0.3)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    try:
        tags = detect_all_tags(cap, args.frames)
    finally:
        cap.release()

    # Try to load homography for world coordinate projection
    H: Optional[np.ndarray] = None
    cam_label = f"Camera {args.camera}"
    try:
        config = Config.load()
        device = get_device_name(args.camera)
        if device:
            cam_name = device_name_slug(device)
            cam_label = device
            H = _load_homography(cam_name, config.data_dir)
    except Exception:
        pass

    aruco_raw = {tid: px for tid, px in tags.items() if tid < 0}
    april_raw = {tid: px for tid, px in tags.items() if tid > 0}

    print(f"\n{cam_label} ({width}×{height}) — {len(tags)} tag(s) detected\n")

    # Build ArUco rows
    aruco_ids = sorted(aruco_raw)
    aruco_px = [aruco_raw[t] for t in aruco_ids]
    aruco_world = _px_to_world(aruco_px, H) if H is not None else None
    aruco_rows = []
    for i, tid in enumerate(aruco_ids):
        aruco_id = -(tid + 1)
        row: dict = {"label": str(aruco_id), "px": aruco_px[i], "wx": None, "wy": None}
        if aruco_world:
            row["wx"], row["wy"] = aruco_world[i]
        aruco_rows.append(row)

    # Build AprilTag rows
    april_ids = sorted(april_raw)
    april_px = [april_raw[t] for t in april_ids]
    april_world = _px_to_world(april_px, H) if H is not None else None
    april_rows = []
    for i, tid in enumerate(april_ids):
        row = {"label": str(tid), "px": april_px[i], "wx": None, "wy": None}
        if april_world:
            row["wx"], row["wy"] = april_world[i]
        april_rows.append(row)

    has_world = H is not None
    world_note = " (no calibration — world coords unavailable)" if not has_world else " (cm)"

    _print_table(f"ArUco 4×4{world_note}:", aruco_rows, has_world)
    _print_table(f"AprilTag 36h11{world_note}:", april_rows, has_world)

    return 0
