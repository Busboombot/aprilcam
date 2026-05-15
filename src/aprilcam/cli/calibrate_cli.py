"""CLI for running playfield calibration.

Usage:
    aprilcam calibrate                     # re-calibrate all cameras in calibration.json
    aprilcam calibrate 0 2                 # calibrate cameras at index 0 and 2
    aprilcam calibrate "Global Shutter"    # calibrate by name pattern
    aprilcam calibrate --width 101 --height 89   # override field dimensions
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

from ..camera.camutil import list_cameras, select_camera_by_pattern
from ..config import Config
from ..daemon.client import ensure_running, ControlClient


class _DaemonCapture:
    """Thin VideoCapture adapter that routes frame reads through the daemon.

    Exposes just enough of the cv.VideoCapture interface that
    :func:`~aprilcam.calibration.homography.detect_all_tags` and
    :func:`~aprilcam.calibration.calibration.calibrate_single` can use it
    without modification.
    """

    def __init__(self, client: ControlClient, cam_name: str) -> None:
        self._client = client
        self._cam_name = cam_name
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def _fetch_frame(self) -> Optional[np.ndarray]:
        """Fetch one JPEG frame from the daemon and decode it to BGR."""
        resp = self._client.rpc("capture_frame", cam_name=self._cam_name)
        data = base64.b64decode(resp["frame_b64"])
        frame = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
        if frame is not None and self._width is None:
            self._height, self._width = frame.shape[:2]
        return frame

    def read(self):
        """Mimic cv.VideoCapture.read() → (ret, frame)."""
        frame = self._fetch_frame()
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop_id: int) -> float:
        """Mimic cv.VideoCapture.get() for width/height props."""
        if prop_id == cv.CAP_PROP_FRAME_WIDTH:
            if self._width is None:
                self._fetch_frame()
            return float(self._width or 0)
        if prop_id == cv.CAP_PROP_FRAME_HEIGHT:
            if self._height is None:
                self._fetch_frame()
            return float(self._height or 0)
        return 0.0

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam calibrate",
        description="Run playfield calibration for one or more cameras.",
    )
    parser.add_argument(
        "cameras",
        nargs="*",
        help="Camera indices or name patterns to calibrate. "
        "If omitted, re-calibrates all cameras in calibration.json.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Playfield width in cm between ArUco corners (default: from calibration.json or 101.0)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Playfield height in cm between ArUco corners (default: from calibration.json or 89.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for calibration.json (default: from daemon config)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to accumulate for tag detection (default: 30)",
    )
    args = parser.parse_args(argv)

    # Start (or connect to) the daemon
    config = Config.load()
    client = ensure_running(config)

    # Resolve calibration save path: explicit --output takes priority,
    # then the daemon's configured path.
    if args.output:
        cal_path = Path(args.output)
    else:
        cal_path = Path(client.rpc("get_calibration_save_path")["path"])

    # Load existing calibration for defaults
    existing = {}
    if cal_path.exists():
        try:
            existing = json.loads(cal_path.read_text())
        except Exception:
            pass

    field_width = args.width or existing.get("field_width_cm", 101.0)
    field_height = args.height or existing.get("field_height_cm", 89.0)

    # Resolve which cameras to calibrate
    camera_indices: list[tuple[int, str]] = []  # (index, label)
    available = list_cameras()

    if args.cameras:
        # User specified cameras by index or name pattern
        for spec in args.cameras:
            idx = select_camera_by_pattern(spec, available)
            if idx is not None:
                label = next((c.device_name or c.name for c in available if c.index == idx), f"Camera {idx}")
                camera_indices.append((idx, label))
            else:
                print(f"  No camera matching '{spec}', skipping.")
    else:
        # Re-calibrate all cameras already in calibration.json
        existing_cameras = existing.get("cameras", {})
        if not existing_cameras:
            print(f"No cameras specified and {cal_path} has no existing entries.")
            print("Specify cameras to calibrate: aprilcam calibrate 0 2")
            return 1
        for device_name in existing_cameras:
            idx = select_camera_by_pattern(device_name, available)
            if idx is not None:
                camera_indices.append((idx, device_name))
            else:
                print(f"  Camera '{device_name}' from calibration.json not found, skipping.")

    if not camera_indices:
        print("No cameras to calibrate.")
        return 1

    print(f"Playfield: {field_width} x {field_height} cm")
    print(f"Output: {cal_path}")
    print(f"Cameras to calibrate: {len(camera_indices)}")
    for idx, label in camera_indices:
        print(f"  [{idx}] {label}")
    print()

    # Run calibration for each camera via daemon
    from ..calibration.calibration import calibrate_single, save_calibration

    for idx, label in camera_indices:
        print(f"Calibrating [{idx}] {label} ...")
        try:
            # Open the camera through the daemon
            resp = client.rpc("open_camera", index=idx)
            cam_name = resp["cam_name"]

            # Warm-up: discard initial frames (mirror stream.calibrate warm-up of 10 frames)
            for _ in range(10):
                client.rpc("capture_frame", cam_name=cam_name)

            # Build a VideoCapture-compatible adapter backed by the daemon
            cap = _DaemonCapture(client, cam_name)

            cal = calibrate_single(
                cap,
                field_width_cm=field_width,
                field_height_cm=field_height,
                num_frames=args.frames,
                camera_index=idx,
            )

            # Merge into existing calibration file
            if cal_path.exists():
                try:
                    cal_data = json.loads(cal_path.read_text())
                except Exception:
                    cal_data = {}
            else:
                cal_data = {}
            cal_data["type"] = "playfield"
            cal_data["field_width_cm"] = field_width
            cal_data["field_height_cm"] = field_height
            if "cameras" not in cal_data:
                cal_data["cameras"] = {}
            cal_data["cameras"][cal.device_name] = cal.to_dict()
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            cal_path.write_text(json.dumps(cal_data, indent=2))

            print(f"Calibration saved to {cal_path}")
            print(f"  Camera: {cal.device_name} {cal.resolution}, {cal.tags_used} tags, RMS {cal.rms_error:.6f}")
            if cal.dist_coeffs is not None:
                print(f"  Barrel distortion correction: yes")

            # Notify the daemon so it hot-reloads the calibration
            client.rpc("reload_calibration", cam_name=cam_name)

            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    print("Done.")
    return 0
