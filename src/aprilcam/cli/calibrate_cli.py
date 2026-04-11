"""CLI for running playfield calibration.

Usage:
    aprilcam calibrate                     # re-calibrate all cameras in calibration.json
    aprilcam calibrate 0 2                 # calibrate cameras at index 0 and 2
    aprilcam calibrate "Global Shutter"    # calibrate by name pattern
    aprilcam calibrate --width 101 --height 89   # override field dimensions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ..camera.camutil import list_cameras, select_camera_by_pattern
from ..config import AppConfig


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
        help="Output path for calibration.json (default: data/calibration.json)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to accumulate for tag detection (default: 30)",
    )
    args = parser.parse_args(argv)

    # Load config for data directory
    try:
        cfg = AppConfig.load()
        data_dir = cfg.data_dir
    except Exception:
        data_dir = Path("data")

    cal_path = Path(args.output) if args.output else data_dir / "calibration.json"

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

    # Run calibration for each camera
    from ..stream import calibrate

    for idx, label in camera_indices:
        print(f"Calibrating [{idx}] {label} ...")
        try:
            calibrate(
                camera=idx,
                field_width_cm=field_width,
                field_height_cm=field_height,
                output=cal_path,
                num_frames=args.frames,
            )
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    print("Done.")
    return 0
