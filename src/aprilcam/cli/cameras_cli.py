from __future__ import annotations

import argparse
from typing import List, Optional

from ..camutil import list_cameras, default_backends, select_camera_by_pattern, macos_avfoundation_device_names
from ..config import AppConfig


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cameras",
        description="List available cameras and suggest index by .env CAMERA pattern. On macOS, AVFoundation is only probed for indices 0-1; use CAP_ANY to find additional devices.")
    parser.add_argument("--max-cams", type=int, default=10, help="Maximum camera indices to probe (default: 10)")
    parser.add_argument("--backend", type=str, choices=["auto", "avfoundation", "v4l2", "msmf", "dshow"], default="auto")
    parser.add_argument("--pattern", type=str, help="Pattern to match camera name (overrides .env CAMERA)")
    parser.add_argument("--quiet", action="store_true", help="Reduce OpenCV logging noise")
    parser.add_argument("--details", action="store_true", help="On macOS, use ffmpeg avfoundation names if available")
    parser.add_argument("--stop-after-failures", type=int, default=4, help="Per-backend consecutive failure cutoff to reduce noise (default 4)")
    args = parser.parse_args(argv)

    # Attempt to read .env for CAMERA pattern; tolerant to missing guard in this tool
    pattern = None
    try:
        cfg = AppConfig.load()
        pattern = cfg.env.get("CAMERA")
    except Exception:
        pass
    if args.pattern:
        pattern = args.pattern

    # Quiet logging if requested
    if args.quiet:
        try:
            import cv2 as cv
            if hasattr(cv, "utils") and hasattr(cv.utils, "logging"):
                cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

    be_map = {
        "auto": None,
        "avfoundation": 1200,
        "v4l2": 200,
        "msmf": 1400,
        "dshow": 700,
    }
    be = be_map.get(args.backend)
    backends = default_backends() if be is None else [be]
    # For avfoundation, probing many indices can be noisy; if default, reduce
    max_probe = args.max_cams
    if args.backend == "avfoundation" and args.max_cams == 10:
        max_probe = 2

    cams = list_cameras(max_probe, backends=backends, stop_after_failures=int(args.stop_after_failures), quiet=bool(args.quiet), detailed_names=bool(args.details))
    print("Cameras:")
    av_names = macos_avfoundation_device_names() if args.details else {}
    for c in cams:
        label = c.name
        # If AVFoundation and a better name exists for that index, append it
        if av_names and c.backend == "AVFOUNDATION" and c.index in av_names:
            label = f"[{c.index}] {av_names[c.index]} (index {c.index}, AVFOUNDATION)"
        else:
            label = f"[{c.index}] {label}"
        print(f"  {label}")
    if not cams:
        print("  (none found)")

    chosen = select_camera_by_pattern(pattern, cams) if pattern else None
    if chosen is not None:
        print(f"Suggested index by pattern '{pattern}': {chosen}")
    else:
        if pattern:
            print(f"No camera matched pattern '{pattern}'.")

    return 0
