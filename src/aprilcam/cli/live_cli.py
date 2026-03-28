"""CLI subcommand: aprilcam live — Open a live visualization window."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam live",
        description="Open a live camera view with tag detection overlays",
    )
    parser.add_argument(
        "-c", "--camera", type=int, default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "-v", "--video", type=str, default=None,
        help="Video file to use instead of a live camera (loops continuously)",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="Play video once instead of looping (only with --video)",
    )
    parser.add_argument(
        "--no-deskew", action="store_true",
        help="Disable perspective deskew (show raw camera view)",
    )
    parser.add_argument(
        "--family", default="36h11",
        help="AprilTag family (default: 36h11)",
    )
    parser.add_argument(
        "--aruco", action="store_true",
        help="Also detect ArUco 4x4 markers (in addition to AprilTags)",
    )
    parser.add_argument(
        "--proc-width", type=int, default=0,
        help="Processing width for detection downscale (0 = no downscale)",
    )
    parser.add_argument(
        "--no-highpass", action="store_true",
        help="Disable high-pass glare removal (enabled by default)",
    )
    parser.add_argument(
        "--clahe", action="store_true",
        help="Apply CLAHE contrast enhancement",
    )
    parser.add_argument(
        "--sharpen", action="store_true",
        help="Apply sharpening filter",
    )
    parser.add_argument(
        "--homography", type=str, default=None,
        help="Path to homography JSON file (default: auto-discover from data/)",
    )
    parser.add_argument(
        "--color-camera", type=int, default=None,
        help="Color camera index for object color classification (used with [d] key)",
    )
    args = parser.parse_args(argv)

    import json
    from pathlib import Path
    import numpy as np

    homography = None

    if args.homography:
        # Explicit path
        hom_path = Path(args.homography)
        if hom_path.is_file():
            try:
                data = json.loads(hom_path.read_text())
                homography = np.array(data["homography"], dtype=np.float64)
                print(f"Loaded homography from {hom_path}")
            except Exception as e:
                print(f"Warning: failed to load homography from {hom_path}: {e}")
    else:
        # Auto-discover per-camera homography file
        try:
            from aprilcam.camutil import get_device_name
            from aprilcam.homography import discover_homography
            import cv2 as cv

            dev_name = get_device_name(args.camera)
            # Probe camera resolution
            cap = cv.VideoCapture(args.camera)
            if cap.isOpened():
                w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                found = discover_homography(dev_name, w, h, "data")
                if found:
                    data = json.loads(found.read_text())
                    homography = np.array(data["homography"], dtype=np.float64)
                    print(f"Loaded homography from {found}")
            else:
                cap.release()
        except Exception as e:
            print(f"Warning: homography auto-discover failed: {e}")

    from aprilcam.liveview import run_live_view

    run_live_view(
        camera_index=args.camera,
        video_path=args.video,
        loop=not args.no_loop,
        deskew=not args.no_deskew,
        family=args.family,
        detect_aruco=args.aruco,
        proc_width=args.proc_width,
        use_highpass=not args.no_highpass,
        use_clahe=args.clahe,
        use_sharpen=args.sharpen,
        homography=homography,
        color_camera=args.color_camera,
    )
    return 0
