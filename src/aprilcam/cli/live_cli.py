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
        "--clahe", action="store_true",
        help="Apply CLAHE contrast enhancement",
    )
    parser.add_argument(
        "--sharpen", action="store_true",
        help="Apply sharpening filter",
    )
    args = parser.parse_args(argv)

    from aprilcam.liveview import run_live_view

    run_live_view(
        camera_index=args.camera,
        video_path=args.video,
        loop=not args.no_loop,
        deskew=not args.no_deskew,
        family=args.family,
        detect_aruco=args.aruco,
        proc_width=args.proc_width,
        use_clahe=args.clahe,
        use_sharpen=args.sharpen,
    )
    return 0
