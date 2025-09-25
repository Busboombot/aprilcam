from __future__ import annotations

from typing import Optional, List

from ..aprilcam import AprilCam, build_detectors, detect_apriltags
from ..config import AppConfig
import cv2 as cv
import numpy as np
import json
from pathlib import Path


def _parse_id_list(s: str | None) -> list[int]:
    if not s:
        return []
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


def main(argv: Optional[List[str]] = None) -> int:
    # Full CLI supporting both interactive mode and offline evaluation over images.
    import argparse
    from ..iohelpers import resolve_data_path, load_homography, open_source_from_meta

    parser = argparse.ArgumentParser(prog="aprilcam", description="Detect AprilTags using input specified by homography JSON.")
    # Interactive/stream options
    parser.add_argument("--backend", type=str, choices=["auto", "avfoundation", "v4l2", "msmf", "dshow"], default="auto",
                        help="Capture backend to prefer for camera sources (homography may include backend)")
    parser.add_argument("--speed-alpha", type=float, default=0.3,
                        help="EMA smoothing factor for speed in [0,1]; higher is more responsive, lower is smoother")
    parser.add_argument("--quiet", action="store_true", help="Reduce OpenCV logging to suppress probe warnings")
    parser.add_argument("--family", type=str, choices=["16h5", "25h9", "36h10", "36h11", "all"], default="36h11",
                        help="AprilTag family to detect (fewer families = faster)")
    parser.add_argument("--proc-width", type=int, default=0,
                        help="Resize width for detection processing (0 = no downscale; downscales for speed, corners scaled back)")
    parser.add_argument("--quad-decimate", type=float, default=1.0, help="AprilTag quad decimate (>=1, larger is faster but less accurate)")
    parser.add_argument("--quad-sigma", type=float, default=0.0, help="AprilTag quad sigma (Gaussian blur) in pixels")
    parser.add_argument("--corner-refine", type=str, choices=["none", "contour", "subpix"], default="subpix",
                        help="Corner refinement method (subpix is slower but more accurate)")
    parser.add_argument("--use-aruco3", action="store_true", help="Use ArUco3 detection (if available)")
    parser.add_argument("--april-min-wb-diff", type=float, default=3.0,
                        help="Minimum white-black intensity difference (lower tolerates blur, may increase false positives)")
    parser.add_argument("--april-min-cluster-pixels", type=int, default=5,
                        help="Minimum connected-pixel cluster size (lower helps small/blurred tags)")
    parser.add_argument("--april-max-line-fit-mse", type=float, default=20.0,
                        help="Max line fit MSE (higher tolerates rotated/interpolated edges)")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="Run full detection every N frames; others use fast optical flow tracking (1 = detect every frame)")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement before detection")
    parser.add_argument("--sharpen", action="store_true", help="Apply unsharp mask before detection")
    parser.add_argument("--print-tags", action="store_true", help="Print per-tag ID, center, orientation, and velocity (fixed-width)")
    parser.add_argument("--homography", type=str, default="homography.json", help="Homography JSON filename (in data dir unless absolute)")
    parser.add_argument("--deskew-overlay", action="store_true", help="Warp the playfield quadrilateral to a rectangle in the overlay window")
    # Offline evaluation options
    parser.add_argument("--images", type=str, help="Directory of images to evaluate detection rates (optional)")
    parser.add_argument("--ids", type=str, help="Comma-separated list of expected IDs (optional)")

    args = parser.parse_args(argv)

    # Reduce OpenCV logging if requested
    if getattr(args, "quiet", False):
        try:
            if hasattr(cv, "utils") and hasattr(cv.utils, "logging"):
                cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

    if not args.images:
        # Interactive: open source from homography and run video
        H, H_meta, H_pix = load_homography(args.homography)
        if not H_meta:
            print("Homography file missing or missing 'source' config. Run homocal to generate it.")
            return 2
        # Backend mapping
        be_map = {
            "auto": None,
            "avfoundation": getattr(cv, "CAP_AVFOUNDATION", 1200),
            "v4l2": getattr(cv, "CAP_V4L2", 200),
            "msmf": getattr(cv, "CAP_MSMF", 1400),
            "dshow": getattr(cv, "CAP_DSHOW", 700),
        }
        be_value = be_map.get(args.backend)
        from ..iohelpers import open_source_from_meta
        cap = open_source_from_meta(H_meta, quiet=bool(args.quiet))
        if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
            print("No camera available.")
            return 1
        app = AprilCam(
            index=0,
            backend=be_value,
            speed_alpha=max(0.0, min(1.0, float(args.speed_alpha))),
            family=args.family,
            proc_width=int(args.proc_width),
            cap_width=None,
            cap_height=None,
            quad_decimate=float(args.quad_decimate),
            quad_sigma=float(args.quad_sigma),
            corner_refine=str(args.corner_refine),
            detect_inverted=bool((H_meta or {}).get("detect_inverted", True)),
            detect_interval=max(1, int(args.detect_interval)),
            use_clahe=bool(args.clahe),
            use_sharpen=bool(args.sharpen),
            april_min_wb_diff=float(args.april_min_wb_diff),
            april_min_cluster_pixels=int(args.april_min_cluster_pixels),
            april_max_line_fit_mse=float(args.april_max_line_fit_mse),
            print_tags=bool(args.print_tags),
            cap=cap,
            homography=H,
            headless=False,
            deskew_overlay=bool(args.deskew_overlay),
            playfield_poly_init=(
                np.array([H_pix[i] for i in (0, 1, 3, 2)], dtype=np.float32) if isinstance(H_pix, list) and len(H_pix) == 4 else None
            ),
        )
        app.run()
        return 0

    # Offline evaluation path
    img_dir = resolve_data_path(args.images)
    if not img_dir.exists():
        print(f"Images directory not found: {img_dir}")
        return 2
    H, H_meta, _ = load_homography(args.homography or "homography.json")
    detect_inverted = bool((H_meta or {}).get("detect_inverted", True))

    # Build default detectors (family 36h11, defaults as in aprilcam)
    detectors = build_detectors("36h11", "subpix", 1.0, 0.0, detect_inverted)
    files = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
    if not files:
        print(f"No images in {img_dir}")
        return 0
    total = len(files)
    counts: dict[int, int] = {}
    for path in files:
        img = cv.imread(str(path))
        if img is None:
            continue
        dets = detect_apriltags(img, detectors, scale=1.0, clahe=False, sharpen=False)
        seen = set()
        for _pts, _raw, tid in dets:
            if tid not in seen:
                counts[tid] = counts.get(tid, 0) + 1
                seen.add(tid)
    expected = set(_parse_id_list(args.ids))
    ids_all = sorted(set(counts.keys()) | expected)
    for tid in ids_all:
        c = counts.get(tid, 0)
        pct = 100.0 * c / float(total)
        mark = "*" if (not expected or tid in expected) else " "
        print(f"{mark} ID {tid:3d}: {c}/{total}  ({pct:5.1f}%)")
    return 0
