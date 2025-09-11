from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set

import cv2 as cv
import numpy as np

from ..iohelpers import resolve_data_path, load_homography, get_data_dir
from ..aprilcam import build_detectors, detect_apriltags


def _warp_point(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    vec = np.array([float(u), float(v), 1.0], dtype=float)
    X = H @ vec
    if abs(X[2]) < 1e-9:
        return 0.0, 0.0
    return X[0] / X[2], X[1] / X[2]


def _parse_ids(s: Optional[str]) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcap",
        description=(
            "Detect AprilTags in a directory of images.\n"
            "Default: summarize detection rate per ID across images.\n"
            "Use --list to print detailed per-detection rows (filename, ID, px,py, wx,wy)."
        ),
    )
    parser.add_argument("--images", type=str, default="screencaps", help="Directory name under data/ or absolute path (default screencaps)")
    parser.add_argument("--homography", type=str, default="homography.json", help="Homography JSON (in data dir unless absolute)")
    parser.add_argument("--family", type=str, choices=["16h5", "25h9", "36h10", "36h11", "all"], default="36h11")
    parser.add_argument("--proc-width", type=int, default=0, help="Resize width for detection (0 = no downscale)")
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument("--quad-sigma", type=float, default=0.0)
    parser.add_argument("--corner-refine", type=str, choices=["none", "contour", "subpix"], default="subpix")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--april-min-wb-diff", type=float, default=3.0)
    parser.add_argument("--april-min-cluster-pixels", type=int, default=5)
    parser.add_argument("--april-max-line-fit-mse", type=float, default=20.0)
    parser.add_argument("--ids", type=str, help="Comma-separated list of expected IDs; if omitted, use union of detected IDs")
    parser.add_argument("--list", action="store_true", help="List per-detection rows (filename, ID, px,py, wx,wy)")
    args = parser.parse_args(argv)

    img_dir = Path(args.images)
    if not img_dir.is_absolute():
        # If the user specified a path under 'data/', treat it as relative to CWD to avoid data/data duplication
        if img_dir.parts and img_dir.parts[0] == 'data':
            img_dir = Path.cwd() / img_dir
        else:
            img_dir = get_data_dir() / img_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    H, H_meta, _ = load_homography(args.homography)
    if H is None:
        print("Homography missing; world coords will not be computed.")

    detectors = build_detectors(args.family, args.corner_refine, float(args.quad_decimate), float(args.quad_sigma), bool((H_meta or {}).get("detect_inverted", True)), float(args.april_min_wb_diff), int(args.april_min_cluster_pixels), float(args.april_max_line_fit_mse))

    # Iterate images (png/jpg)
    files = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
    if not files:
        print(f"No images in {img_dir}")
        return 0

    per_image_tags: Dict[str, Set[int]] = {}
    for path in files:
        img = cv.imread(str(path))
        if img is None:
            continue
        w = img.shape[1]
        scale = min(1.0, float(args.proc_width) / float(w)) if (args.proc_width and args.proc_width > 0 and w > 0) else 1.0
        dets = detect_apriltags(img, detectors, scale=scale, clahe=bool(args.clahe), sharpen=bool(args.sharpen))
        # Accumulate image-level set for summary
        tag_set: Set[int] = set()
        for pts, _raw, tid in dets:
            tag_set.add(int(tid))
            if args.list:
                c = pts.astype(np.float32).mean(axis=0)
                px, py = float(c[0]), float(c[1])
                if H is not None:
                    wx, wy = _warp_point(H, px, py)
                    print(f"{path.name},ID:{tid},px:{px:.1f},py:{py:.1f},wx:{wx:.1f},wy:{wy:.1f}")
                else:
                    print(f"{path.name},ID:{tid},px:{px:.1f},py:{py:.1f}")
        per_image_tags[path.name] = tag_set
    if args.list:
        return 0

    # Summary report: determine expected IDs and compute detection rate per ID
    expected = set(_parse_ids(args.ids))
    if not expected:
        # Use union of detected tags across images
        for s in per_image_tags.values():
            expected |= s
    if not expected:
        print("No tags detected in any image.")
        return 0

    total = len(files)
    # Union of detected IDs across all images (to flag IDs that never appear)
    present: Set[int] = set()
    for s in per_image_tags.values():
        present |= s
    for tid in sorted(expected):
        count = sum(1 for s in per_image_tags.values() if tid in s)
        pct = 100.0 * count / float(total) if total > 0 else 0.0
        note = "  (absent)" if tid not in present else ""
        print(f"ID {tid:3d}: {pct:4.1f}%  ({count}/{total}){note}")
    return 0
