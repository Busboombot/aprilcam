from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np


def render_tag_image(tag_id: int, size: int = 800) -> np.ndarray:
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # OpenCV 4.7+ may provide generateImageMarker; older provides drawMarker
    img_gray: np.ndarray
    if hasattr(cv.aruco, "generateImageMarker"):
        img_gray = cv.aruco.generateImageMarker(dictionary, tag_id, size)
    else:
        img_gray = np.zeros((size, size), dtype=np.uint8)
        cv.aruco.drawMarker(dictionary, tag_id, size, img_gray, 1)
    return cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)


def add_quiet_zone(img: np.ndarray, quiet_ratio: float) -> np.ndarray:
    if quiet_ratio <= 0:
        return img
    h, w = img.shape[:2]
    pad = max(1, int(round(min(h, w) * float(quiet_ratio))))
    out = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    out[pad:pad + h, pad:pad + w] = img
    return out


def add_label_below(img: np.ndarray, label: str, margin: int = 20, font_scale: float = 1.0, thickness: int = 2) -> np.ndarray:
    # Compute text size
    (w, h), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 10
    new_h = img.shape[0] + margin + h + baseline + pad
    new_w = max(img.shape[1], w + 2 * pad)
    out = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
    # Center tag horizontally
    x_tag = (new_w - img.shape[1]) // 2
    out[0:img.shape[0], x_tag:x_tag + img.shape[1]] = img
    # Draw text centered
    x_text = (new_w - w) // 2
    y_text = img.shape[0] + margin + h
    cv.putText(out, label, (x_text, y_text), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv.LINE_AA)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AprilTag 36h11 images with optional quiet zone and labels.")
    parser.add_argument("--out-dir", default="april-tag-images", help="Output directory (default: april-tag-images)")
    parser.add_argument("--start", type=int, default=0, help="Start ID (inclusive), default 0")
    parser.add_argument("--end", type=int, default=586, help="End ID (inclusive), default 586 for 36h11")
    parser.add_argument("--size", type=int, default=800, help="Tag image size in pixels (square), default 800")
    parser.add_argument("--quiet-ratio", type=float, default=2.0/7.0, help="Quiet zone thickness as fraction of tag side per edge (default ~0.2857)")
    parser.add_argument("--with-label", action="store_true", help="Append an ID label below the tag (off by default)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DICT_APRILTAG_36h11 supports IDs 0..586
    start = max(args.start, 0)
    end = min(args.end, 586)
    if start > end:
        print("No IDs to generate (start > end)")
        return 1

    for tag_id in range(start, end + 1):
        tag_img = render_tag_image(tag_id, size=args.size)
        tag_img = add_quiet_zone(tag_img, float(args.quiet_ratio))
        if args.with_label:
            tag_img = add_label_below(tag_img, f"ID: {tag_id}", margin=30, font_scale=1.2, thickness=2)
        out_path = out_dir / f"tag36h11_{tag_id}.png"
        cv.imwrite(str(out_path), tag_img)
    print(f"Wrote images to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
