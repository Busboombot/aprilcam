from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, List

import cv2 as cv

from ..iohelpers import resolve_data_path, load_homography, open_source_from_meta, get_data_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atscreencap",
        description="Capture N frames at fixed intervals using the source defined in the homography JSON.",
    )
    parser.add_argument("--homography", type=str, default="homography.json", help="Homography JSON filename (in data dir unless absolute)")
    parser.add_argument("--count", type=int, default=10, help="Number of frames to capture (default 10)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures (default 1.0)")
    parser.add_argument("--outdir", type=str, default="screencaps", help="Directory under data/ to place images (default screencaps)")
    args = parser.parse_args(argv)

    # Resolve output directory under ROOT/data
    data_out = get_data_dir() / args.outdir
    data_out.mkdir(parents=True, exist_ok=True)

    H_path = resolve_data_path(args.homography)
    print(f"Loading homography: {H_path}")
    H, H_meta, _ = load_homography(args.homography)
    if not H_meta:
        print("Homography file missing or missing 'source' config. Run homocal to generate it.")
        return 2

    # Describe source
    src_desc = "camera"
    if H_meta.get("type") == "screen":
        mon = H_meta.get("monitor", 1)
        region = H_meta.get("region")
        fps = H_meta.get("fps", 30.0)
        src_desc = f"screen (monitor {mon}, region {tuple(region) if region else 'full'}, fps {fps})"
    else:
        idx = H_meta.get("index")
        be = H_meta.get("backend")
        cw = H_meta.get("cap_width")
        ch = H_meta.get("cap_height")
        wh = f", {cw}x{ch}" if (cw and ch) else ""
        src_desc = f"camera (index {idx}, backend {be}{wh})"

    cap = open_source_from_meta(H_meta)
    if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
        print("Failed to open source from homography.")
        return 3

    print(f"Opened source: {src_desc}")
    print(f"Saving {int(args.count)} frames every {float(args.interval):.2f}s to {data_out}")

    # Capture loop
    ts0 = time.time()
    saved = 0
    for i in range(max(1, int(args.count))):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Capture failed at index {i}.")
            break
        out_path = data_out / f"cap_{i:04d}.png"
        cv.imwrite(str(out_path), frame)
        saved += 1
        print(f"[{i+1}/{int(args.count)}] Saved {out_path}")
        if i + 1 < int(args.count):
            # Sleep remaining interval accounting for time spent reading/writing
            dt = args.interval - (time.time() - ts0 - i * args.interval)
            if dt > 0:
                time.sleep(dt)

    # Clean up
    try:
        if hasattr(cap, "release"):
            cap.release()
    except Exception:
        pass

    print(f"Done. Saved {saved}/{int(args.count)} image(s) to {data_out}")
    return 0
