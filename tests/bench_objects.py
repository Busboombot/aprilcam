"""Benchmark tag detection + one-shot object detection.

Usage:
    uv run python tests/bench_objects.py [camera_index]

Streams 100 frames for tag detection (measuring FPS), then runs
one-shot object detection on the last frame and reports results.
"""

import argparse
import time

import cv2 as cv
import numpy as np

from aprilcam import detect_tags
from aprilcam.objects import SquareDetector

parser = argparse.ArgumentParser(description="Benchmark tag + object detection")
parser.add_argument("camera", type=int, nargs="?", default=0, help="Camera index")
parser.add_argument("--frames", type=int, default=100, help="Number of frames")
args = parser.parse_args()

num_frames = args.frames
print(f"Camera {args.camera} — streaming {num_frames} frames (tags only)...")

t0 = time.monotonic()
tag_counts = []
last_tags = []

for i, tags in enumerate(detect_tags(camera=args.camera)):
    tag_counts.append(len(tags))
    last_tags = tags
    ids = [t.id for t in tags]
    if i % 20 == 0:
        print(f"  frame {i+1:3d}: {len(tags)} tags {ids}")
    if i + 1 >= num_frames:
        break

elapsed = time.monotonic() - t0
fps = num_frames / elapsed

print(f"\n--- Tag Detection ---")
print(f"Frames:    {num_frames}")
print(f"Elapsed:   {elapsed:.2f}s")
print(f"FPS:       {fps:.1f}")
print(f"Target:    {'PASS' if fps >= 40 else 'FAIL'} (>40 FPS)")

# One-shot object detection on camera
print(f"\n--- Object Detection (one-shot) ---")
cap = cv.VideoCapture(args.camera)
for _ in range(5):
    cap.read()
ret, frame = cap.read()
cap.release()

if ret:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    det = SquareDetector()
    t1 = time.monotonic()
    objects = det.detect(gray)
    dt = (time.monotonic() - t1) * 1000
    print(f"Objects:   {len(objects)} detected in {dt:.1f}ms")
    for o in objects:
        print(f"  ({o.center_px[0]:.0f}, {o.center_px[1]:.0f}) area={o.area_px:.0f} {o.bbox[2]}x{o.bbox[3]}")
else:
    print("Failed to capture frame for object detection")
