"""Benchmark object detection rate on a live camera.

Usage:
    uv run python tests/bench_objects.py [camera_index] [--color-camera INDEX]

Streams 100 frames with detect_objects=True, prints detected objects
per frame, and reports the overall FPS. Target: >40 FPS.
"""

import argparse
import sys
import time

from aprilcam import detect_tags

parser = argparse.ArgumentParser(description="Benchmark object detection FPS")
parser.add_argument("camera", type=int, nargs="?", default=0, help="B&W camera index")
parser.add_argument("--color-camera", type=int, default=None, help="Color camera index for color classification")
parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
args = parser.parse_args()

num_frames = args.frames
print(f"Camera {args.camera} — streaming {num_frames} frames with object detection...")
if args.color_camera is not None:
    print(f"Color camera: {args.color_camera}")

t0 = time.monotonic()
tag_counts = []
obj_counts = []

for i, result in enumerate(detect_tags(camera=args.camera, detect_objects=True, color_camera=args.color_camera)):
    tag_counts.append(len(result.tags))
    obj_counts.append(len(result.objects))

    parts = []
    for o in result.objects:
        parts.append(f"{o.color}@({o.center_px[0]:.0f},{o.center_px[1]:.0f})")
    tag_ids = [t.id for t in result.tags]
    print(f"  frame {i+1:3d}: {len(result.tags)} tags {tag_ids}, {len(result.objects)} objects {' '.join(parts) if parts else ''}")

    if i + 1 >= num_frames:
        break

elapsed = time.monotonic() - t0
fps = num_frames / elapsed

print(f"\n--- Results ---")
print(f"Frames:      {num_frames}")
print(f"Elapsed:     {elapsed:.2f}s")
print(f"FPS:         {fps:.1f}")
print(f"Tags:        {sum(tag_counts)} total, {sum(tag_counts)/num_frames:.1f} avg/frame")
print(f"Objects:     {sum(obj_counts)} total, {sum(obj_counts)/num_frames:.1f} avg/frame")
print(f"Target:      {'PASS' if fps >= 40 else 'FAIL'} (>40 FPS)")
