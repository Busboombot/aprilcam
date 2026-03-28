"""Benchmark tag detection rate on a live camera.

Usage:
    uv run python tests/bench_tags.py [camera_index]

Streams 100 frames from the specified camera (default 0),
prints detected tags per frame, and reports the overall FPS.
"""

import sys
import time

from aprilcam import detect_tags

camera = int(sys.argv[1]) if len(sys.argv) > 1 else 0
num_frames = 100

print(f"Camera {camera} — streaming {num_frames} frames...")

t0 = time.monotonic()
tag_counts = []

for i, tags in enumerate(detect_tags(camera=camera)):
    tag_counts.append(len(tags))
    if tags:
        parts = []
        for t in tags:
            spd = f"{t.speed_px:.0f}px/s" if t.speed_px else "still"
            parts.append(f"{t.id}:{spd}")
        print(f"  frame {i+1:3d}: {' '.join(parts)}")
    else:
        print(f"  frame {i+1:3d}: no tags")
    if i + 1 >= num_frames:
        break

elapsed = time.monotonic() - t0
fps = num_frames / elapsed
total_tags = sum(tag_counts)
avg_tags = total_tags / num_frames

print(f"\n--- Results ---")
print(f"Frames:    {num_frames}")
print(f"Elapsed:   {elapsed:.2f}s")
print(f"FPS:       {fps:.1f}")
print(f"Tags seen: {total_tags} total, {avg_tags:.1f} avg/frame")
