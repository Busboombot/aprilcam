---
id: '012'
title: Object Detection for Colored Cubes
status: done
branch: sprint/012-object-detection-for-colored-cubes
use-cases:
- SUC-012-001
- SUC-012-002
- SUC-012-003
---

# Sprint 012: Object Detection for Colored Cubes

## Goals

Detect colored cubes on the playfield at >40fps using dual-camera
fusion. The fast B&W camera (Arducam, ~49fps) provides precise
positions via contour detection. The slow color camera (HD USB,
~2fps) classifies colors via HSV thresholding. Results are fused
by world-coordinate proximity and color labels persist across
frames so the B&W camera can track objects at full speed.

## Problem

Detecting colored cubes currently requires the agent to open a
separate color camera, write custom HSV thresholding code, build
a separate homography, and do nearest-neighbor matching. This takes
3-5 seconds per cycle and is fragile (camera conflicts, false
positives, different homographies). The B&W camera alone can see
cubes as bright squares but can't determine their color.

## Solution

### Dual-Camera Fusion Architecture

1. **B&W camera (primary)**: Runs at full frame rate (~49fps).
   On each frame, detect bright square-shaped contours alongside
   AprilTags. Filter out tag regions and robot region. Report
   object centers in world coordinates. This is the **position
   authority**.

2. **Color camera (secondary)**: Runs asynchronously at its native
   rate (~2fps). On each frame, run HSV thresholding for each
   target color, detect contours, map to world coordinates via its
   own homography. This is the **color authority**.

3. **Fusion**: Match B&W objects to color objects by world-coordinate
   proximity (<5cm). Once a B&W object gets a color label, that
   label sticks until the object moves significantly or disappears.
   The color camera only needs to re-confirm periodically.

4. **Output**: `detect_tags()` returns tags as usual. A new
   `ObjectRecord` dataclass carries detected objects. The generator
   yields a `FrameResult` containing both `tags` and `objects`.

### Performance Target

- Tag detection + square detection on B&W: >40fps
- Color classification: runs in background, doesn't block main loop
- Fusion adds negligible overhead (world-coord matching)

## Success Criteria

- `detect_tags(camera=3, detect_objects=True, color_camera=2)` yields
  both tags and colored objects per frame at >40fps
- Objects include world_xy, color label, bounding box, and object type
- Color labels persist across frames without re-querying color camera
- AprilTags, ArUco corners, and robot body are excluded from objects
- Benchmark test confirms >40fps detection rate

## Scope

### In Scope

- `ObjectRecord` dataclass (world_xy, color, bbox, object_type, confidence)
- `FrameResult` dataclass wrapping `list[TagRecord]` and `list[ObjectRecord]`
- Square detection on B&W frames (contour-based, runs per frame)
- HSV color classification on color frames (configurable color ranges)
- Dual-camera fusion by world-coordinate proximity
- Color label persistence (sticky labels across frames)
- Tag/robot exclusion filtering
- `detect_objects` parameter on `detect_tags()`
- MCP tool: `get_objects(source_id)`
- Frame annotation: draw detected objects on annotated frames
- Benchmark test for >40fps
- Configurable HSV color ranges

### Out of Scope

- Ball detection (different shape heuristics — future sprint)
- Object identity tracking across pickup/dropoff (future sprint)
- Custom ML-based detection
- New camera hardware

## Test Strategy

- Unit tests for square detection on synthetic images
- Unit tests for HSV color classification on synthetic colored patches
- Unit tests for fusion logic (world-coord matching, label persistence)
- Unit tests for tag/robot exclusion filtering
- Integration test with real test images (if available in tests/data/)
- **Benchmark test**: `tests/bench_objects.py` — 100 frames from B&W
  camera, confirm >40fps with square detection enabled
- All 345 existing tests must continue to pass

## Architecture Notes

- `ObjectRecord` is a new frozen dataclass similar to `TagRecord`
- `FrameResult` wraps both tag and object lists; `detect_tags()`
  yields `FrameResult` which is backward-compatible (iterable as
  `list[TagRecord]` for existing callers)
- Square detection runs in `process_frame()` alongside tag detection
  — same grayscale, no extra conversion
- Color classification runs in a background thread on the color
  camera, updating a shared `ColorMap` dict
- Fusion happens at yield time in the generator, matching B&W
  objects to the latest color map

## GitHub Issues

(None linked.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [x] Sprint planning documents are complete
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
