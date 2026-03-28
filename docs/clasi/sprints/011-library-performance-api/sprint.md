---
id: "011"
title: "Library Performance & API"
status: active
branch: sprint/011-library-performance-api
use-cases:
  - SUC-011-001
  - SUC-011-002
  - SUC-011-003
  - SUC-011-004
---

# Sprint 011: Library Performance & API

## Goals

Make AprilCam fast and easy to use as a Python library. Clients that
import aprilcam directly should be able to open a camera, load a
previously saved homography, and iterate over tag detections in a
single line of code — with world coordinates, velocities, and
orientations computed at maximum frame rate.

## Problem

1. **Homography is not persisted per camera** — only one global
   `data/homography.json` exists. Multi-camera setups lose calibration
   every session.
2. **Detection pipeline wastes CPU** — grayscale conversion happens 2-3
   times per frame; ArUco detector is rebuilt every call; playfield
   corner detection runs every frame even though corners barely move.
3. **No simple library API** — users must call private methods
   (`_init_capture()`) and manage state manually. No generator
   interface.
4. **Camera errors are silent** — OpenCV returns False with no
   explanation when a camera is in use by another process. No way to
   find out which process is contending.
5. **Nothing exported from `__init__.py`** — library users must know
   internal module paths.

## Solution

1. Per-camera homography files keyed by device name + resolution
   (e.g., `data/homography-brio-501-1920x1080.json`), with auto-
   discovery on camera open.
2. Eliminate redundant grayscale conversions, cache ArUco detector,
   throttle playfield corner re-detection to every Nth frame.
3. A `detect_tags()` generator function as the primary library API —
   opens camera, loads homography, yields tag records per frame.
4. Camera contention detection using `lsof` (macOS) / `fuser` (Linux)
   to identify the blocking process by PID and name.
5. Export key classes from `__init__.py`.

## Success Criteria

- `detect_tags(camera=0)` works out of the box with auto-loaded
  homography and yields tag records with world coordinates.
- Homography files persist per camera and survive restarts.
- Grayscale conversion happens exactly once per frame in the detection
  pipeline.
- Camera open failure due to contention reports the blocking process.
- `from aprilcam import AprilCam, detect_tags, TagRecord` works.

## Scope

### In Scope

- Per-camera homography file naming and auto-discovery
- Detection pipeline optimization (grayscale, detector caching,
  corner detection throttling)
- `detect_tags()` generator API
- Camera contention error reporting
- Public API exports in `__init__.py`
- Agent-facing documentation updates

### Out of Scope

- MCP server changes (beyond benefiting from shared improvements)
- Web UI changes
- New detection algorithms or families
- Streamable HTTP transport

## Test Strategy

- Unit tests for homography file naming/discovery logic
- Unit tests for the `detect_tags()` generator (using test images)
- Integration test verifying grayscale conversion count (mock cv2)
- Test camera contention error messages (mock subprocess output)
- Verify `__init__.py` exports with import checks

## Architecture Notes

- Device name for homography file key comes from ffmpeg
  (macOS, already implemented) or v4l2 (Linux). Slugified:
  lowercase, spaces→hyphens, strip special chars.
- `detect_tags()` is a module-level generator function in a new
  `aprilcam.stream` module (or directly in `__init__.py`).
- Grayscale optimization changes `detect_apriltags()` and
  `Playfield.update()` signatures to accept pre-converted gray.

## GitHub Issues

(None linked.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [x] Sprint planning documents are complete
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

- [ ] #001 — Per-camera homography file persistence
- [ ] #002 — Detection pipeline optimization
- [ ] #003 — detect_tags generator API (depends: 001, 002)
- [ ] #004 — Camera contention error reporting
- [ ] #005 — Public API exports and documentation (depends: 001-004)
