---
sprint: "011"
status: done
---

# Architecture Update -- Sprint 011: Library Performance & API

## What Changed

### Modified Files

- **`aprilcam/aprilcam.py`** — `detect_apriltags()` and
  `process_frame()` refactored to accept pre-converted grayscale,
  eliminating redundant `cvtColor` calls.
- **`aprilcam/playfield.py`** — `update()` accepts optional gray
  parameter; ArUco detector cached as instance attribute; corner
  re-detection throttled to every Nth frame.
- **`aprilcam/camutil.py`** — `CameraInfo` extended with `device_name`
  field. New `camera_slug()` function for homography file naming.
- **`aprilcam/homography.py`** — New `homography_path(device_name, resolution)`
  and `discover_homography(device_name, resolution)` functions for
  per-camera file persistence in `data/`.
- **`aprilcam/__init__.py`** — Exports `AprilCam`, `TagRecord`,
  `detect_tags`, `AprilTag`, `Playfield`.

### New Files

- **`aprilcam/stream.py`** — `detect_tags()` generator function:
  opens camera, auto-loads homography, yields `List[TagRecord]` per
  frame with velocity/orientation. Context manager support.
- **`aprilcam/errors.py`** — `CameraInUseError`, `CameraNotFoundError`,
  `CameraPermissionError` exception classes with contention diagnosis.

### New Conventions

- Homography files stored as `data/homography-<slug>-<WxH>.json`
  where slug is the slugified device name (e.g., `brio-501`).
- `data/homography.json` remains as global fallback.

## Why

Library consumers need a fast, simple API. The current interface
requires calling private methods and managing state manually. The
detection pipeline wastes CPU on redundant grayscale conversions.
Camera errors are silent, making multi-process debugging painful.

## Impact on Existing Components

- `detect_apriltags()` signature changes (adds optional `gray` param)
  — internal callers updated. MCP server unaffected (calls through
  `process_frame()` which handles the conversion).
- `Playfield.update()` signature changes (adds optional `gray` param)
  — backward compatible, gray is optional.
- Homography save path changes — new files use per-camera naming,
  but `load_homography()` falls back to global file for backward
  compat.

## Migration Concerns

- Existing `data/homography.json` files continue to work as fallback.
- No breaking changes to MCP tool signatures.
- `__init__.py` exports are additive (no removals).
