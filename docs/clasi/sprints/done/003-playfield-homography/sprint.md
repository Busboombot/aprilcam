---
id: '003'
title: Playfield & Homography
status: done
branch: sprint/003-playfield-homography
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 003: Playfield & Homography

## Goals

Expose the existing playfield detection and homography calibration
capabilities as MCP tools so that an LLM agent can establish a playfield,
obtain deskewed (top-down) views, and optionally calibrate pixel
coordinates to real-world units -- all without interactive CLI usage.

1. `create_playfield(camera_id)` -- detect four ArUco 4x4 corner markers,
   establish the playfield polygon, return a `playfield_id`.
2. Transparent deskew -- any tool that accepts a `camera_id` also accepts
   a `playfield_id`; captures through a playfield automatically return
   the perspective-corrected (top-down) image.
3. Pixel-only deskew -- when no real-world measurements are provided,
   compute a pixel-only homography from the four detected corners to
   produce a rectangular top-down view.
4. `calibrate_playfield(playfield_id, measurements)` -- accept real-world
   distances between corner markers and compute a pixel-to-centimeter
   homography matrix.
5. `get_playfield_info(playfield_id)` -- return polygon corners,
   calibration status, and field dimensions.
6. Refactor `playfield.py` and `homography.py` internals so the core
   logic is callable from MCP tool handlers without subprocess or CLI
   dependencies.

## Problem

The playfield detection (`playfield.py`) and homography calibration
(`homography.py`) currently require interactive CLI invocation with
argparse, camera handles, and GUI windows. An LLM agent using the MCP
server cannot call these workflows. The agent needs to:

- Point a camera at a surface with four ArUco corner markers and get
  back a stable, deskewed top-down image for visual reasoning.
- Optionally supply physical measurements to convert pixel positions
  to centimeter coordinates.
- Query the playfield state (corners, calibration, dimensions) to
  plan robot actions.

## Solution

1. **Playfield registry** -- Add a `PlayfieldRegistry` (or extend the
   existing camera registry pattern from Sprint 002) that maps
   `playfield_id` strings to `(camera_id, Playfield, homography_matrix)`
   tuples. A playfield wraps a camera: it captures a frame, runs ArUco
   detection, and caches the polygon.

2. **Library-level API** -- Extract the core logic from `homography.py`'s
   `main()` and `Playfield.update()` into standalone functions that
   accept a BGR frame and return structured results (polygon corners,
   homography matrix, deskewed image). No argparse, no cv.imshow, no
   file I/O.

3. **MCP tool handlers** -- Register three new tools on the MCP server:
   - `create_playfield` -- grabs frames from `camera_id`, calls
     `Playfield.update()` until the polygon locks, stores in registry.
   - `calibrate_playfield` -- takes measurements dict
     `{width, height, units}`, calls `compute_homography()`, stores
     the matrix alongside the playfield.
   - `get_playfield_info` -- reads registry, returns JSON with corners,
     calibration status, field dimensions.

4. **Playfield-as-camera** -- When a `playfield_id` is passed where a
   `camera_id` is expected, the server captures from the underlying
   camera and applies `Playfield.deskew()` before returning the image.

## Success Criteria

- An MCP client can call `create_playfield("cam_0")` and receive a
  `playfield_id` when four ArUco 4x4 markers (IDs 0-3) are visible.
- Capturing an image via the playfield_id returns a deskewed top-down
  rectangle even without calibration measurements.
- After calling `calibrate_playfield` with width/height/units, the
  `get_playfield_info` response includes `calibrated: true` and the
  field dimensions in centimeters.
- All existing CLI tools (`homocal`, `aprilcam`, `playfield`) continue
  to work unchanged.
- Unit tests cover polygon ordering, deskew output dimensions, and
  homography computation with synthetic data.

## Scope

### In Scope

- `create_playfield` MCP tool
- `calibrate_playfield` MCP tool
- `get_playfield_info` MCP tool
- Playfield-as-camera pass-through for existing capture tools
- Refactoring `playfield.py` to separate detection logic from dataclass
- Refactoring `homography.py` to separate calibration logic from CLI
- Unit tests for core functions (polygon ordering, deskew, homography)
- Integration test: create playfield from a synthetic frame with ArUco
  markers

### Out of Scope

- Continuous tag detection loop (Sprint 004)
- Multi-camera compositing (Sprint 006)
- Saving/loading playfield state to disk (future sprint)
- Camera intrinsic calibration (lens distortion correction)
- Non-ArUco marker support (AprilTag 36h11 corners)

## Test Strategy

**Unit tests** (`tests/test_playfield.py`, `tests/test_homography.py`):
- Generate synthetic frames with four ArUco 4x4 markers at known pixel
  positions using `cv2.aruco.generateImageMarker()`.
- Verify `_order_poly` produces UL, UR, LR, LL order regardless of
  input ID-to-position mapping.
- Verify `deskew()` output dimensions match expected rectangle.
- Verify `compute_homography()` maps known pixel points to expected
  world coordinates within tolerance.
- Verify `FieldSpec` unit conversions (inch to cm).

**Real-image tests** (`tests/test_playfield_real.py`):
- Mock the camera to return captured test images from `tests/data/`
  (`playfield_cam3.jpg`, `playfield_cam3_moved.jpg`) instead of a
  live camera feed.
- Verify ArUco corner detection finds all 4 corners (IDs 0-3) in
  both images.
- Verify `create_playfield` succeeds and returns a valid polygon from
  the real images.
- Verify `compute_homography()` against `data/homography.json` — the
  computed matrix from detected corners should match the reference
  within tolerance.
- Verify `deskew()` produces a rectangular top-down image from the
  real playfield capture.
- Verify AprilTag detection finds the expected tags (IDs 0-6, 30 in
  the first image) after deskew.

**Integration tests** (`tests/test_mcp_playfield.py`):
- Mock the camera registry to return captured test images from
  `tests/data/` (not synthetic — real playfield frames).
- Call `create_playfield` tool handler and verify registry state.
- Call `calibrate_playfield` and verify homography is stored.
- Call `get_playfield_info` and verify JSON schema.
- Call a capture tool with a playfield_id and verify the returned
  image is deskewed (dimensions match expected output).

## Architecture Notes

- The `Playfield` dataclass currently caches the polygon after first
  successful detection and never re-detects. This is intentional for
  Sprint 003 -- the playfield is assumed static. Dynamic re-detection
  is deferred.
- The homography matrix stored after calibration uses the same
  convention as `homography.py`: maps `[u, v, 1]` pixel coordinates
  to `[X, Y, W]` world coordinates in centimeters (divide by W).
- Playfield IDs follow the pattern `pf_{camera_id}` (e.g., `pf_cam_0`).
  Only one playfield per camera is supported in this sprint.
- The `PlayfieldRegistry` is a simple dict in the server module; no
  persistence across server restarts.

### Homography File Specification (`data/homography.json`)

The `calibrate_playfield` tool (and the existing `homocal` CLI) persist
calibration results to `data/homography.json`. This file is the
canonical on-disk representation of a playfield calibration and is
loaded by downstream tools (detection loop, CLI viewers) at startup.

**Schema:**

```json
{
  "units": "cm",
  "width_cm": 102.0,
  "height_cm": 89.0,
  "pixel_points": [
    [269.25, 91.0],
    [961.25, 94.75],
    [255.0, 692.0],
    [954.25, 711.0]
  ],
  "world_points_cm": [
    [0.0, 0.0],
    [102.0, 0.0],
    [0.0, 89.0],
    [102.0, 89.0]
  ],
  "homography": [
    [0.1528, 0.0036, -41.47],
    [-0.0008, 0.1514, -13.56],
    [3.65e-05, 1.89e-05, 1.0]
  ],
  "note": "Maps [u,v,1]^T pixels to [X,Y,W]^T; use X/W,Y/W in centimeters.",
  "source": {
    "type": "camera",
    "index": 1,
    "backend": "auto",
    "cap_width": null,
    "cap_height": null
  },
  "detect_inverted": true
}
```

**Field definitions:**

| Field | Type | Description |
|-------|------|-------------|
| `units` | `str` | Always `"cm"` for now |
| `width_cm` | `float` | Physical width of the playfield in centimeters (UL→UR distance) |
| `height_cm` | `float` | Physical height of the playfield in centimeters (UL→LL distance) |
| `pixel_points` | `float[4][2]` | Detected ArUco corner centers in pixel coords, order: UL, UR, LL, LR |
| `world_points_cm` | `float[4][2]` | Corresponding world coordinates in cm, order: UL, UR, LL, LR |
| `homography` | `float[3][3]` | 3×3 projective matrix mapping `[u,v,1]` → `[X,Y,W]`; world = `(X/W, Y/W)` cm |
| `note` | `str` | Human-readable description of the homography convention |
| `source` | `object` | Camera source metadata (type, index, backend, resolution hints) |
| `detect_inverted` | `bool` | Whether inverted marker detection was enabled during calibration |

**Corner ordering:** `pixel_points` and `world_points_cm` use the same
index order — `[0]` = upper-left, `[1]` = upper-right, `[2]` = lower-left,
`[3]` = lower-right. This matches the `CORNER_ID_MAP` in `homography.py`.

**Loading:** The MCP server should load this file at startup (if it exists)
to pre-populate a calibrated playfield, so that an agent reconnecting to
the server doesn't need to re-calibrate.

### Test Image

A reference playfield image is saved at `tests/data/playfield_cam3.jpg`
(captured from Camera 3 — the B&W overhead camera). This image contains
all 4 ArUco corner markers (IDs 0-3) and 8 AprilTags (IDs 0-6, 30) and
should be used for offline homography and detection tests.

## GitHub Issues

(None linked yet.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
