---
id: "003"
title: "Playfield & Homography"
status: planning
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

**Integration tests** (`tests/test_mcp_playfield.py`):
- Mock the camera registry to return synthetic frames.
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

## GitHub Issues

(None linked yet.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
