---
sprint: "003"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 003: Playfield & Homography

## What Changed

### New: PlayfieldRegistry (MCP server module)

A server-side registry mapping `playfield_id` strings to playfield
state. Each entry holds:

```
PlayfieldEntry:
  playfield_id: str          # e.g., "pf_cam_0"
  camera_id: str             # underlying camera
  playfield: Playfield       # locked polygon instance
  field_spec: FieldSpec?     # set after calibration
  homography: np.ndarray?    # 3x3 matrix, set after calibration
```

The registry is an in-memory dict in the MCP server module, following
the same pattern as the camera registry from Sprint 002. No persistence
across server restarts.

### New: Three MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| `create_playfield` | `camera_id: str`, `max_frames: int = 30` | `{playfield_id, corners, calibrated}` |
| `calibrate_playfield` | `playfield_id: str`, `measurements: {width, height, units}` | `{playfield_id, calibrated, width_cm, height_cm}` |
| `get_playfield_info` | `playfield_id: str` | `{playfield_id, camera_id, corners, calibrated, width_cm?, height_cm?, homography?}` |

### Modified: Capture tool routing (MCP server)

The existing capture tool (e.g., `capture_image`) gains awareness of
playfield IDs. When the `camera_id` parameter matches a registered
`playfield_id`, the server:

1. Resolves the underlying `camera_id` from the playfield entry.
2. Captures a raw frame from that camera.
3. Applies `Playfield.deskew()` to produce the top-down view.
4. Returns the deskewed image instead of the raw frame.

This is implemented as a resolution step at the top of the capture
handler, not a separate code path.

### Modified: `src/aprilcam/homography.py`

- New function: `calibrate_from_corners(pixel_corners, field_spec)`
  that takes four pixel corner positions (UL, UR, LR, LL) and a
  `FieldSpec`, builds the world-coordinate correspondences, and
  returns the 3x3 homography matrix. This extracts the core logic
  from `main()` lines 218-232.
- `FieldSpec` remains in `homography.py` (no move needed; it has no
  CLI dependencies).
- `main()` is refactored to call `calibrate_from_corners()` internally.
- `detect_aruco_4x4()` and `compute_homography()` are unchanged.

### Modified: `src/aprilcam/playfield.py`

- `_detect_corners()` and `_order_poly()` remain as methods on
  `Playfield` but are documented as safe to call from non-GUI contexts
  (they already are -- no changes needed to the methods themselves).
- The `Playfield` constructor gains an optional `polygon` parameter
  to allow direct initialization with a known polygon (for testing
  and for the MCP tool to inject a pre-detected polygon).
- `deskew()` is unchanged; it already works as a pure function on
  a frame + cached polygon.

### Not Changed

- `src/aprilcam/display.py` -- `PlayfieldDisplay` is a GUI-only class
  and is not used by the MCP server. No modifications.
- `src/aprilcam/models.py` -- `AprilTag` and `AprilTagFlow` are
  unchanged.
- CLI entry points (`homocal`, `playfield`, `aprilcam`) -- continue
  to work as wrappers. Their internal calls route through the same
  refactored library functions.

## Why

Sprint 002 established the MCP server with camera tools (open, capture,
list). The natural next step is to let the agent establish a playfield
on top of a camera -- this is the primary workflow for robotics use
cases where the agent needs a stable, calibrated top-down view of a
playing surface.

The existing `playfield.py` and `homography.py` already contain all
the computer vision logic. The gap is purely at the interface level:
the code is locked behind CLI `main()` functions with argparse and
file I/O. This sprint extracts library-level functions and wires them
into MCP tool handlers.

## Impact on Existing Components

### MCP Server Module

- Gains `PlayfieldRegistry` dict and three new tool registrations.
- Capture tool handler gains a playfield resolution prefix (4-5 lines).
- No changes to existing camera registry or camera tools.

### `homography.py`

- `main()` body shrinks as calibration math moves to
  `calibrate_from_corners()`. External behavior unchanged.
- No new dependencies.

### `playfield.py`

- Constructor signature changes (adds optional `polygon` kwarg).
  Existing callers that don't pass `polygon` are unaffected
  (default is `None`, same as current behavior).

### Import Graph

```
MCP server
  ├── camera registry (Sprint 002)
  ├── PlayfieldRegistry (new)
  │     ├── playfield.Playfield
  │     └── homography.FieldSpec, calibrate_from_corners
  └── tool handlers
```

No circular imports. `playfield.py` imports from `models.py`.
`homography.py` imports from `config.py` and `screencap.py` (only in
`main()`; the new library function has no such imports).

## Migration Concerns

None. This sprint adds new MCP tools and refactors internals without
changing any external interfaces or stored data formats. The
`homography.json` file format produced by the CLI is unchanged.
