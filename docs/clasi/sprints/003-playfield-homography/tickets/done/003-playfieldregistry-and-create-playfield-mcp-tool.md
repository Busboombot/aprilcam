---
id: "003"
title: "PlayfieldRegistry and create_playfield MCP tool"
status: todo
use-cases: [SUC-001]
depends-on: ["001", "002"]
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# PlayfieldRegistry and create_playfield MCP tool

## Description

Implement the `PlayfieldRegistry` in `mcp_server.py` and the
`create_playfield` MCP tool. The registry maps `playfield_id` strings
(pattern: `pf_{camera_id}`) to `PlayfieldEntry` objects holding the
camera_id, locked `Playfield` instance, and optional calibration data.

The `create_playfield(camera_id, max_frames=30)` tool:
1. Validates that `camera_id` exists in the camera registry.
2. Captures up to `max_frames` frames from the camera.
3. Calls `Playfield.update()` on each frame until the polygon locks.
4. If all 4 ArUco corners are found, registers the playfield and returns
   `{playfield_id, corners, calibrated: false}`.
5. If detection fails after `max_frames`, returns an error listing
   missing corner IDs.

Calling `create_playfield` twice for the same camera replaces the
previous playfield entry.

## Acceptance Criteria

- [ ] `PlayfieldEntry` dataclass holds `playfield_id`, `camera_id`, `playfield`, `field_spec`, `homography`
- [ ] `PlayfieldRegistry` class exists with `register()`, `get()`, `list()`, `remove()` methods
- [ ] `create_playfield` tool is registered on the MCP server
- [ ] Returns `{playfield_id, corners, calibrated}` on success
- [ ] Returns error with missing corner IDs on detection failure
- [ ] Corners are returned in UL, UR, LR, LL order as nested list
- [ ] Calling twice for same camera replaces the old entry
- [ ] Unknown `camera_id` returns a clear error message
- [ ] Integration test with mocked camera returning `tests/data/playfield_cam3.jpg`

## Testing

- **Existing tests to run**: `uv run pytest tests/test_mcp_server.py` (existing MCP tests pass)
- **New tests to write**: `tests/test_mcp_playfield.py` — integration tests for `create_playfield` with mocked camera
- **Verification command**: `uv run pytest`
