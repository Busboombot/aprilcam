---
id: "004"
title: "calibrate_playfield MCP tool"
status: todo
use-cases: [SUC-003]
depends-on: ["003"]
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# calibrate_playfield MCP tool

## Description

Implement the `calibrate_playfield` MCP tool that accepts a
`playfield_id` and real-world measurements `{width, height, units}`,
computes the pixel-to-world homography matrix, and stores it in the
playfield registry entry.

The tool:
1. Looks up the playfield entry by `playfield_id`.
2. Creates a `FieldSpec` from the measurements (supports "inch" and "cm").
3. Retrieves the pixel corner positions from the locked polygon
   (UL, UR, LR, LL from `Playfield.get_polygon()`).
4. Builds the corner dict expected by `calibrate_from_corners()`.
5. Calls `calibrate_from_corners(pixel_corners, field_spec)` to get H.
6. Stores the `FieldSpec` and homography matrix in the playfield entry.
7. Returns `{playfield_id, calibrated: true, width_cm, height_cm}`.

Calling calibrate again overwrites the previous calibration.

## Acceptance Criteria

- [ ] `calibrate_playfield` tool is registered on the MCP server
- [ ] Accepts `playfield_id`, `width`, `height`, `units` parameters
- [ ] Converts inches to cm correctly via `FieldSpec`
- [ ] Stores homography matrix and field_spec in the PlayfieldEntry
- [ ] Returns `{playfield_id, calibrated, width_cm, height_cm}`
- [ ] Unknown `playfield_id` returns error
- [ ] Missing/invalid measurements return error
- [ ] Re-calibration overwrites previous calibration
- [ ] Stored homography correctly maps corner pixels to expected world coords (within 0.1 cm)

## Testing

- **Existing tests to run**: `uv run pytest tests/test_mcp_server.py`
- **New tests to write**: Add calibration tests to `tests/test_mcp_playfield.py`
- **Verification command**: `uv run pytest`
