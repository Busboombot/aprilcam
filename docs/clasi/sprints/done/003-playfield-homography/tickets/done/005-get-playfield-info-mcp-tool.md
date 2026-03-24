---
id: '005'
title: get_playfield_info MCP tool
status: done
use-cases:
- SUC-004
depends-on:
- '003'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# get_playfield_info MCP tool

## Description

Implement the `get_playfield_info` MCP tool that returns the current
state of a registered playfield as a JSON object. This is a read-only
query tool that the agent uses to inspect playfield configuration.

The tool returns:
- `playfield_id` -- the playfield handle
- `camera_id` -- the underlying camera handle
- `corners` -- 4x2 array in UL, UR, LR, LL order (pixel coordinates)
- `calibrated` -- boolean
- `width_cm` and `height_cm` -- present only if calibrated
- `homography` -- 3x3 matrix as nested list, present only if calibrated

## Acceptance Criteria

- [ ] `get_playfield_info` tool is registered on the MCP server
- [ ] Returns all specified fields for a calibrated playfield
- [ ] Returns corners and `calibrated: false` for an uncalibrated playfield
- [ ] Omits `width_cm`, `height_cm`, `homography` when uncalibrated
- [ ] Unknown `playfield_id` returns error
- [ ] JSON output matches the schema described in sprint.md architecture notes
- [ ] Integration test: create playfield, query info, verify all fields present

## Testing

- **Existing tests to run**: `uv run pytest tests/test_mcp_server.py`
- **New tests to write**: Add info query tests to `tests/test_mcp_playfield.py`
- **Verification command**: `uv run pytest`
