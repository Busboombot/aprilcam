---
id: '007'
title: Real-image and integration tests for playfield tools
status: done
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
depends-on:
- '003'
- '004'
- '005'
- '006'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Real-image and integration tests for playfield tools

## Description

Write comprehensive tests using the real playfield images in
`tests/data/` (`playfield_cam3.jpg`, `playfield_cam3_moved.jpg`) to
validate the full end-to-end pipeline: corner detection, playfield
creation, deskew, calibration, and homography accuracy.

These tests mock the camera to return the captured test images instead
of a live camera feed. They verify that all sprint use cases work
correctly against real-world data, not just synthetic frames.

Additionally, verify the computed homography matrix against the
reference `data/homography.json` file (within tolerance).

## Acceptance Criteria

- [ ] `tests/test_playfield_real.py` exists with real-image tests
- [ ] ArUco corner detection finds all 4 corners (IDs 0-3) in `playfield_cam3.jpg`
- [ ] ArUco corner detection finds all 4 corners in `playfield_cam3_moved.jpg`
- [ ] `create_playfield` succeeds with mocked camera returning real image
- [ ] `deskew()` produces a rectangular top-down image from real playfield capture
- [ ] Computed homography matrix matches `data/homography.json` reference within tolerance
- [ ] AprilTag detection finds expected tags (IDs 0-6, 30) after deskew
- [ ] MCP integration test: full flow (create, calibrate, info, capture) with mocked camera
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` (full suite)
- **New tests to write**: `tests/test_playfield_real.py`, additional end-to-end cases in `tests/test_mcp_playfield.py`
- **Verification command**: `uv run pytest`
