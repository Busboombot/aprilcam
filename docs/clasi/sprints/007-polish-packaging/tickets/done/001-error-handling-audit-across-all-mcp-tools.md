---
id: '001'
title: Error handling audit across all MCP tools
status: done
use-cases:
- SUC-001
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Error handling audit across all MCP tools

## Description

Review every MCP tool handler in the aprilcam server for unguarded
exceptions. Currently, hardware failures (camera not found, frame grab
returning None, camera disconnection mid-loop) and invalid parameter
combinations can produce unhandled Python exceptions that crash tool
calls or leave the server in an inconsistent state. This ticket adds
structured error handling so every tool returns a parseable
`{"error": {"code": ..., "message": ...}}` response instead of
propagating raw exceptions.

Key areas to audit:
- `open_camera` / `close_camera` -- invalid index, already-open,
  already-closed, hardware not responding
- `capture_frame` / `get_frame` -- camera disconnected, frame grab
  returns None
- `start_detection` / `stop_detection` -- camera unavailable, loop
  already running, loop already stopped
- `get_tags` / `get_tag_history` -- no active loop, loop stopped due
  to error
- `create_playfield` / `create_playfield_from_image` -- no ArUco
  corners detected, insufficient corners
- `calibrate_playfield` -- playfield not found, invalid measurements
- `deskew_image` -- no homography available
- Image processing tools (`detect_lines`, `detect_circles`,
  `detect_contours`, `detect_motion`, `detect_qr_codes`,
  `crop_region`, `apply_transform`) -- invalid parameters, missing
  camera/playfield
- Composite tools -- invalid composite_id, missing cameras

Consider implementing a decorator or context manager that wraps tool
handlers to catch exceptions uniformly and return structured errors.

## Acceptance Criteria

- [ ] Every MCP tool handler is wrapped with try/except that catches
      exceptions and returns a structured error dict
- [ ] `open_camera` with invalid index returns `{"error": ...}`, not
      an unhandled exception
- [ ] `capture_frame` on a disconnected camera returns a structured
      error and the server remains operational
- [ ] Detection loop stops cleanly when camera returns None frames,
      with an error status queryable via `get_tags`
- [ ] `create_playfield` returns a descriptive error when ArUco
      corners are not detected in the frame
- [ ] Image processing tools return errors for invalid camera_id or
      playfield_id references
- [ ] Server continues to accept new requests after any tool error
      (no crash, no hung state)
- [ ] An error-handling decorator or context manager is implemented to
      standardize the pattern across all tools
- [ ] Error messages include relevant context (camera index, tool
      name, parameter values) to aid debugging

## Testing

- **Existing tests to run**: `uv run pytest` -- full suite
- **New tests to write**:
  - Test `open_camera` with invalid index returns structured error
  - Test `capture_frame` when mock camera returns None
  - Test detection loop behavior when camera disconnects mid-loop
  - Test `create_playfield` with image containing no ArUco markers
  - Test each image processing tool with invalid camera_id
  - Test server remains responsive after error in any tool
- **Verification command**: `uv run pytest`
