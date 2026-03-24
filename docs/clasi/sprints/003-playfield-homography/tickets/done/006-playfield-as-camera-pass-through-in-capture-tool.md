---
id: '006'
title: Playfield-as-camera pass-through in capture tool
status: done
use-cases:
- SUC-002
depends-on:
- '003'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Playfield-as-camera pass-through in capture tool

## Description

Modify the existing `capture_frame` MCP tool so that when a
`playfield_id` is passed as the `camera_id` parameter, the server:

1. Checks if the ID is registered in `PlayfieldRegistry`.
2. If yes, resolves the underlying `camera_id` from the playfield entry.
3. Captures a raw frame from the camera via the camera registry.
4. Applies `Playfield.deskew()` to produce a top-down rectangular image.
5. Encodes and returns the deskewed image (base64 or file path).

If the ID is not found in the playfield registry, it falls through to
the normal camera registry lookup (existing behavior unchanged).

The deskewed image dimensions are derived from the corner distances
(the pixel-only perspective transform), not the raw camera resolution.

## Acceptance Criteria

- [ ] Passing a `playfield_id` to `capture_frame` returns a deskewed image
- [ ] The deskewed image dimensions differ from the raw camera resolution
- [ ] Passing a normal `camera_id` still works unchanged (no regression)
- [ ] Unknown ID (not in either registry) returns "Unknown camera_id" error
- [ ] Both `base64` and `file` format options work with playfield captures
- [ ] Integration test: create playfield, capture via playfield_id, verify image is deskewed

## Testing

- **Existing tests to run**: `uv run pytest tests/test_mcp_server.py` (existing capture tests pass)
- **New tests to write**: Add pass-through capture tests to `tests/test_mcp_playfield.py`
- **Verification command**: `uv run pytest`
