---
id: 008
title: Add create_playfield_from_image MCP tool for static image files
status: done
use-cases: []
depends-on: []
github-issue: ''
todo: ''
---

# Add create_playfield_from_image MCP tool for static image files

## Description

Currently, `create_playfield` requires a live camera (`camera_id`).
There is no MCP path to create a playfield from a static image file.
This means an MCP client cannot demonstrate or use the deskew pipeline
without a live camera attached.

Add a `create_playfield_from_image` MCP tool that accepts an image file
path, detects the ArUco corners, creates a playfield, and registers it
in the PlayfieldRegistry. The resulting `playfield_id` should work with
`get_playfield_info` and `calibrate_playfield`.

Also add a `deskew_image` MCP tool that takes a `playfield_id` and an
image file path, applies the playfield's homography deskew, and returns
the deskewed image (base64 or file). This replaces the camera-based
`capture_frame` path for static images.

## Acceptance Criteria

- [ ] New `create_playfield_from_image(image_path)` MCP tool exists
- [ ] It detects ArUco corners and registers a playfield (camera_id set to "file:<path>")
- [ ] It returns the same JSON shape as `create_playfield` (playfield_id, corners, calibrated)
- [ ] New `deskew_image(playfield_id, image_path, format)` MCP tool exists
- [ ] It reads the image, applies deskew, returns the result as base64 or file
- [ ] Existing `get_playfield_info` and `calibrate_playfield` work with file-based playfields
- [ ] Tests cover both tools using the existing test images
- [ ] All existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Tests in `tests/test_mcp_playfield.py` for both new tools using `tests/data/playfield_cam3_moved.jpg`
- **Verification command**: `uv run pytest`
