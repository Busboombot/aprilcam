---
id: '003'
title: get_composite_frame and get_composite_tags MCP tools
status: done
use-cases:
- SUC-003
- SUC-004
depends-on:
- '002'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# get_composite_frame and get_composite_tags MCP tools

## Description

Register two MCP tools that let an agent query a composite view created
by `create_composite`:

### `get_composite_frame(composite_id, format?)`

Captures a frame from both cameras, detects tags on the secondary camera,
maps tag positions to primary camera coordinates using the stored
homography, renders tag overlays (bounding boxes and IDs) onto the
primary camera's frame, and returns the annotated image.

- Captures from the primary (color) camera for the base image.
- Captures from the secondary (B&W) camera and runs tag detection.
- Maps detected tag corners/centers via `map_tags_to_primary`.
- Calls `render_tag_overlay(frame, mapped_tags)` to draw overlays.
- Returns the annotated frame in the requested format (`base64` or
  `file`), defaulting to `base64`.

### `get_composite_tags(composite_id)`

Captures from the secondary camera, detects tags, maps coordinates to
primary camera space, and returns structured JSON data without rendering
an image.

- Each tag in the response includes: `id`, `center_px` (in primary
  coords), `corners_px` (in primary coords), `orientation_yaw`.
- If the composite has an associated `playfield_id` with a calibrated
  world-coordinate homography, includes `world_xy` for each tag.
- Returns an empty list (not an error) when no tags are detected.

### `render_tag_overlay(frame, mapped_tags)` helper

A helper function (in `composite.py`) that draws tag bounding boxes and
ID labels onto a frame at the mapped positions. Uses `cv2.polylines`
for bounding boxes and `cv2.putText` for labels, following the drawing
conventions from `display.py`.

## Acceptance Criteria

- [ ] `get_composite_frame` tool is registered in the MCP server and discoverable
- [ ] `get_composite_frame` accepts `composite_id` (required) and `format` (optional, default `"base64"`)
- [ ] `get_composite_frame` captures from the primary camera for the base image
- [ ] `get_composite_frame` captures from the secondary camera and runs tag detection
- [ ] `get_composite_frame` maps detected tag positions from secondary to primary coordinates
- [ ] `get_composite_frame` renders tag bounding boxes and IDs as overlays on the primary frame
- [ ] `get_composite_frame` returns the annotated image in base64 format when `format="base64"`
- [ ] `get_composite_frame` returns a file path when `format="file"`
- [ ] `get_composite_frame` returns error for invalid `composite_id`
- [ ] `get_composite_tags` tool is registered in the MCP server and discoverable
- [ ] `get_composite_tags` accepts `composite_id` (required)
- [ ] `get_composite_tags` returns a list of tag objects with `id`, `center_px`, `corners_px`, `orientation_yaw`
- [ ] `get_composite_tags` coordinates are in primary camera pixel space (transformed via homography)
- [ ] `get_composite_tags` includes `world_xy` when a calibrated playfield is associated
- [ ] `get_composite_tags` returns an empty list (not an error) when no tags are detected
- [ ] `get_composite_tags` returns error for invalid `composite_id`
- [ ] `render_tag_overlay` draws polyline bounding boxes around each tag at mapped corner positions
- [ ] `render_tag_overlay` draws tag ID labels near each tag

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite for regressions
- **New tests to write**: Added to `tests/test_mcp_composite.py` and `tests/test_composite.py`
  - Test `get_composite_frame` with mock cameras returning synthetic images with known tags -- verify output image contains overlay pixels at expected positions
  - Test `get_composite_frame` base64 output is valid base64-encoded image
  - Test `get_composite_frame` file output writes a file and returns a valid path
  - Test `get_composite_frame` error for invalid composite_id
  - Test `get_composite_tags` returns correctly transformed tag coordinates (verify against manually computed expected values)
  - Test `get_composite_tags` returns empty list when secondary camera has no tags
  - Test `get_composite_tags` includes `world_xy` when playfield is calibrated
  - Test `get_composite_tags` error for invalid composite_id
  - Test `render_tag_overlay` modifies the frame (output differs from input)
  - Test `render_tag_overlay` with empty tag list returns frame unchanged
- **Verification command**: `uv run pytest tests/test_composite.py tests/test_mcp_composite.py -v`
