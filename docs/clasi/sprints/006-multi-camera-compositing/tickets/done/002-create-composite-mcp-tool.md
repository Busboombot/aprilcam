---
id: '002'
title: create_composite MCP tool
status: done
use-cases:
- SUC-001
- SUC-002
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# create_composite MCP tool

## Description

Register a `create_composite` MCP tool in `mcp_server.py` that allows
an AI agent to pair two cameras into a composite view with a cross-camera
homography. The tool supports two modes:

1. **Auto-detect**: Capture a frame from each camera, detect ArUco 4x4
   markers (IDs 0-3), find shared marker IDs between both frames, and
   compute the homography from matched marker centers. This is the
   default when `correspondence_points` is not provided.

2. **Manual**: Use caller-supplied `correspondence_points` directly to
   compute the homography, bypassing marker detection. Useful when
   markers are not visible to both cameras or when the agent has
   identified corresponding points through other means.

The tool validates inputs, delegates to `CompositeManager` and
`compute_cross_camera_homography` from ticket 001, and returns a
structured response with the `composite_id`, `reprojection_error`,
and `num_correspondences`.

### Parameters

- `primary_camera_id` (str, required) -- camera for visual frames
- `secondary_camera_id` (str, required) -- camera for tag detection
- `playfield_id` (str, optional) -- associate a playfield for world coords
- `correspondence_points` (list of `[[px, py], [sx, sy]]` pairs, optional)

### Error Conditions

- Unknown `primary_camera_id` or `secondary_camera_id`
- Fewer than 4 shared ArUco markers detected (auto-detect mode)
- Fewer than 4 correspondence point pairs provided (manual mode)
- Degenerate homography (near-singular matrix)

## Acceptance Criteria

- [ ] `create_composite` tool is registered in the MCP server and discoverable via tool listing
- [ ] Tool accepts `primary_camera_id` and `secondary_camera_id` as required string parameters
- [ ] Tool accepts optional `playfield_id` string parameter
- [ ] Tool accepts optional `correspondence_points` parameter (list of point pairs)
- [ ] Auto-detect mode: captures frames from both cameras and detects ArUco 4x4 markers
- [ ] Auto-detect mode: identifies shared marker IDs between both frames and uses their centers as correspondences
- [ ] Auto-detect mode: returns error with descriptive message when fewer than 4 shared markers are found
- [ ] Manual mode: computes homography directly from supplied correspondence points
- [ ] Manual mode: returns error when fewer than 4 point pairs are provided
- [ ] Returns JSON with `composite_id` (str), `reprojection_error` (float), and `num_correspondences` (int)
- [ ] Returns error with descriptive message for invalid camera IDs
- [ ] Returns error for degenerate homography (collinear or near-duplicate points)

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite for regressions
- **New tests to write**: `tests/test_mcp_composite.py`
  - Test tool registration and parameter schema
  - Test auto-detect mode with synthetic images containing ArUco markers at known positions (mock camera captures)
  - Test auto-detect mode error when fewer than 4 shared markers
  - Test manual mode with explicit correspondence points
  - Test manual mode error with fewer than 4 points
  - Test error response for invalid camera IDs
  - Test that `playfield_id` is stored on the resulting composite
- **Verification command**: `uv run pytest tests/test_mcp_composite.py -v`
