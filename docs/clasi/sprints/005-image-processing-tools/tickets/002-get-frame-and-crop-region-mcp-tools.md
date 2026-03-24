---
id: "002"
title: "get_frame and crop_region MCP tools"
status: todo
use-cases: ["SUC-001", "SUC-007"]
depends-on: ["001"]
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# get_frame and crop_region MCP tools

## Description

Implement two MCP tools for basic frame retrieval and cropping:

1. **`get_frame(source_id, format?)`** — Captures a raw frame from a
   camera or playfield (via `resolve_source`), returns it with no
   processing applied. Uses `format_image_output` for the response.
   This is the simplest image tool and serves as the baseline.

2. **`crop_region(source_id, x, y, w, h, format?)`** — Captures a
   frame via `resolve_source`, extracts a rectangular sub-image at
   the given coordinates. Clips coordinates to frame bounds (no error
   if the region extends past the edge, just clip). Returns the
   cropped sub-image via `format_image_output`.

Both tools are registered as MCP tools on the server.

## Acceptance Criteria

- [ ] `get_frame` MCP tool registered and callable
- [ ] `get_frame` accepts `source_id` (required) and `format` (optional, default "base64")
- [ ] `get_frame` returns raw frame as base64 ImageContent by default
- [ ] `get_frame` returns file path as TextContent when format="file"
- [ ] `crop_region` MCP tool registered and callable
- [ ] `crop_region` accepts `source_id`, `x`, `y`, `w`, `h` (required) and `format` (optional)
- [ ] `crop_region` clips coordinates to frame bounds without error
- [ ] `crop_region` returns the cropped sub-image in the requested format
- [ ] Crop of a region entirely outside bounds returns an error or empty image with clear message
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: In `tests/test_image_processing.py`:
  - `test_get_frame_base64` — verify base64 output from mock camera
  - `test_get_frame_file` — verify file output path returned
  - `test_crop_region` — crop a known region, verify dimensions
  - `test_crop_region_clip` — region extends past edge, verify clipping
  - `test_crop_out_of_bounds` — region entirely outside, verify behavior
- **Verification command**: `uv run pytest`
