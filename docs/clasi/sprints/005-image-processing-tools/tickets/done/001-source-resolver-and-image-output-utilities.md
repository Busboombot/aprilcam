---
id: '001'
title: Source resolver and image output utilities
status: done
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
- SUC-006
- SUC-007
- SUC-008
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Source resolver and image output utilities

## Description

Add two shared utility functions to `mcp_server.py` that all 8 image
processing tools will depend on:

1. **`resolve_source(source_id)`** — Looks up a source_id in both the
   camera registry and playfield registry. Captures a frame from the
   underlying camera. If the source is a playfield, applies the deskew
   homography transform before returning. Returns a BGR ndarray ready
   for processing. Raises a descriptive error if the source_id is not
   found in either registry.

2. **`format_image_output(frame, format, quality)`** — Takes a BGR
   ndarray and encodes it for MCP response. When `format="base64"`,
   encodes as JPEG (with configurable quality) and returns an
   `ImageContent` block. When `format="file"`, writes to a temp file
   and returns a `TextContent` block with the file path. Defaults to
   base64 if format is omitted.

These are the foundational building blocks for all image processing
MCP tools in this sprint.

## Acceptance Criteria

- [ ] `resolve_source()` function exists in `mcp_server.py`
- [ ] `resolve_source()` returns BGR ndarray for a valid camera source_id
- [ ] `resolve_source()` returns deskewed BGR ndarray for a valid playfield source_id
- [ ] `resolve_source()` raises a clear error for an invalid/unknown source_id
- [ ] `format_image_output()` function exists in `mcp_server.py`
- [ ] `format_image_output(frame, "base64")` returns MCP `ImageContent` with JPEG data
- [ ] `format_image_output(frame, "file")` writes a temp file and returns `TextContent` with the path
- [ ] Quality parameter controls JPEG compression level
- [ ] Default format is `"base64"` when not specified
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: In `tests/test_image_processing.py`:
  - `test_resolve_source_camera` — mock camera registry, verify frame returned
  - `test_resolve_source_playfield` — mock playfield registry, verify deskewed frame returned
  - `test_resolve_source_invalid` — unknown source_id raises error
  - `test_format_image_base64` — verify base64 ImageContent structure
  - `test_format_image_file` — verify temp file written and path returned
- **Verification command**: `uv run pytest`
