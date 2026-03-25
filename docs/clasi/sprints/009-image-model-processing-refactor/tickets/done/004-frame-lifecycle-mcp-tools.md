---
id: '004'
title: Frame lifecycle MCP tools
status: done
use-cases:
- SUC-001
- SUC-002
- SUC-005
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Frame lifecycle MCP tools

## Description

Implement the MCP tools that let agents create, inspect, save, release, and list
frames. These tools expose the FrameEntry/FrameRegistry infrastructure (ticket 001)
to MCP clients, enabling the handle-based frame workflow.

### New MCP tools

1. **`create_frame(source_id, operations?)`** -- Capture a frame from an open
   camera or playfield. Store it in the FrameRegistry. If `operations` is
   provided, run the batch pipeline (ticket 005) before returning. Returns
   `frame_id` and any operation results.

2. **`create_frame_from_image(image_path, operations?)`** -- Load an image from
   disk (JPEG/PNG). Store it in the FrameRegistry with `source = "file:<path>"`.
   If `operations` provided, run the batch pipeline. This enables testing without
   a live camera.

3. **`get_frame_image(frame_id, stage)`** -- Return the image at a specific
   processing stage: `"original"`, `"deskewed"`, or `"processed"`. Returns
   base64-encoded image.

4. **`save_frame(frame_id, output_dir)`** -- Write the frame to disk as a
   directory containing `original.jpg`, `deskewed.jpg`, `processed.jpg`, and
   `metadata.json` (with frame_id, source, timestamp, operations_applied, results).

5. **`release_frame(frame_id)`** -- Explicitly remove a frame from the registry
   before auto-eviction.

6. **`list_frames()`** -- Return a list of all frames currently in the ring
   buffer with summary info (frame_id, source, timestamp, operations_applied).

### Note on operations parameter

The `operations` parameter on `create_frame` and `create_frame_from_image` will
call into the batch pipeline implemented in ticket 005. For this ticket, implement
the tools with `operations` as an optional parameter. If ticket 005 is not yet
complete, the tools should work without operations (capture/load only). Wire up
the operations dispatch when ticket 005 lands.

## Acceptance Criteria

- [x] `create_frame(source_id)` captures from camera/playfield and returns frame_id
- [x] `create_frame_from_image(image_path)` loads from disk and returns frame_id
- [x] `create_frame_from_image` validates the file exists and is a valid image
- [x] `get_frame_image(frame_id, "original")` returns the raw captured image
- [x] `get_frame_image(frame_id, "deskewed")` returns deskewed (or original if not deskewed)
- [x] `get_frame_image(frame_id, "processed")` returns pipeline output
- [x] `save_frame(frame_id, output_dir)` writes directory with 3 images + metadata.json
- [x] `metadata.json` contains frame_id, source, timestamp, operations_applied, results
- [x] `release_frame(frame_id)` removes the frame from the registry
- [x] `list_frames()` returns summary of all frames in the ring buffer
- [x] Error handling: invalid frame_id returns clear error message
- [x] Error handling: invalid image_path returns clear error message

## Implementation Notes

### Key files
- `src/aprilcam/mcp_server.py` -- add all 6 MCP tool functions
- `src/aprilcam/mcp_server.py` -- instantiate FrameRegistry alongside existing
  CameraRegistry and PlayfieldRegistry

### Design decisions
- `create_frame` uses the existing `resolve_source()` pattern to find the camera
  or playfield, then calls `capture_frame()` on it
- `create_frame_from_image` uses `cv2.imread()` to load the image
- `get_frame_image` returns base64 by default (consistent with existing tools)
- `save_frame` creates the output directory if it doesn't exist
- `save_frame` uses `cv2.imwrite()` for images and `json.dump()` for metadata
- The FrameRegistry instance is a module-level singleton (like camera_registry)

## Testing

- **Existing tests to run**: `uv run pytest` (full suite, ensure no regressions)
- **New tests to write**:
  - `test_create_frame_from_image` -- load test image, verify frame_id returned
  - `test_get_frame_image_stages` -- verify each stage returns correct image
  - `test_save_frame_writes_directory` -- verify directory structure and contents
  - `test_release_frame` -- verify frame removed, subsequent get raises error
  - `test_list_frames` -- verify summary output format
  - `test_create_frame_invalid_path` -- verify error on bad image path
- **Verification command**: `uv run pytest`
