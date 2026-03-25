---
id: '005'
title: Batch operation pipeline and process_frame tool
status: done
use-cases:
- SUC-001
- SUC-002
depends-on:
- '001'
- '004'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Batch operation pipeline and process_frame tool

## Description

Implement the batch operation dispatch pipeline and the `process_frame` MCP tool.
This is the core processing engine that maps operation names to processing
functions and executes them in sequence on a FrameEntry.

### Operation dispatch

Create a dispatch mechanism that maps operation name strings to processing
functions. Each operation:
- Reads from the appropriate FrameEntry slot
- Stores results in `frame.results[operation_name]`
- Some operations (deskew) modify image slots; detection operations don't

| Operation | Function | Reads from | Writes to |
|-----------|----------|-----------|-----------|
| `"deskew"` | deskew logic | `original` | `deskewed` (new array), `processed` (ref) |
| `"detect_tags"` | AprilTag detection | `processed` | `results["detect_tags"]`, `apriltags` |
| `"detect_aruco"` | ArUco detection | `processed` | `results["detect_aruco"]`, `aruco_corners` |
| `"detect_lines"` | Hough lines | `processed` | `results["detect_lines"]` |
| `"detect_circles"` | Hough circles | `processed` | `results["detect_circles"]` |
| `"detect_contours"` | Contour detection | `processed` | `results["detect_contours"]` |
| `"detect_qr"` | QR code detection | `processed` | `results["detect_qr"]` |

### process_frame MCP tool

`process_frame(frame_id, operations)` -- Run a list of operations on an existing
frame. Operations execute in the order given. All results are returned in one
response.

### Wire up create_frame operations

After the pipeline is implemented, wire it into `create_frame` and
`create_frame_from_image` (ticket 004) so that their optional `operations`
parameter works. When operations are provided at creation time, the pipeline
runs immediately after capture/load.

## Acceptance Criteria

- [x] Operation dispatch maps operation name strings to processing functions
- [x] Operations execute in the order specified in the list
- [x] `"deskew"` reads from `original`, writes new array to `deskewed`, updates `processed` ref
- [x] `"detect_tags"` reads from `processed`, stores results without modifying image
- [x] `"detect_aruco"` reads from `processed`, stores results without modifying image
- [x] `"detect_lines"` reads from `processed`, stores results
- [x] `"detect_circles"` reads from `processed`, stores results
- [x] `"detect_contours"` reads from `processed`, stores results
- [x] `"detect_qr"` reads from `processed`, stores results
- [x] `process_frame` MCP tool returns all results in one response
- [x] `operations_applied` list on FrameEntry updated after each operation
- [x] `create_frame(source_id, operations=[...])` runs pipeline during creation
- [x] `create_frame_from_image(path, operations=[...])` runs pipeline during creation
- [x] Unknown operation names return a clear error message
- [x] Operations reuse existing functions from `image_processing.py` and `aprilcam.py`

## Implementation Notes

### Key files
- `src/aprilcam/mcp_server.py` -- `process_frame` MCP tool, dispatch logic
- `src/aprilcam/image_processing.py` -- existing `process_detect_lines`,
  `process_detect_circles`, `process_detect_contours`, `process_detect_qr_codes`
- `src/aprilcam/aprilcam.py` -- `detect_apriltags()` for tag detection
- `src/aprilcam/playfield.py` -- deskew/warp logic for the `"deskew"` operation

### Design decisions
- The dispatch dict is a simple `Dict[str, Callable]` mapping
- Each callable takes a `FrameEntry` and optional config, returns results
- Detection functions from `image_processing.py` are wrapped to work with
  FrameEntry (they currently take raw numpy arrays)
- Deskew requires a playfield context (for the homography matrix); if no
  playfield is associated with the frame's source, deskew is a no-op or error
- Results are JSON-serializable dicts (same format as existing MCP tool responses)

## Testing

- **Existing tests to run**: `uv run pytest` (full suite, ensure no regressions)
- **New tests to write**:
  - `test_operation_dispatch_known_ops` -- verify all operations are registered
  - `test_process_frame_single_op` -- run one operation, verify results
  - `test_process_frame_batch_ops` -- run multiple operations, verify order
  - `test_process_frame_deskew_modifies_slots` -- verify slot promotion on deskew
  - `test_process_frame_detect_preserves_image` -- verify detection doesn't modify slots
  - `test_process_frame_unknown_op` -- verify error on bad operation name
  - `test_create_frame_with_operations` -- verify pipeline runs at creation
- **Verification command**: `uv run pytest`
