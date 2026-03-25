---
id: '006'
title: stream_tags and detection loop refactor
status: done
use-cases:
- SUC-004
depends-on:
- '001'
- '003'
- '005'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# stream_tags and detection loop refactor

## Description

Implement the `stream_tags` MCP tool and refactor the detection loop to use the
batch operation pipeline internally. This connects the continuous detection mode
with the new frame model.

### stream_tags MCP tool

`stream_tags(source_id, operations)` -- Start a continuous detection loop on a
camera or playfield with a fixed operation pipeline. The pipeline is set at start
time and cannot be changed while running (stop and restart to change it).

Each iteration:
1. Capture a frame from the source
2. Create a FrameEntry in the frame ring buffer
3. Run the fixed operation pipeline (batch operations from ticket 005)
4. Extract tag detections from results
5. Create TagRecords and write to the detection ring buffer
6. If a playfield is involved, compute velocity via Playfield.add_tag()

### stop_stream MCP tool

`stop_stream()` -- Stop the currently running stream. Same semantics as existing
`stop_detection`.

### Existing tools unchanged

`get_tags()` and `get_tag_history()` continue to read from the detection ring
buffer as before. No changes to their API or behavior.

### Internal refactoring

The existing `DetectionLoop` in `detection.py` should be refactored to use the
batch operation pipeline internally, rather than calling individual detection
functions directly. This ensures a single code path for both `process_frame`
and `stream_tags`.

## Acceptance Criteria

- [ ] `stream_tags(source_id, operations)` starts a continuous detection loop
- [ ] Operations pipeline is fixed at start time
- [ ] Each frame is stored in the frame ring buffer as a FrameEntry
- [ ] TagRecords are written to the detection ring buffer (existing behavior)
- [ ] Velocity is computed by Playfield when applicable
- [ ] `stop_stream()` stops the running loop
- [ ] `get_tags()` returns latest detections (unchanged API)
- [ ] `get_tag_history()` returns detection history (unchanged API)
- [ ] Frames in the ring buffer are accessible via `get_frame_image` during streaming
- [ ] DetectionLoop uses batch pipeline internally (single code path)

## Implementation Notes

### Key files
- `src/aprilcam/mcp_server.py` -- `stream_tags` and `stop_stream` MCP tools
- `src/aprilcam/detection.py` -- `DetectionLoop` refactoring
- `src/aprilcam/playfield.py` -- velocity computation via `add_tag()`

### Design decisions
- `stream_tags` replaces `start_detection` as the preferred API, but
  `start_detection` can remain as a compatibility alias
- The operations list defaults to `["detect_tags"]` if not specified
- Frame ring buffer (300 entries at 30fps = ~10s of history) and detection
  ring buffer both receive data from each iteration
- TagRecord construction remains in DetectionLoop (per architecture decision)
- The loop thread captures, processes, and writes to both buffers in each cycle

### Relationship to existing detection
- `start_detection` / `stop_detection` currently exist and work
- `stream_tags` is the new preferred interface with explicit pipeline control
- Consider having `start_detection` delegate to `stream_tags` with default ops

## Testing

- **Existing tests to run**: `uv run pytest` (full suite, ensure no regressions)
- **New tests to write**:
  - `test_stream_tags_creates_frames` -- verify frames appear in ring buffer
  - `test_stream_tags_writes_tag_records` -- verify detection ring buffer populated
  - `test_stop_stream` -- verify loop stops cleanly
  - `test_stream_tags_fixed_pipeline` -- verify operations don't change mid-stream
  - `test_get_tags_during_stream` -- verify existing query tools work
- **Verification command**: `uv run pytest`
