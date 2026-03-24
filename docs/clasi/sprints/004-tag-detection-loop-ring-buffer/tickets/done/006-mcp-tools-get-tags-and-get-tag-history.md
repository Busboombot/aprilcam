---
id: '006'
title: "MCP tools \u2014 get_tags and get_tag_history"
status: done
use-cases:
- SUC-003
- SUC-004
depends-on:
- '005'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# MCP tools — get_tags and get_tag_history

## Description

Register two new MCP tools in `src/aprilcam/mcp_server.py` that allow
AI agents to query the current and historical tag state from an active
detection loop.

### get_tags tool

```python
@server.tool()
async def get_tags(
    source_id: str,
) -> list[TextContent]:
```

1. Look up `source_id` in `detection_registry`. If not found, return
   error with message indicating no active detection loop.
2. Call `ring_buffer.get_latest()`.
3. If None (no frames processed yet), return
   `{"source_id": ..., "frame": null, "tags": []}`.
4. Otherwise, return `record.to_dict()` with `source_id` added.

Response shape (example):
```json
{
  "source_id": "abc-123",
  "timestamp": 12345.678,
  "frame_index": 42,
  "tags": [
    {
      "id": 7,
      "center_px": [320.5, 240.1],
      "corners_px": [[...], [...], [...], [...]],
      "orientation_yaw": 1.23,
      "world_xy": [15.2, 8.7],
      "in_playfield": true,
      "vel_px": [10.5, -3.2],
      "speed_px": 10.98,
      "vel_world": [1.2, -0.4],
      "speed_world": 1.26,
      "heading_rad": -0.30,
      "timestamp": 12345.678,
      "frame_index": 42
    }
  ]
}
```

### get_tag_history tool

```python
@server.tool()
async def get_tag_history(
    source_id: str,
    num_frames: int = 30,
) -> list[TextContent]:
```

1. Look up `source_id` in `detection_registry`. If not found, return
   error.
2. Call `ring_buffer.get_last_n(num_frames)`.
3. Return
   `{"source_id": ..., "frames": [record.to_dict() for record in records]}`.

The `num_frames` parameter defaults to 30 (~1 second at 30fps). The
agent can request up to 300 (the full buffer). Values above the buffer
capacity return all available frames.

## Acceptance Criteria

- [ ] `get_tags` MCP tool is registered and callable
- [ ] `get_tags` returns the latest `FrameRecord` as JSON with `source_id` included
- [ ] `get_tags` returns `{"source_id": ..., "frame": null, "tags": []}` when no frames have been processed
- [ ] `get_tags` returns error when no detection loop is active for the source
- [ ] `get_tag_history` MCP tool is registered and callable
- [ ] `get_tag_history` returns the last N `FrameRecord` objects as JSON
- [ ] `get_tag_history` defaults to 30 frames when `num_frames` is not specified
- [ ] `get_tag_history` returns all available frames when `num_frames` exceeds buffer size
- [ ] `get_tag_history` returns error when no detection loop is active for the source
- [ ] All tag fields are present in the JSON output (id, center_px, corners_px, orientation_yaw, world_xy, in_playfield, vel_px, speed_px, vel_world, speed_world, heading_rad, timestamp, frame_index)
- [ ] Optional fields are `null` when not available (e.g., world_xy without calibration)
- [ ] All existing MCP tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/test_mcp_server.py tests/test_mcp_playfield.py`
- **New tests to write** (in `tests/test_mcp_detection.py`):
  - `test_get_tags_active_loop` -- start detection with mock camera yielding test images, wait for frames, call `get_tags`, verify response has tags with expected fields
  - `test_get_tags_no_loop` -- call `get_tags` without starting detection, verify error response
  - `test_get_tags_empty_buffer` -- start detection, immediately call `get_tags` before any frames are processed, verify empty response
  - `test_get_tag_history_default` -- start detection, wait for several frames, call `get_tag_history`, verify returns up to 30 frames
  - `test_get_tag_history_custom_n` -- call with `num_frames=5`, verify exactly 5 frames returned
  - `test_get_tag_history_exceeds_buffer` -- call with `num_frames=1000`, verify returns all available (no error)
  - `test_get_tag_history_no_loop` -- call without detection loop, verify error
  - `test_get_tags_all_fields_present` -- verify every expected field is in the tag JSON
- **Verification command**: `uv run pytest tests/test_mcp_detection.py -v`
