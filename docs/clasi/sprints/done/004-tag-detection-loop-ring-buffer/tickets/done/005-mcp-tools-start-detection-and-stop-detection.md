---
id: '005'
title: "MCP tools \u2014 start_detection and stop_detection"
status: done
use-cases:
- SUC-001
- SUC-002
depends-on:
- '004'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# MCP tools — start_detection and stop_detection

## Description

Register two new MCP tools in `src/aprilcam/mcp_server.py` that allow
AI agents to start and stop background detection loops on open cameras
or playfields.

### Detection Loop Registry

Add a `DetectionEntry` dataclass and a module-level
`detection_registry` dict in `mcp_server.py` that maps `source_id`
(camera_id or playfield_id) to an active `DetectionLoop` instance plus
its `RingBuffer`. This enforces the one-loop-per-source constraint.

```python
@dataclass
class DetectionEntry:
    source_id: str
    loop: DetectionLoop
    ring_buffer: RingBuffer
    aprilcam: AprilCam

detection_registry: dict[str, DetectionEntry] = {}
```

### start_detection tool

```python
@server.tool()
async def start_detection(
    source_id: str,
    family: str = "36h11",
    proc_width: int = 960,
    detect_interval: int = 1,
    use_clahe: bool = False,
    use_sharpen: bool = False,
) -> list[TextContent]:
```

1. Resolve `source_id` -- look up in camera registry first, then
   playfield registry. If playfield, get the underlying camera capture.
   If neither, return error.
2. Check one-loop-per-source -- if `source_id` already has an active
   detection loop, return error.
3. Create a headless `AprilCam` with the given parameters. If the
   source is a playfield, pass the playfield's homography and polygon.
4. Create a `RingBuffer(maxlen=300)`.
5. Create a `DetectionLoop(source, aprilcam, ring_buffer)`.
6. Call `loop.start()`.
7. Register in `detection_registry`.
8. Return `{"source_id": source_id, "status": "started"}`.

### stop_detection tool

```python
@server.tool()
async def stop_detection(
    source_id: str,
) -> list[TextContent]:
```

1. Look up `source_id` in `detection_registry`. If not found, return
   error.
2. Call `loop.stop()`.
3. Remove from `detection_registry`.
4. Return `{"source_id": source_id, "status": "stopped"}`.

### Cleanup

When the MCP server shuts down (`main()` finally block), stop all
active detection loops before closing cameras.

## Acceptance Criteria

- [ ] `start_detection` MCP tool is registered and callable
- [ ] `start_detection` resolves `source_id` from camera or playfield registry
- [ ] `start_detection` creates and starts a `DetectionLoop` with the given parameters
- [ ] `start_detection` returns `{"source_id": ..., "status": "started"}` on success
- [ ] `start_detection` returns error if `source_id` is not a valid camera or playfield
- [ ] `start_detection` returns error if a loop is already active for that source
- [ ] `start_detection` passes playfield homography and polygon to AprilCam when source is a playfield
- [ ] `stop_detection` MCP tool is registered and callable
- [ ] `stop_detection` stops the detection loop and removes it from the registry
- [ ] `stop_detection` returns `{"source_id": ..., "status": "stopped"}` on success
- [ ] `stop_detection` returns error if `source_id` has no active loop
- [ ] Server shutdown stops all active detection loops
- [ ] All existing MCP tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/test_mcp_server.py tests/test_mcp_playfield.py` -- verify no regressions in camera/playfield tools
- **New tests to write** (in `tests/test_mcp_detection.py`):
  - `test_start_detection_with_camera` -- open a mock camera, call `start_detection`, verify success response and loop is running
  - `test_start_detection_invalid_source` -- call with nonexistent source_id, verify error response
  - `test_start_detection_duplicate` -- start detection twice on the same source, verify second call returns error
  - `test_stop_detection` -- start then stop, verify success and loop is no longer running
  - `test_stop_detection_invalid` -- stop on a source with no active loop, verify error
  - `test_start_detection_with_playfield` -- create a playfield, start detection on playfield_id, verify homography is passed through
- **Verification command**: `uv run pytest tests/test_mcp_detection.py tests/test_mcp_server.py -v`
