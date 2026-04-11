---
id: "004"
title: "detect_tags integration and MCP get_objects tool"
status: done
use-cases: ["SUC-012-001"]
depends-on: ["003"]
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# detect_tags integration and MCP get_objects tool

## Description

Wire the object detection pipeline into the existing `detect_tags()` streaming
generator and expose detected objects through a new MCP tool. This ticket
connects the components built in tickets 001-003 into the live detection
workflow.

### detect_tags() updates

Update `src/aprilcam/stream.py` `detect_tags()` to accept two new optional
parameters:

- `detect_objects: bool = False` -- when True, run SquareDetector on each
  frame alongside tag detection
- `color_camera: int | str | None = None` -- when provided (and
  `detect_objects=True`), start a ColorCameraThread for the specified camera
  to provide color labels via fusion

**Behavior when `detect_objects=True`**:

1. At startup, create a `SquareDetector()` and `ObjectFuser()` instance
2. If `color_camera` is specified, create a `ColorClassifier()` and
   `ColorCameraThread(color_camera, fuser, classifier, homography)`, then
   call `start()`
3. On each frame iteration:
   a. Run existing tag detection (unchanged)
   b. Run `square_detector.detect(gray, homography, tag_corners)` where
      `tag_corners` comes from the detected tags
   c. Run `fuser.fuse(bw_objects)` to apply color labels
   d. Yield a `FrameResult(tags=tags, objects=fused_objects, timestamp=..., frame_index=...)`
      instead of the raw tag list
4. On cleanup (generator close or exception), stop the ColorCameraThread
   if it was started

**Backward compatibility**: When `detect_objects=False` (default), yield
`FrameResult(tags=tags, objects=[], ...)`. Since FrameResult iterates as
a tag list (ticket 001), existing callers that iterate over results are
unaffected.

### Package exports

Update `src/aprilcam/__init__.py` to export:
- `ObjectRecord`
- `FrameResult`

### MCP tool: get_objects

Add a new MCP tool `get_objects(source_id: str)` in `src/aprilcam/mcp_server.py`:

- Requires an active detection loop on the given source (started with
  `start_detection` and `detect_objects=True`)
- Returns the latest list of detected objects as a JSON array
- Each object includes: `center_px`, `world_xy`, `color`, `bbox`, `area_px`,
  `object_type`, `confidence`
- If detection loop is not running or `detect_objects` was not enabled,
  return an error message

The tool follows the same pattern as the existing `get_tags` tool.

## Acceptance Criteria

- [ ] `detect_tags()` accepts `detect_objects` and `color_camera` parameters
- [ ] When `detect_objects=True`, SquareDetector runs on each frame
- [ ] When `color_camera` is provided, ColorCameraThread starts and provides color fusion
- [ ] `detect_tags()` yields `FrameResult` instances (not raw tag lists)
- [ ] Existing callers iterating over `detect_tags()` results still work (backward compat)
- [ ] ColorCameraThread is stopped on generator cleanup
- [ ] `ObjectRecord` and `FrameResult` are exported from `aprilcam.__init__`
- [ ] MCP `get_objects` tool returns detected objects as JSON
- [ ] MCP `get_objects` returns an error if detection loop is not active or objects not enabled
- [ ] `start_detection` MCP tool accepts an `detect_objects` parameter
- [ ] All new code has type hints
- [ ] All tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- verify all existing tag detection and streaming tests still pass (critical for backward compat)
- **New tests to write**:
  - `tests/test_detect_objects_integration.py`:
    - `test_detect_tags_default_yields_frame_result` -- call detect_tags() with defaults, verify yields FrameResult, verify iteration gives tags
    - `test_detect_tags_with_objects` -- mock camera and SquareDetector, call with `detect_objects=True`, verify FrameResult contains objects
    - `test_detect_tags_with_color_camera` -- mock both cameras, verify ColorCameraThread started and stopped
    - `test_detect_tags_cleanup_stops_thread` -- close the generator, verify ColorCameraThread.stop() called
    - `test_backward_compat_iteration` -- existing code pattern `for tags in detect_tags(...): for tag in tags:` still works
  - `tests/test_mcp_get_objects.py`:
    - `test_get_objects_returns_json` -- start detection with objects, call get_objects, verify JSON response
    - `test_get_objects_no_detection_loop` -- call without active detection, verify error response
    - `test_get_objects_detection_without_objects` -- detection running but detect_objects=False, verify error
- **Verification command**: `uv run pytest tests/test_detect_objects_integration.py tests/test_mcp_get_objects.py -v`
