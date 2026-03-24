---
id: '007'
title: Integration tests
status: done
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
depends-on:
- '006'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Integration tests

## Description

Write end-to-end integration tests that exercise the full MCP tool flow
for the detection loop: `start_detection` -> `get_tags` ->
`get_tag_history` -> `stop_detection`. These tests use a mock camera
backed by test images from `tests/data/` (`playfield_cam3.jpg`,
`playfield_cam3_moved.jpg`) to simulate a real camera feed without
hardware.

The integration tests verify that the entire pipeline works together:
MCP tool registration, detection loop lifecycle, ring buffer storage,
tag record serialization, and velocity computation.

### Test Scenarios

1. **Full round-trip** -- Open a mock camera that cycles through
   `playfield_cam3.jpg` and `playfield_cam3_moved.jpg`. Start
   detection, wait for enough frames, call `get_tags` and verify tag
   positions match expected detections. Call `get_tag_history` and
   verify multiple frames are returned with consistent ordering.
   Stop detection and verify clean shutdown.

2. **Velocity computation** -- Feed two different images (tags in
   different positions) through the loop. Verify that `vel_px` and
   `speed_px` fields in tag records reflect the displacement between
   frames. Verify near-zero velocity when the same image is repeated.

3. **All JSON fields present** -- Verify that every field specified in
   the `TagRecord` dataclass appears in the MCP tool JSON output,
   including `null` for unavailable optional fields.

4. **Error cases** -- `get_tags` before any detection is started.
   `stop_detection` with an invalid source_id. Starting detection
   on an already-active source.

5. **Playfield-aware detection** -- Create a playfield from a test
   image, start detection on the playfield_id, verify that
   `in_playfield` is correctly set and (if calibrated) `world_xy`
   is populated.

### Mock Camera

Create a `MockVideoCapture` class (or use an existing one from the
test suite) that implements `read() -> (bool, ndarray)` and
`isOpened() -> bool`, cycling through a list of image file paths.
This allows deterministic testing without camera hardware.

## Acceptance Criteria

- [ ] `tests/test_mcp_detection.py` contains integration tests for the full MCP detection flow
- [ ] Full round-trip test: start -> get_tags -> get_tag_history -> stop all succeed
- [ ] Tag positions in `get_tags` response match expected detections from test images
- [ ] `get_tag_history` returns frames in chronological order (oldest first)
- [ ] Velocity fields (`vel_px`, `speed_px`) are non-None after multiple frames with tag movement
- [ ] Velocity is near-zero when the same image is repeated
- [ ] All TagRecord fields are present in JSON output
- [ ] Optional fields are `null` when not available
- [ ] Error case: `get_tags` returns error when no loop is active
- [ ] Error case: `stop_detection` returns error for unknown source
- [ ] Error case: `start_detection` returns error for duplicate source
- [ ] Playfield-aware detection sets `in_playfield` correctly
- [ ] All tests pass: `uv run pytest tests/ -v`

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite to verify no regressions
- **New tests to write** (in `tests/test_mcp_detection.py`):
  - `test_full_detection_roundtrip` -- open mock camera, start_detection, wait, get_tags, get_tag_history, stop_detection
  - `test_tag_positions_match_expected` -- verify detected tag IDs and center_px values are plausible for the test images
  - `test_velocity_computation` -- cycle between two different test images, verify non-zero vel_px/speed_px
  - `test_velocity_near_zero_static` -- repeat the same test image, verify near-zero velocity
  - `test_all_json_fields_present` -- check every expected key in the get_tags response
  - `test_history_chronological_order` -- verify frame_index and timestamp are monotonically increasing in get_tag_history
  - `test_error_get_tags_no_loop` -- get_tags without start, expect error
  - `test_error_stop_unknown_source` -- stop_detection with bogus ID
  - `test_error_start_duplicate` -- start_detection twice on same source
  - `test_playfield_detection` -- create playfield from test image, start detection, verify in_playfield flag
- **Verification command**: `uv run pytest tests/test_mcp_detection.py -v`
