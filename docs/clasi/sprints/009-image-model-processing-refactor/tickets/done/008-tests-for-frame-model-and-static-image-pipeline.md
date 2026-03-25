---
id: 008
title: Tests for frame model and static image pipeline
status: done
use-cases:
- SUC-001
- SUC-004
- SUC-005
depends-on:
- '001'
- '002'
- '003'
- '004'
- '005'
- '007'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Tests for frame model and static image pipeline

## Description

Comprehensive test ticket covering unit tests, integration tests, backward
compatibility tests, and regression verification for the sprint 009 changes.
This ticket is executed last to validate all other tickets together.

While individual tickets include their own test criteria, this ticket ensures
end-to-end coverage and cross-cutting concerns are tested.

### Unit tests

- **FrameEntry slot promotion**: Verify zero-copy on creation, new array on
  deskew, reference update on processed slot
- **FrameRegistry ring buffer**: Verify capacity, deterministic IDs, auto-eviction,
  release, list, thread safety
- **Batch pipeline dispatch**: Verify operation mapping, execution order, result
  storage, unknown operation handling
- **Velocity EMA**: Verify Playfield EMA + dead-band produces correct values
  for known input sequences

### Integration tests (static image flow)

Using `tests/data/playfield_cam3.jpg` (or other test images in `tests/data/`):

- `create_frame_from_image` loads the test image, returns frame_id
- `process_frame(frame_id, ["detect_tags"])` detects tags in the static image
- `get_frame_image(frame_id, "original")` returns the original image
- `get_frame_image(frame_id, "processed")` returns the processed image
- `save_frame(frame_id, tmp_dir)` writes the expected directory structure
- `process_frame(frame_id, ["detect_lines", "detect_contours"])` runs multiple
  operations on a single frame
- Full pipeline: `create_frame_from_image(path, operations=["deskew", "detect_tags"])`
  with a playfield context

### Backward compatibility tests

- Existing `detect_lines`, `detect_circles`, `detect_contours`, `detect_qr_codes`
  tools return identical response format after wrapper refactoring
- `get_tags` and `get_tag_history` work unchanged during streaming
- `start_detection` / `stop_detection` still work (if kept as aliases)

### Regression

- All existing tests in the test suite pass without modification (or with
  minimal updates for interface changes)

## Acceptance Criteria

- [x] Unit tests pass for FrameEntry slot promotion logic
- [x] Unit tests pass for FrameRegistry ring buffer (capacity, eviction, IDs)
- [x] Unit tests pass for batch pipeline operation dispatch
- [x] Unit tests pass for Playfield velocity EMA + dead-band
- [x] Integration test: create_frame_from_image with static test image succeeds
- [x] Integration test: process_frame with detect_tags on static image returns detections
- [x] Integration test: save_frame writes correct directory structure
- [x] Integration test: full pipeline (create + operations) works in one call
- [x] Backward compat: existing per-operation tools return same response format
- [x] Regression: `uv run pytest` passes with zero failures
- [x] AprilTag family field populated in detection results

## Implementation Notes

### Key files
- `tests/` -- all test files
- `tests/data/playfield_cam3.jpg` -- primary test image for integration tests
- `tests/data/` -- other test images if available

### Design decisions
- Use `pytest` fixtures for FrameRegistry setup/teardown
- Use `tmp_path` fixture for save_frame tests
- Static image tests don't require camera hardware
- Mock camera for create_frame tests; real image for create_frame_from_image tests
- Velocity parity tests: run same input sequence through old (AprilCam) and new
  (Playfield) code paths, compare outputs

### Test organization
- `tests/test_frame.py` -- FrameEntry and FrameRegistry unit tests
- `tests/test_pipeline.py` -- batch operation pipeline tests
- `tests/test_frame_mcp.py` -- MCP tool integration tests with static images
- `tests/test_velocity.py` -- Playfield velocity computation tests
- Or add to existing test files if that's the established pattern

## Testing

- **Existing tests to run**: `uv run pytest` (full regression suite)
- **New tests to write**: All tests described above
- **Verification command**: `uv run pytest -v` (verbose to see all test names)
