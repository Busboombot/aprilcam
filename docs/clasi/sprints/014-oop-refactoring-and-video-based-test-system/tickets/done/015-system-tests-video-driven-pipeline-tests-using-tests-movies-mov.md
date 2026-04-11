---
id: '015'
title: 'System tests: video-driven pipeline tests using tests/movies/*.mov'
status: done
use-cases:
  - SUC-002
  - SUC-010
depends-on:
  - "014"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
  - docs/clasi/todo/video-based-test-system.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# System tests: video-driven pipeline tests using tests/movies/*.mov

## Description

Create `tests/system/` with full pipeline tests driven by the four test videos
in `tests/movies/`. Each test constructs a `VideoCamera`, builds a `Playfield`,
runs the `DetectionPipeline` to completion, and asserts on detection results.

System tests verify end-to-end behavior: tags are detected, positions are
plausible, velocity is computed, `estimate()` extrapolation is consistent.
No camera hardware required.

## Acceptance Criteria

- [ ] `tests/system/` directory exists with `__init__.py`.
- [ ] `tests/system/test_detection_pipeline_videos.py` exists with parametrized
      tests for all four video files.
- [ ] For each of `bright-gsc.mov`, `bright-ov9782.mov`, `dim-gsc.mov`,
      `dim-ov9782.mov`:
      - [ ] Pipeline runs to completion (EOF) without raising.
      - [ ] At least one tag is detected in the video.
      - [ ] Detected tag `center_px` values are within frame dimensions.
      - [ ] If velocity is non-zero, `estimate(t)` returns a position within
            2× the frame dimensions (sanity check on extrapolation).
- [ ] Bright lighting tests (`bright-*.mov`) detect at least one tag.
- [ ] Dim lighting tests (`dim-*.mov`) detect at least one tag (may require
      relaxed assertion or `pytest.mark.xfail` if dim videos are genuinely very dark).
- [ ] Both camera types (`*-gsc.mov`, `*-ov9782.mov`) produce detections.
- [ ] All system tests pass with `uv run pytest tests/system/`.
- [ ] No test requires a connected camera.

## Implementation Plan

### Approach

1. Create `tests/system/__init__.py`.
2. Create `tests/system/test_detection_pipeline_videos.py`.
3. Use `@pytest.mark.parametrize` over the four video filenames.
4. Each parametrized test:
   a. Construct `VideoCamera(path)`.
   b. Construct `Playfield(video_cam, width_cm=100, height_cm=80)`.
   c. Call `field.start()`.
   d. Collect frames via `field.stream()` until the generator terminates (EOF).
   e. Call `field.stop()`.
   f. Assert detection results.
5. Use a shared `conftest.py` fixture for the movies directory path.

### Files to Create

- `tests/system/__init__.py`
- `tests/system/test_detection_pipeline_videos.py`
- `tests/conftest.py` (if not already created in ticket 013/014) — movies path fixture.

### Key Implementation Notes

- `field.stream()` terminates when `VideoCamera.read()` returns `None` (EOF);
  the generator loop must check for this.
- Collect all `Tag` objects across all frames into a set by ID.
- For dim videos: if detection genuinely fails (dim videos may be very dark),
  mark with `@pytest.mark.xfail(strict=False, reason="dim lighting may not detect")`.
  Update once actual detection results are known.
- Frame count: videos may have hundreds of frames; the test should complete in
  under 60 seconds total for all four.
- Detection assertions: assert `len(all_detected_ids) >= 1` — at least one tag
  seen across all frames of the video.

### Testing Plan

- Verification: `uv run pytest tests/system/ -v` green (or xfail for dim if expected).
- Run time target: under 60 seconds total for all four videos.

### Documentation Updates

- Comment in test file explaining the parametrize strategy and the dim-lighting
  xfail rationale.
