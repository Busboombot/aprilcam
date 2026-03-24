---
id: '003'
title: "Refactor AprilCam \u2014 extract process_frame()"
status: done
use-cases:
- SUC-005
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Refactor AprilCam — extract process_frame()

## Description

Extract the detection/tracking core from `AprilCam.run()` into a new
public method `process_frame(frame_bgr, timestamp) -> list[TagRecord]`
that can be called by both the existing interactive loop and the new
`DetectionLoop` class.

Currently, `run()` (lines 242-410 of `src/aprilcam/aprilcam.py`)
contains the full pipeline inline: read frame, convert to gray, detect
or track, update playfield, filter detections, update tag models,
compute velocities, print output, render display. The detection/tracking
core (steps 2-6: gray conversion, detection/LK tracking, playfield
update, filtering, tag model updates) needs to be extracted so the
`DetectionLoop` can call it without any display or keyboard logic.

### Changes Required

1. **Move local state to instance attributes** -- The following
   variables currently live as locals in `run()` and must become
   instance attributes initialized in `__init__()` or a new
   `reset_state()` method:
   - `prev_gray` (`Optional[np.ndarray]`)
   - `tracks` (`dict[int, np.ndarray]`)
   - `tag_models` (`dict[int, AprilTagModel]`)
   - `frame_idx` (`int`)
   - `vel_ema` (`dict[int, float]`) -- only needed for print output in `run()`
   - `last_seen` (`dict[int, tuple]`) -- only needed for print output in `run()`

2. **Add `reset_state()` method** -- Resets all tracking state to
   initial values. Called at the start of `run()` and available for
   the `DetectionLoop` to call on loop start.

3. **Add `process_frame(frame_bgr, timestamp) -> list[TagRecord]`** --
   Takes a BGR frame and monotonic timestamp. Performs detection (or
   LK tracking based on `detect_interval` and `frame_idx`), updates
   playfield, filters detections to playfield polygon, updates tag
   models, increments `frame_idx`, and returns a list of `TagRecord`
   objects for the detected tags. Velocity computation should use
   the `AprilTagFlow` data from `self.playfield.get_flows()`.

4. **Simplify `run()`** -- The `run()` loop should call
   `process_frame()` for steps 2-6, then handle display (step 8),
   keyboard input (step 9), and speed printing (step 7) using the
   returned `TagRecord` list.

### Constraints

- `run()` behavior must be identical after refactoring -- same output,
  same display, same keyboard handling.
- `process_frame()` must not open windows, call `cv.waitKey()`, or
  print to stdout.
- `process_frame()` must be safe to call from a non-main thread
  (the `DetectionLoop` thread).

## Acceptance Criteria

- [ ] `reset_state()` method exists on `AprilCam` and resets `prev_gray`, `tracks`, `tag_models`, `frame_idx`
- [ ] `process_frame(frame_bgr, timestamp)` method exists and returns `list[TagRecord]`
- [ ] `process_frame()` performs detection/tracking, playfield filtering, and tag model updates
- [ ] `process_frame()` increments `frame_idx` and updates `prev_gray`
- [ ] `process_frame()` does not open windows, render overlays, or print output
- [ ] `run()` calls `process_frame()` internally -- no detection logic remains duplicated in `run()`
- [ ] `run()` still handles display, keyboard input, pause, and speed printing
- [ ] Existing CLI `aprilcam run` subcommand works identically after refactor (manual smoke test)
- [ ] `process_frame()` returns `TagRecord` objects with velocity fields populated from `AprilTagFlow`
- [ ] All existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite, especially `tests/test_cli_smoke.py`
- **New tests to write** (in `tests/test_detection.py`):
  - `test_process_frame_returns_tagrecords` -- load `tests/data/playfield_cam3.jpg`, create a headless AprilCam with the image as a frame, call `process_frame()`, verify it returns a non-empty list of `TagRecord` objects with valid fields
  - `test_process_frame_increments_frame_idx` -- call process_frame twice, verify `frame_idx` goes from 0 to 2
  - `test_reset_state` -- process some frames, call `reset_state()`, verify `frame_idx` is 0 and `tracks` is empty
  - `test_process_frame_no_display_side_effects` -- verify that no OpenCV windows are created during process_frame (headless mode)
- **Verification command**: `uv run pytest tests/test_detection.py tests/test_cli_smoke.py -v`
