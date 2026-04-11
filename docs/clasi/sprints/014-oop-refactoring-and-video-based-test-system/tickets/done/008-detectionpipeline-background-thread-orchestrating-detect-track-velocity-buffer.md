---
id: "008"
title: 'DetectionPipeline: background thread orchestrating detect-track-velocity-buffer'
status: done
use-cases:
  - SUC-002
  - SUC-004
  - SUC-005
  - SUC-006
  - SUC-007
depends-on:
  - "002"
  - "003"
  - "006"
  - "007"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# DetectionPipeline: background thread orchestrating detect-track-velocity-buffer

## Description

Create `src/aprilcam/core/pipeline.py` with `DetectionPipeline`.

`DetectionPipeline` runs a background thread that continuously reads frames
from a `Camera` (or `VideoCamera`), runs `TagDetector` every N frames,
runs `OpticalFlowTracker` on all frames, updates per-tag `VelocityEstimator`
instances, and writes results to the `RingBuffer`. It evolves from the existing
`DetectionLoop` in `detection.py`, composing the new extracted classes.

Callers start/stop the pipeline and register `on_frame` callbacks. The
`Playfield` class (ticket 010) owns and controls a `DetectionPipeline`.

## Acceptance Criteria

- [ ] `core/pipeline.py` exists with `DetectionPipeline` class.
- [ ] `DetectionPipeline(camera, detector, tracker, *, detect_interval: int = 3,
      ring_buffer_size: int = 300)` constructor.
- [ ] `start()` — starts the background thread; idempotent if already running.
- [ ] `stop()` — signals thread to stop; joins thread; idempotent.
- [ ] `is_running: bool` property.
- [ ] `ring_buffer: RingBuffer` property — the live ring buffer.
- [ ] `on_frame(callback: Callable[[list[TagRecord]], None])` — registers a
      callback invoked on the background thread after each frame is processed.
- [ ] Pipeline loop: read frame → grayscale → detect (every N frames) →
      track → per-tag velocity → build `TagRecord` list → write to ring buffer →
      call callbacks.
- [ ] `VelocityEstimator` instances created per tag ID; recycled across frames.
- [ ] Thread-safe: `ring_buffer` reads from the main thread are safe while
      the pipeline writes from the background thread.
- [ ] `core/__init__.py` exports `DetectionPipeline`.

## Implementation Plan

### Approach

1. Create `pipeline.py` with `DetectionPipeline`.
2. Lift the core loop from `detection.py`'s `DetectionLoop._run_loop()` and
   `aprilcam.py`'s `process_frame()`.
3. Replace inline detection/tracking calls with `TagDetector.detect()` and
   `OpticalFlowTracker.update()`.
4. Replace inline velocity EMA with `VelocityEstimator.update()`.
5. Per-tag `VelocityEstimator` map: `dict[int, VelocityEstimator]`.
6. Existing `DetectionLoop` in `detection.py` is kept for now (compatibility shim).

### Files to Create

- `src/aprilcam/core/pipeline.py`

### Files to Modify

- `src/aprilcam/core/__init__.py` — export `DetectionPipeline`
- `src/aprilcam/core/aprilcam.py` — optionally delegate main loop to `DetectionPipeline`

### Key Implementation Notes

- Use `threading.Thread(daemon=True)` for the background thread.
- Stop signal: `threading.Event` set by `stop()`, checked in loop.
- Ring buffer writes are atomic per frame (lock held only during list append).
- `on_frame` callbacks run synchronously inside the loop; keep them fast.
- `VideoCamera.read()` returning `None` means EOF — pipeline should stop cleanly.

### Testing Plan

- Smoke: `from aprilcam.core import DetectionPipeline` succeeds.
- Unit: `pipeline.start()` sets `is_running = True`.
- Unit: `pipeline.stop()` after start sets `is_running = False`; thread joined.
- Unit: with `VideoCamera`, pipeline processes all frames and stops at EOF.
- Unit: `on_frame` callback called at least once for a multi-frame video.
- Unit: `ring_buffer` contains `TagRecord` entries after processing.

### Documentation Updates

- Docstrings on `DetectionPipeline`, `start()`, `stop()`, `on_frame()`, `ring_buffer`.
