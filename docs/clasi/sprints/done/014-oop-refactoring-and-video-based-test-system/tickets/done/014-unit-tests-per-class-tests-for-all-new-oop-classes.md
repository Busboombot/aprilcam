---
id: '014'
title: 'Unit tests: per-class tests for all new OOP classes'
status: done
use-cases:
  - SUC-010
depends-on:
  - "013"
github-issue: ''
todo:
  - docs/clasi/todo/video-based-test-system.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Unit tests: per-class tests for all new OOP classes

## Description

Create `tests/unit/` with one test file per major class. Unit tests verify
constructor, methods, properties, and edge cases in isolation. Hardware is not
required; `VideoCamera` or synthetic frames are used where needed.

These tests form the regression safety net for the OOP refactor.

## Acceptance Criteria

- [ ] `tests/unit/` directory exists with `__init__.py`.
- [ ] `tests/unit/test_camera.py` — tests `Camera.list()`, `Camera.find()`,
      `is_open`, `close()`, context manager.
- [ ] `tests/unit/test_video_camera.py` — tests `VideoCamera` construction,
      sequential `read()`, EOF returns `None`, context manager.
- [ ] `tests/unit/test_tag_detector.py` — tests `DetectorConfig` defaults,
      `TagDetector()` construction, `detect()` on blank frame returns `[]`,
      stateless (same result twice).
- [ ] `tests/unit/test_optical_flow_tracker.py` — tests `update()` with
      detections, `update()` without (propagate), `reset()`.
- [ ] `tests/unit/test_velocity_estimator.py` — tests first-call zero, constant
      velocity, deadband suppression, `predict_position()`, `reset()`.
- [ ] `tests/unit/test_detection_pipeline.py` — tests `start()`, `stop()`,
      `is_running`, `on_frame` callback firing, ring buffer populated; uses
      `VideoCamera`.
- [ ] `tests/unit/test_tag.py` — tests properties when snapshot is None,
      properties after `update()` with seeded ring buffer, `position_at(t)`.
- [ ] `tests/unit/test_tag_record.py` — tests `TagRecord.to_dict()`,
      `TagRecord.estimate(t)` extrapolation.
- [ ] `tests/unit/test_ring_buffer.py` — tests write, read latest, history,
      capacity wrapping.
- [ ] All unit tests pass with `uv run pytest tests/unit/`.
- [ ] No test requires a connected camera.

## Implementation Plan

### Approach

Write focused pytest functions. Use `VideoCamera` for pipeline tests. Use
synthetic NumPy frames for detector and tracker tests. Mock `DetectionPipeline`
for `Tag` tests (inject a fake `RingBuffer` with seeded records).

### Files to Create

- `tests/unit/__init__.py`
- `tests/unit/test_camera.py`
- `tests/unit/test_video_camera.py`
- `tests/unit/test_tag_detector.py`
- `tests/unit/test_optical_flow_tracker.py`
- `tests/unit/test_velocity_estimator.py`
- `tests/unit/test_detection_pipeline.py`
- `tests/unit/test_tag.py`
- `tests/unit/test_tag_record.py`
- `tests/unit/test_ring_buffer.py`

### Key Implementation Notes

- `conftest.py` at `tests/` level: provide `video_path(name)` fixture returning
  `Path("tests/movies/<name>")`.
- `TagDetector` test: `np.ones((480, 640, 3), dtype=np.uint8) * 255` is a
  blank white frame — no tags will be detected.
- `OpticalFlowTracker` test: create two identical grayscale frames (zero shift)
  and verify `update()` returns same number of detections.
- `DetectionPipeline` test: use `bright-gsc.mov` via `VideoCamera`; call
  `start()`; wait for pipeline to process all frames; call `stop()`; assert
  `ring_buffer` is non-empty.
- `Tag` test: instantiate with a stub `DetectionPipeline` whose `ring_buffer`
  contains a seeded `TagRecord`; call `update()`; assert properties match.

### Testing Plan

- Verification: `uv run pytest tests/unit/ -v` all green.
- Aim for under 30 seconds total (video processing in pipeline test may take a
  few seconds).

### Documentation Updates

- `tests/unit/conftest.py` if created: docstring explaining shared fixtures.
