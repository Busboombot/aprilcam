---
id: '006'
title: 'OpticalFlowTracker: LK tracking with 4-corner motion decomposition'
status: todo
use-cases:
  - SUC-004
depends-on:
  - "002"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# OpticalFlowTracker: LK tracking with 4-corner motion decomposition

## Description

Create `src/aprilcam/core/tracker.py` with `OpticalFlowTracker`.

`OpticalFlowTracker` applies Lucas-Kanade optical flow to the 4 corners of each
known tag between full-detection frames. It produces a full per-tag motion
decomposition: 2D translation, rotation rate (yaw delta), and scale delta
(z-axis proxy). All 4-corner data is preserved — no information is discarded
to just center position.

This absorbs `AprilCam.lk_track()` and the detect-or-track branching logic
from `process_frame()` in `aprilcam.py`.

## Acceptance Criteria

- [ ] `core/tracker.py` exists with `OpticalFlowTracker` class.
- [ ] `OpticalFlowTracker()` default constructor.
- [ ] `update(gray: np.ndarray, detections: list[Detection] | None) -> list[Detection]`:
      - When `detections` is not None: reset tracked points to new detections.
      - When `detections` is None: propagate existing tracked points via LK flow.
      - Returns list of `Detection` objects with updated `corners` and `center`.
- [ ] Output `Detection` objects include a `motion` field (or updated corners)
      preserving 2D translation, rotation rate, and scale delta from 4-corner tracking.
- [ ] `OpticalFlowTracker.reset()` clears all tracked state.
- [ ] Absorbs `AprilCam.lk_track()` and detect-or-track branching from `process_frame()`.
- [ ] `aprilcam.py` delegates to `OpticalFlowTracker`; no duplicate LK code remains.
- [ ] `core/__init__.py` exports `OpticalFlowTracker`.

## Implementation Plan

### Approach

1. Create `tracker.py`. Define `OpticalFlowTracker`.
2. Lift `lk_track()` from `aprilcam.py` (~lines 400-480) into `update()`.
3. Lift the detect-or-track branching from `process_frame()` into `update()`.
4. Extend `Detection` (from ticket 002) with optional `translation`, `rotation_rate`,
   `scale_delta` fields (all default to None for standard detections).
5. Update `aprilcam.py` to instantiate `OpticalFlowTracker` and delegate.

### Files to Create

- `src/aprilcam/core/tracker.py`

### Files to Modify

- `src/aprilcam/core/detector.py` — extend `Detection` with motion fields (or add `TrackedDetection` subtype)
- `src/aprilcam/core/aprilcam.py` — delegate tracking to `OpticalFlowTracker`
- `src/aprilcam/core/__init__.py` — export `OpticalFlowTracker`

### Key Implementation Notes

- LK params: `winSize=(21, 21)`, `maxLevel=3`, standard termination criteria.
- 4-corner tracking: store `prev_gray` and `prev_pts` (N×4×2 array) as instance state.
- Rotation rate from 4-corner: fit rigid body transform (translation + rotation)
  using `cv.estimateAffinePartial2D` on the corner pairs.
- Scale delta: ratio of mean corner span before vs. after.
- When LK fails for a point (status=0), fall back to center-only approximation
  rather than dropping the tag entirely.

### Testing Plan

- Smoke: `from aprilcam.core import OpticalFlowTracker` succeeds.
- Unit: `update(gray, detections=[...])` resets and returns same detections with no motion.
- Unit: `update(gray2, None)` propagates points and returns updated positions.
- Unit: `reset()` clears state so next `update(gray, None)` returns empty list.
- Use two synthetic grayscale frames with known shift for the propagation test.

### Documentation Updates

- Docstrings on `OpticalFlowTracker`, `update()`, `reset()`.
