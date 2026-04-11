---
id: '005'
title: 'Calibration split: extract calibrate() and types from homography.py'
status: done
use-cases:
- SUC-009
depends-on: []
github-issue: ''
todo:
- docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Calibration split: extract calibrate() and types from homography.py

## Description

Create `src/aprilcam/calibration/calibration.py` by splitting the current
`homography.py`. Pure math (homography matrix computation, coordinate
transforms) stays in `homography.py`. The calibration workflow (`calibrate()`
function), `CameraCalibration` dataclass, and `FieldSpec` dataclass move to
`calibration.py`.

`calibration.py` owns: detect ArUco corners, compute homography, persist JSON,
load JSON. `homography.py` owns: matrix math, coordinate transforms.

## Acceptance Criteria

- [x] `calibration/calibration.py` exists with `calibrate()`, `CameraCalibration`,
      `FieldSpec`.
- [x] `calibrate()` function: detect corners, compute homography, write JSON to disk.
- [x] `CameraCalibration` dataclass: contains `homography: np.ndarray`, `field_spec: FieldSpec`,
      `device_name: str`, `resolution: tuple[int, int]`.
- [x] `FieldSpec` dataclass: `width_cm`, `height_cm` (and unit-conversion helpers).
- [x] `homography.py` retains only pure math functions (no `calibrate()`, no `FieldSpec`,
      no `CameraCalibration`).
- [x] All existing callers of `CameraCalibration`, `FieldSpec`, `calibrate()`
      updated to import from `calibration.calibration` (not `calibration.homography`).
- [x] `calibration/__init__.py` exports `calibrate`, `CameraCalibration`, `FieldSpec`.
- [x] `from aprilcam.calibration import calibrate` works at the package level.

## Implementation Plan

### Approach

1. Create `calibration/calibration.py`.
2. Move `FieldSpec`, `CameraCalibration` from `homography.py` to `calibration.py`.
3. Move the `calibrate()` function from `homography.py` (or `stream.py`) to
   `calibration.py`.
4. Add re-exports in `homography.py` for backward compat during transition:
   `from .calibration import CameraCalibration, FieldSpec`.
5. Update `calibration/__init__.py`.

### Files to Create

- `src/aprilcam/calibration/calibration.py`

### Files to Modify

- `src/aprilcam/calibration/homography.py` — remove moved items; add re-exports
- `src/aprilcam/calibration/__init__.py` — export new names
- `src/aprilcam/stream.py` — update import of `calibrate` (if needed before deletion)

### Key Implementation Notes

- `FieldSpec` conversion helpers (`width_cm`, `height_cm` properties from inches)
  stay on `FieldSpec`.
- `calibrate()` reads from a `Camera` (or path) and a `FieldSpec`, writes
  `CameraCalibration` to JSON.
- The existing `load_calibration_for_camera()` may stay in `homography.py` or
  move to `calibration.py` — move it to `calibration.py` since it is part of
  the calibration workflow.

### Testing Plan

- Smoke: `from aprilcam.calibration import calibrate, CameraCalibration, FieldSpec` succeeds.
- Unit: `FieldSpec(width_in=40, height_in=35, units="inch").width_cm` returns correct value.
- Unit: `CameraCalibration` can be round-tripped to/from JSON dict.

### Documentation Updates

- Docstrings on `calibrate()`, `CameraCalibration`, `FieldSpec`.
