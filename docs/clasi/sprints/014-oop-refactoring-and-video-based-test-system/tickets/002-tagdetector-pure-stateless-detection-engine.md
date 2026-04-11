---
id: '002'
title: 'TagDetector: pure stateless detection engine'
status: in-progress
use-cases:
- SUC-003
depends-on: []
github-issue: ''
todo:
- docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# TagDetector: pure stateless detection engine

## Description

Create `src/aprilcam/core/detector.py` with `DetectorConfig` and `TagDetector`.

`TagDetector` is a pure, stateless detection engine: given a BGR frame, return
detections. No camera, no threads, no ring buffer. This extracts and consolidates
`AprilCam._build_detectors()`, `_maybe_preprocess()`, `detect_apriltags()`, and
the duplicate module-level versions of those functions from `aprilcam.py`.

## Acceptance Criteria

- [ ] `core/detector.py` exists with `DetectorConfig` dataclass and `TagDetector` class.
- [ ] `DetectorConfig` fields: `family`, `quad_decimate`, `quad_sigma`, `corner_refine`,
      `detect_inverted`, `use_clahe`, `use_sharpen`, `use_highpass`, `highpass_ksize`,
      `april_min_wb_diff`, `april_min_cluster_pixels`, `april_max_line_fit_mse`,
      `detect_aruco_4x4`, `proc_width` — all with sensible defaults.
- [ ] `TagDetector(config: DetectorConfig | None = None)` constructs detector objects.
- [ ] `TagDetector.detect(frame_bgr: np.ndarray) -> list[Detection]` runs preprocessing
      and detection; returns list of `Detection` objects.
- [ ] `Detection` dataclass: `id: int`, `center: tuple[float, float]`,
      `corners: np.ndarray` (4x2), `family: str`.
- [ ] Calling `detect()` twice on the same frame returns equivalent results (stateless).
- [ ] Duplicate module-level `build_detectors()` / `detect_apriltags()` removed from
      `aprilcam.py`; `aprilcam.py` delegates to `TagDetector` instead.
- [ ] `core/__init__.py` exports `TagDetector`, `DetectorConfig`.

## Implementation Plan

### Approach

1. Define `Detection` dataclass in `detector.py`.
2. Define `DetectorConfig` dataclass with all tuning params.
3. `TagDetector.__init__` builds detector objects (lifted from `_build_detectors()`).
4. `TagDetector.detect()` runs preprocessing + detection (lifted from
   `_maybe_preprocess()` + `detect_apriltags()`).
5. Update `aprilcam.py`: create a `TagDetector` in `__init__`, delegate to it.
6. Remove duplicate module-level functions from `aprilcam.py`.

### Files to Create

- `src/aprilcam/core/detector.py`

### Files to Modify

- `src/aprilcam/core/aprilcam.py` — delegate to `TagDetector`; remove duplicates
- `src/aprilcam/core/__init__.py` — export `TagDetector`, `DetectorConfig`

### Key Implementation Notes

- `family="all"` means build one detector per family, run all, merge results.
- The preprocessing pipeline (highpass, CLAHE, sharpen, resize) lives inside
  `detect()` — no separate public method needed.
- `TagDetector` stores the built detector objects as instance attributes; only
  `detect()` is called per-frame.

### Testing Plan

- Smoke: `from aprilcam.core import TagDetector` succeeds.
- Smoke: `TagDetector()` constructs without error.
- Unit: `TagDetector().detect(np.ones((480,640,3), dtype=np.uint8)*255)` returns `[]`.
- Unit: calling `detect()` twice returns equivalent results.
- Unit: `DetectorConfig()` has expected default values.

### Documentation Updates

- Docstrings on `DetectorConfig`, `TagDetector`, `detect()`, `Detection`.
