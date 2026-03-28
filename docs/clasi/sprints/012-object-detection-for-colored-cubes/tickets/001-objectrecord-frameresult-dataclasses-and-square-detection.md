---
id: "001"
title: "ObjectRecord, FrameResult dataclasses and square detection"
status: todo
use-cases: ["SUC-012-001"]
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# ObjectRecord, FrameResult dataclasses and square detection

## Description

Introduce the core data structures and detection logic for non-tag objects
(colored cubes). This ticket creates `src/aprilcam/objects.py` containing
three components:

### ObjectRecord (frozen dataclass)

A single detected object in one frame. Fields:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `center_px` | `tuple[float, float]` | required | Pixel coordinates (x, y) of the object center |
| `world_xy` | `tuple[float, float] \| None` | `None` | World coordinates if homography/calibration available |
| `color` | `str` | `"unknown"` | Color label (e.g., "red", "blue") |
| `bbox` | `tuple[int, int, int, int]` | required | Bounding box as (x, y, w, h) |
| `area_px` | `float` | required | Contour area in pixels |
| `object_type` | `str` | `"cube"` | Object classification |
| `confidence` | `float` | `1.0` | Detection confidence (0.0-1.0) |

### FrameResult class

A unified container for one frame's detections, holding both tags and
objects. Designed for backward compatibility with code that currently
expects `list[TagRecord]`.

- `tags`: `list[TagRecord]` -- detected AprilTags/ArUco markers
- `objects`: `list[ObjectRecord]` -- detected non-tag objects
- `timestamp`: `float` -- time.time() when the frame was processed
- `frame_index`: `int` -- sequential frame counter

Backward compatibility protocol:
- `__iter__` yields from `self.tags` so existing `for tag in result` loops work
- `__len__` returns `len(self.tags)` so `if result:` checks tag presence
- `__getitem__` delegates to `self.tags` for index access

### SquareDetector class

Detects square-ish contours in a grayscale frame. Constructor parameters:

- `min_area`: minimum contour area in pixels (default 200)
- `max_area`: maximum contour area in pixels (default 5000)

`detect(gray, homography=None, tag_corners=None, exclusion_point=None, exclusion_radius=50)` method:

1. Apply adaptive threshold (Gaussian, block_size=11, C=2) to grayscale input
2. Find contours (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
3. Filter contours by:
   - Area within `[min_area, max_area]`
   - Aspect ratio of bounding rect < 2.0 (roughly square)
   - Solidity > 0.7 (contour area / convex hull area)
4. Exclude contours whose center falls within any provided tag corner polygon
   (tag_corners is a list of Nx4x2 arrays from AprilTag detection)
5. Exclude contours whose center is within `exclusion_radius` pixels of
   `exclusion_point` (e.g., robot position)
6. Return `list[ObjectRecord]` with `color="unknown"`. If `homography` is
   provided, compute `world_xy` by applying the homography to `center_px`.

### process_frame() update

Update `aprilcam/aprilcam.py` `process_frame()` to accept an optional
`detect_squares=False` parameter. When `True`:

- Run `SquareDetector.detect()` on the already-computed grayscale frame
- Pass detected tag corners as exclusion regions
- Return results that include both tags and detected objects

## Acceptance Criteria

- [ ] `ObjectRecord` is a frozen dataclass with all specified fields and defaults
- [ ] `FrameResult` holds both `tags` and `objects` lists with `timestamp` and `frame_index`
- [ ] `iter(FrameResult(...))` yields tags (backward compat)
- [ ] `len(FrameResult(...))` returns number of tags (backward compat)
- [ ] `FrameResult(...)[0]` returns the first tag (backward compat)
- [ ] `SquareDetector` finds square contours on synthetic grayscale images
- [ ] `SquareDetector` excludes regions overlapping with provided tag corners
- [ ] `SquareDetector` excludes regions within `exclusion_radius` of `exclusion_point`
- [ ] `SquareDetector` respects configurable `min_area` and `max_area`
- [ ] `process_frame()` accepts `detect_squares` param without breaking existing callers
- [ ] All new code has type hints
- [ ] All tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- verify no regressions in tag detection or streaming
- **New tests to write**: `tests/test_objects.py` containing:
  - `test_object_record_fields` -- verify frozen dataclass, defaults, and field access
  - `test_object_record_immutable` -- verify assignment raises FrozenInstanceError
  - `test_frame_result_iter_yields_tags` -- create FrameResult with tags and objects, iterate, confirm only tags yielded
  - `test_frame_result_len_returns_tag_count` -- len() returns tag count, not object count
  - `test_frame_result_getitem` -- index access returns tags
  - `test_square_detector_synthetic` -- draw white squares on black background, verify detection
  - `test_square_detector_excludes_tags` -- provide tag corner polygons overlapping a square, verify it is excluded
  - `test_square_detector_exclusion_radius` -- place square near exclusion point, verify excluded
  - `test_square_detector_aspect_ratio_filter` -- draw a long rectangle, verify it is rejected
  - `test_square_detector_solidity_filter` -- draw irregular shape, verify rejected
  - `test_square_detector_area_bounds` -- draw tiny and huge squares, verify filtered by min/max area
  - `test_square_detector_homography` -- provide a homography matrix, verify world_xy is populated
- **Verification command**: `uv run pytest tests/test_objects.py -v`
