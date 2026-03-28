---
id: "002"
title: "HSV color classifier with configurable color ranges"
status: todo
use-cases: ["SUC-012-001"]
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# HSV color classifier with configurable color ranges

## Description

Add a `ColorClassifier` class to `src/aprilcam/objects.py` that identifies
colored objects in BGR frames using HSV color space thresholding. This is the
color detection counterpart to ticket 001's shape-based SquareDetector. The
classifier is designed for the dual-camera workflow: the color camera provides
color labels that are later fused onto B&W camera detections.

### ColorClassifier class

**Constructor**: `ColorClassifier(color_ranges=None, min_area=200, max_area=5000)`

- `color_ranges`: optional `dict[str, list[tuple[tuple[int,int,int], tuple[int,int,int]]]]`
  mapping color names to lists of HSV range pairs (lower, upper). Each color
  can have multiple ranges to handle hue wraparound (e.g., red).
- `min_area` / `max_area`: contour area filter bounds (pixels).

**Default color ranges** (HSV, OpenCV convention where H is 0-180):

| Color  | Range(s) |
|--------|----------|
| red    | (0, 50, 50)-(12, 255, 255) and (165, 50, 50)-(180, 255, 255) |
| green  | (35, 50, 50)-(85, 255, 255) |
| blue   | (90, 50, 50)-(130, 255, 255) |
| yellow | (15, 50, 50)-(35, 255, 255) |
| orange | (12, 80, 80)-(22, 255, 255) |
| purple | (125, 40, 40)-(160, 255, 255) |

**`classify(frame_bgr, homography=None) -> list[ObjectRecord]`**:

1. Convert `frame_bgr` to HSV using `cv2.cvtColor`
2. For each color in `color_ranges`:
   a. Create a combined mask by OR-ing `cv2.inRange` for each (lower, upper) pair
   b. Apply morphological open (3x3 kernel) to reduce noise
   c. Find contours (RETR_EXTERNAL)
   d. Filter contours by area within `[min_area, max_area]`
   e. For each passing contour, compute center from moments, bounding rect,
      and area. If `homography` is provided, compute `world_xy`.
   f. Create `ObjectRecord` with the color label
3. Return all detected ObjectRecords across all colors

**`classify_at_point(frame_bgr, x, y, radius=20) -> str`**:

1. Extract a circular ROI of the given radius around (x, y)
2. Convert to HSV
3. For each color, compute the percentage of pixels in that color's range
4. Return the color name with the highest percentage, or `"unknown"` if no
   color exceeds 10% of the ROI pixels

This method is useful for classifying the color of an already-detected object
at a known position.

## Acceptance Criteria

- [ ] `ColorClassifier` class exists in `src/aprilcam/objects.py`
- [ ] Default constructor provides all 6 color ranges (red, green, blue, yellow, orange, purple)
- [ ] Red range handles hue wraparound with two sub-ranges
- [ ] `classify()` returns `list[ObjectRecord]` with correct color labels
- [ ] `classify()` applies morphological open to reduce noise
- [ ] `classify()` filters contours by min_area and max_area
- [ ] `classify()` computes world_xy when homography is provided
- [ ] Custom `color_ranges` override the defaults entirely
- [ ] `classify_at_point()` returns the dominant color name at a given pixel location
- [ ] `classify_at_point()` returns `"unknown"` when no color exceeds 10% threshold
- [ ] All new code has type hints
- [ ] All tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- verify no regressions
- **New tests to write**: `tests/test_color_classifier.py` containing:
  - `test_classify_red_patch` -- create a 100x100 BGR image with a pure red square, verify classified as "red"
  - `test_classify_green_patch` -- same for green
  - `test_classify_blue_patch` -- same for blue
  - `test_classify_yellow_patch` -- same for yellow
  - `test_classify_multiple_colors` -- image with red and blue squares, verify both detected with correct labels
  - `test_classify_returns_object_records` -- verify returned items are ObjectRecord instances with correct fields
  - `test_classify_with_homography` -- provide identity homography, verify world_xy populated
  - `test_classify_area_filter` -- tiny colored dot below min_area is not detected
  - `test_custom_color_ranges` -- pass custom ranges, verify only those colors are detected
  - `test_classify_at_point_red` -- red patch, query center, returns "red"
  - `test_classify_at_point_unknown` -- gray patch, query center, returns "unknown"
  - `test_classify_at_point_out_of_bounds` -- query near edge, does not crash
- **Verification command**: `uv run pytest tests/test_color_classifier.py -v`
