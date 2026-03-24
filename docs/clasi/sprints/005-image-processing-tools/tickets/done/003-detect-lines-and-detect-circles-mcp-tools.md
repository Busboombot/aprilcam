---
id: '003'
title: detect_lines and detect_circles MCP tools
status: done
use-cases:
- SUC-002
- SUC-003
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# detect_lines and detect_circles MCP tools

## Description

Create `src/aprilcam/image_processing.py` with pure-CV processing
functions, and register MCP tools that combine `resolve_source` with
these functions:

1. **`process_detect_lines(frame, threshold, min_length, max_gap)`** —
   Converts frame to grayscale, applies Canny edge detection, then
   `cv2.HoughLinesP`. Returns a list of line segment dicts with
   `x1, y1, x2, y2` coordinates. Parameters have sensible defaults.

2. **`process_detect_circles(frame, min_radius, max_radius, param1, param2)`** —
   Converts to grayscale, applies `cv2.HoughCircles` with
   `HOUGH_GRADIENT`. Returns a list of circle dicts with `x, y, radius`.
   Parameters have sensible defaults.

3. **MCP tools `detect_lines` and `detect_circles`** — Each calls
   `resolve_source`, runs the processing function, returns structured
   JSON results with the detected geometry.

## Acceptance Criteria

- [ ] `src/aprilcam/image_processing.py` module created
- [ ] `process_detect_lines()` function accepts frame and optional parameters
- [ ] `process_detect_lines()` returns list of `{"x1", "y1", "x2", "y2"}` dicts
- [ ] `process_detect_lines()` returns empty list when no lines detected
- [ ] `process_detect_circles()` function accepts frame and optional parameters
- [ ] `process_detect_circles()` returns list of `{"x", "y", "radius"}` dicts
- [ ] `process_detect_circles()` returns empty list when no circles detected
- [ ] `detect_lines` MCP tool registered with `source_id` and optional tuning parameters
- [ ] `detect_circles` MCP tool registered with `source_id` and optional tuning parameters
- [ ] Both tools return structured JSON (not images) as their primary output
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**:
  - Unit tests in `tests/test_image_processing.py`:
    - `test_detect_lines_synthetic` — draw known lines on blank image, verify detection
    - `test_detect_lines_empty` — blank image, verify empty list
    - `test_detect_circles_synthetic` — draw known circles, verify center/radius
    - `test_detect_circles_empty` — blank image, verify empty list
  - MCP tool tests verifying resolve_source integration
- **Verification command**: `uv run pytest`
