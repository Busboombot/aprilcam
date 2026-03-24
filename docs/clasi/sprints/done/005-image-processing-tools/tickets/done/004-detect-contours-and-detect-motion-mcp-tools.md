---
id: '004'
title: detect_contours and detect_motion MCP tools
status: done
use-cases:
- SUC-004
- SUC-005
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# detect_contours and detect_motion MCP tools

## Description

Add contour detection and motion detection to `image_processing.py`,
plus their MCP tool registrations:

1. **`process_detect_contours(frame, min_area)`** — Converts to
   grayscale, applies threshold, runs `cv2.findContours`. Filters
   contours by `min_area` (default 100 px^2). Returns a list of
   contour dicts, each with `points` (list of [x,y]), `area`, and
   `bounding_box` ({x, y, w, h}).

2. **`process_detect_motion(frame, prev_frame)`** — Computes absolute
   difference between two grayscale frames, thresholds the result,
   finds contours of changed regions. Returns a list of motion region
   dicts with `bounding_box` and `area`.

3. **Server state**: Add a `_motion_prev_frames` dict to the server
   state, keyed by source_id. On each `detect_motion` call, the
   current frame becomes the stored previous frame for the next call.
   First call for a source returns empty (no previous frame).

4. **MCP tools `detect_contours` and `detect_motion`** — Each calls
   `resolve_source`, runs the processing function, returns structured
   JSON.

## Acceptance Criteria

- [ ] `process_detect_contours()` added to `image_processing.py`
- [ ] `process_detect_contours()` returns list with `points`, `area`, `bounding_box` per contour
- [ ] `process_detect_contours()` filters by `min_area` parameter
- [ ] `process_detect_contours()` returns empty list for blank frame
- [ ] `process_detect_motion()` added to `image_processing.py`
- [ ] `process_detect_motion()` returns motion regions with `bounding_box` and `area`
- [ ] `process_detect_motion()` returns empty list when prev_frame is None (first call)
- [ ] `_motion_prev_frames` dict tracks previous frame per source_id
- [ ] `detect_contours` MCP tool registered with `source_id` and optional `min_area`
- [ ] `detect_motion` MCP tool registered with `source_id`
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**:
  - Unit tests in `tests/test_image_processing.py`:
    - `test_detect_contours_synthetic` — draw shapes on blank image, verify contour data
    - `test_detect_contours_min_area` — verify small contours filtered out
    - `test_detect_motion_two_frames` — two frames with known difference, verify regions
    - `test_detect_motion_no_prev` — first call returns empty list
  - MCP tool tests verifying prev_frame state management
- **Verification command**: `uv run pytest`
