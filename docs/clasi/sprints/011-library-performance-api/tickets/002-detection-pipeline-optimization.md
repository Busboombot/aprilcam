---
id: "002"
title: "Detection pipeline optimization"
status: todo
use-cases: [SUC-011-001]
depends-on: []
github-issue: ""
todo: ""
---

# Detection pipeline optimization

## Description

Eliminate redundant CPU work in the frame processing pipeline.

### Changes

1. **Grayscale once**: In `process_frame()`, convert BGR→gray once
   (line ~401). Pass the gray image to `detect_apriltags()` and
   `playfield.update()` instead of letting each re-convert.

2. **`detect_apriltags()` signature**: Add optional `gray` param.
   If provided, skip internal `cvtColor`. If None, convert internally
   for backward compat.

3. **`Playfield.update()` signature**: Add optional `gray` param.
   If provided, skip internal `cvtColor`.

4. **Cache ArUco detector**: `Playfield._build_aruco4_detector()` is
   called every frame in `_detect_corners()`. Cache the detector as
   `self._aruco_detector` on first call.

5. **Throttle corner re-detection**: Add `corner_detect_interval`
   parameter (default 30 = ~1/sec at 30fps). Only run ArUco corner
   detection every Nth frame. On non-detection frames, keep the
   existing polygon.

## Acceptance Criteria

- [ ] `cvtColor(BGR2GRAY)` called exactly once per frame in the
      detection pipeline
- [ ] ArUco detector is cached (not rebuilt per call)
- [ ] Playfield corner detection runs every Nth frame (configurable)
- [ ] All existing tests pass with no regressions

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test that gray param is respected; test
  corner detection interval logic
- **Verification command**: `uv run pytest`
