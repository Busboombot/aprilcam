---
id: "003"
title: "detect_tags generator API"
status: todo
use-cases: [SUC-011-001]
depends-on: ["001", "002"]
github-issue: ""
todo: ""
---

# detect_tags generator API

## Description

Create a `detect_tags()` generator function as the primary library
interface. One call opens a camera, loads homography, and yields
tag records per frame.

### Changes

1. **New `aprilcam/stream.py`**: Module with `detect_tags()` generator.

   ```python
   def detect_tags(
       camera: int | str = 0,
       homography: str | Path | None = "auto",
       family: str = "36h11",
       data_dir: str | Path = "data",
       proc_width: int = 0,
   ) -> Generator[list[TagRecord], None, None]:
   ```

   - `camera`: index (int) or device name pattern (str)
   - `homography="auto"`: discover per-camera file from `data_dir`
   - Opens camera via `AprilCam`, calls `process_frame()` in a loop
   - Yields `List[TagRecord]` per frame (with world coords, velocity,
     orientation)
   - On generator close: releases camera
   - Also usable as context manager

2. **`__init__.py`**: Export `detect_tags`.

## Acceptance Criteria

- [ ] `for tags in detect_tags(camera=0): ...` works
- [ ] Auto-loads per-camera homography when `homography="auto"`
- [ ] Returns pixel-only tags when no homography found
- [ ] Camera released on generator close or exception
- [ ] Tags include world_xy, velocity, orientation when homography
      is available

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test generator yields TagRecords from test
  images; test auto-discovery of homography; test cleanup on close
- **Verification command**: `uv run pytest`
