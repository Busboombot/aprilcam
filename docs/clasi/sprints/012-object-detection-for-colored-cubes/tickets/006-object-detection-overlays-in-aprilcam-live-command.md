---
id: "006"
title: "Object detection overlays in aprilcam live command"
status: todo
use-cases: ["SUC-012-001"]
depends-on: ["004"]
github-issue: ""
todo: ""
---

# Object detection overlays in aprilcam live command

## Description

Add object detection support to `aprilcam live` so detected colored
cubes are drawn on the live view alongside AprilTag overlays.

### Changes

1. **`src/aprilcam/cli/live_cli.py`**: Add CLI flags:
   - `--detect-objects` / `-d`: enable square detection on each frame
   - `--color-camera <index>`: optional color camera index for color
     classification (runs in background thread via ColorCameraThread)

2. **`src/aprilcam/liveview.py`**: Update `run_live_view()` to accept
   `detect_objects` and `color_camera` params. When enabled:
   - Create a SquareDetector
   - Optionally start a ColorCameraThread with an ObjectFuser
   - On each frame, run square detection, fuse colors if available
   - Draw detected objects on the display frame:
     - Color-coded rectangle around each object (use the object's color,
       or white for "unknown")
     - Color label text above the rectangle
     - World coordinates below if available

3. **Drawing colors**: Map color names to BGR values for drawing:
   - red → (0, 0, 255), green → (0, 255, 0), blue → (255, 0, 0),
     yellow → (0, 255, 255), orange → (0, 165, 255),
     purple → (255, 0, 255), unknown → (255, 255, 255)

## Acceptance Criteria

- [ ] `aprilcam live -c 3 --detect-objects` shows object rectangles
- [ ] `aprilcam live -c 3 -d --color-camera 2` shows colored rectangles
- [ ] Objects drawn with color-coded outlines and labels
- [ ] World coordinates shown when homography is loaded
- [ ] Existing tag overlays are unaffected
- [ ] ColorCameraThread is stopped cleanly on exit
- [ ] All tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Smoke test that the CLI parser accepts the
  new flags without error
- **Verification command**: `uv run pytest`
