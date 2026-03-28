---
status: pending
---

# Library Performance, API Surface, and Camera Contention

Improve AprilCam's library interface for clients that import it
directly (not via MCP). The goals: faster detection, simpler API,
per-camera homography persistence, and better camera error reporting.

## 1. Per-Camera Homography Persistence

**Problem**: Only one global `data/homography.json` exists. Multi-camera
setups require re-calibrating every session because the file gets
overwritten.

**Current state**:
- `config.py:150-164` — `AppConfig.load_homography()` loads from a
  single hardcoded path
- `homography.py:254-266` — writes JSON with camera source metadata
  but doesn't use it for file naming
- The JSON already contains a `source` field with camera index, backend,
  and resolution — this metadata exists but is unused

**Proposed fix**:
- Name files by device name + resolution (cross-platform):
  `data/homography-brio-501-1920x1080.json`
- Device name comes from ffmpeg on macOS (already implemented in
  `camutil._macos_avfoundation_device_names()`), or
  `v4l2-ctl --list-devices` on Linux
- Slugify the name: lowercase, spaces→hyphens, strip special chars
- If two identical cameras exist (same name + resolution), append
  index as tiebreaker: `data/homography-brio-501-1920x1080-1.json`
- On `open_camera`, auto-discover matching homography file from `data/`
- Add a `load_homography(device_name, resolution)` that finds the
  right file
- Keep backward compat: `data/homography.json` still works as fallback
- Document the `data/` directory convention so agents know where to look

**Effort**: Small — mostly file naming and a lookup function.

## 2. Detection Pipeline Speed

**Problem**: Redundant work in the frame processing pipeline.

**Analysis of current `process_frame()` path** (aprilcam.py:382-493):
```
cap.read() → frame_bgr
  → cvtColor(BGR→GRAY)          # line 401
  → detect_apriltags(frame_bgr)
      → cvtColor(BGR→GRAY)      # line 316 — REDUNDANT
      → optional preprocessing (CLAHE, sharpen)
      → ArucoDetector.detectMarkers(gray)
  → playfield.update(frame)
      → cvtColor(BGR→GRAY)      # playfield.py:44 — REDUNDANT AGAIN
      → ArUco corner detection
  → homography transform per tag  # efficient (numpy matmul)
  → velocity EMA                  # efficient
```

**Bottleneck**: Grayscale conversion happens 2-3 times per frame.
On 4K at 30fps this is ~7-10% wasted CPU.

**Proposed fixes**:
- Convert to grayscale once in `process_frame()`, pass gray to all
  downstream functions (`detect_apriltags()`, `playfield.update()`)
- Change `detect_apriltags(frame_bgr)` signature to accept gray directly
- Change `Playfield.update(frame_bgr)` to accept optional gray param
- Profile the full pipeline with cProfile to find other bottlenecks

**Other speed opportunities**:
- ArUco 4x4 detection for playfield corners runs every frame now
  (after the continuous re-detection fix). Consider running it every
  Nth frame since corners rarely move fast
- `_build_aruco4_detector()` creates a new detector object every call
  in Playfield — cache it
- The `_maybe_preprocess()` chain allocates intermediate arrays;
  could be done in-place

## 3. Generator/Streaming Library Interface

**Problem**: No simple, fast iteration API. Library users must manually
manage `AprilCam`, `VideoCapture`, and `process_frame()` state.

**Current API** (ugly, requires private method):
```python
cam = AprilCam(index=0, homography=H, headless=True)
cap = cam._init_capture()   # underscore = private
cam.reset_state()
while cap.isOpened():
    ret, frame = cap.read()
    tags = cam.process_frame(frame, time.monotonic())
```

**Proposed**: A generator that yields tag records per frame:
```python
from aprilcam import detect_tags

for tags in detect_tags(camera=0, homography="auto"):
    for t in tags:
        print(f"Tag {t.id} at world ({t.world_x:.1f}, {t.world_y:.1f})")
```

Design notes:
- `detect_tags()` opens camera, loads per-camera homography from
  `data/`, creates AprilCam, yields `List[TagRecord]` per frame
- Maintains velocity/orientation state internally
- Context manager for cleanup: `with detect_tags(...) as stream:`
- `homography="auto"` means look up from `data/` by camera identity
- Should be the primary recommended API for library users
- Also expose lower-level `AprilCam.frames()` generator for custom
  pipelines

## 4. Camera Contention Errors

**Problem**: When a camera is in use by another process, OpenCV silently
fails (`cap.isOpened()` returns False). The user sees "Failed to open
camera" with no explanation.

**Current state** (aprilcam.py:354-365):
```python
if not self.cap or not self.cap.isOpened():
    print("Failed to open camera.")  # ← only prints, no detail
    return None
```

**Proposed fixes**:
- On macOS: use `lsof` or IOKit to find processes holding camera FDs
- On Linux: check `/dev/video*` locks, use `fuser`
- Produce error messages like:
  ```
  Camera 0 is in use by process 'python3' (PID 12345).
  Kill it with: kill 12345
  ```
- Distinguish between "camera doesn't exist", "camera busy",
  "permission denied"
- Raise proper exceptions instead of printing and returning None
- Add a `--force` or contention-resolution option that can kill the
  blocking process (with confirmation)

## 5. Public API and Exports

**Problem**: `__init__.py` exports nothing useful. Library users must
know internal module paths.

**Current** (`__init__.py`):
```python
__all__ = ["__version__"]
```

**Proposed**: Export the key classes and the generator function:
```python
from aprilcam.aprilcam import AprilCam
from aprilcam.detection import TagRecord, DetectionLoop
from aprilcam.models import AprilTag
from aprilcam.playfield import Playfield

__all__ = [
    "AprilCam", "TagRecord", "DetectionLoop",
    "AprilTag", "Playfield", "detect_tags",
]
```

## 6. Agent Instructions Update

After implementing the above, update agent-facing documentation so
that AI agents using AprilCam as a library (or via MCP) understand:
- The `data/` directory convention for homography files
- The recommended `detect_tags()` generator API
- How to handle camera contention errors
- The available public imports
