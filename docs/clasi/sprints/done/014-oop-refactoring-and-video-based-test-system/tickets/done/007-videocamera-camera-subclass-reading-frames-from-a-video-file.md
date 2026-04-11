---
id: '007'
title: 'VideoCamera: Camera subclass reading frames from a video file'
status: done
use-cases:
  - SUC-002
  - SUC-010
depends-on:
  - "001"
github-issue: ''
todo:
  - docs/clasi/todo/video-based-test-system.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# VideoCamera: Camera subclass reading frames from a video file

## Description

Create `src/aprilcam/camera/video_camera.py` with `VideoCamera(Camera)`.

`VideoCamera` subclasses `Camera` and reads frames sequentially from a `.mov`
(or other OpenCV-readable) video file. It provides the same `read()` / `close()` /
context manager interface as `Camera`. At EOF, `read()` returns `None`.

This enables the full detection pipeline to run against recorded test videos
without any connected camera hardware — the foundation of the video-based test system.

## Acceptance Criteria

- [ ] `camera/video_camera.py` exists with `VideoCamera(Camera)` class.
- [ ] `VideoCamera(path: str | Path)` constructor opens the video file.
- [ ] `video_camera.read() -> np.ndarray | None` returns frames sequentially;
      returns `None` at EOF.
- [ ] `video_camera.is_open` is True while frames remain, False after EOF or close.
- [ ] `video_camera.close()` releases the VideoCapture; idempotent.
- [ ] Context manager: `with VideoCamera(path) as cam:` closes on exit.
- [ ] `video_camera.name` returns the filename stem.
- [ ] `video_camera.index` returns -1 (sentinel for file-based camera).
- [ ] `VideoCamera` does not call `Camera._open()` (no hardware access).
- [ ] `camera/__init__.py` exports `VideoCamera`.
- [ ] `from aprilcam.camera import VideoCamera` works at package level.

## Implementation Plan

### Approach

1. Create `video_camera.py`. `VideoCamera.__init__(path)` opens a
   `cv.VideoCapture(str(path))` directly — bypasses `Camera._open()`.
2. Override `read()` to call `_cap.read()` and return `None` when
   `_cap.read()` fails (EOF).
3. Override `name` property to return `Path(path).stem`.
4. Override `index` property to return `-1`.
5. `close()` inherited from `Camera` works unchanged.
6. Add `VideoCamera` to `camera/__init__.py`.

### Files to Create

- `src/aprilcam/camera/video_camera.py`

### Files to Modify

- `src/aprilcam/camera/__init__.py` — export `VideoCamera`

### Key Implementation Notes

- `VideoCamera.__init__` must NOT call `super().__init__()` in a way that
  triggers hardware probing. Pass `index=-1, name=path.stem` to `super()`.
- The `_cap` attribute is set in `__init__` directly (opened from file path).
- No `_open()` override needed if `_cap` is set before `read()` is called.
- `read()` return: on EOF `ret=False`, return `None`. Caller checks for `None`.

### Testing Plan

- Smoke: `from aprilcam.camera import VideoCamera` succeeds.
- Unit: `VideoCamera("tests/movies/bright-gsc.mov")` constructs.
- Unit: `read()` returns a BGR numpy array on first call.
- Unit: `read()` returns `None` after all frames are exhausted.
- Unit: context manager closes on exit.
- Unit: `video_camera.name == "bright-gsc"`.

### Documentation Updates

- Docstrings on `VideoCamera`, `__init__()`, `read()`.
