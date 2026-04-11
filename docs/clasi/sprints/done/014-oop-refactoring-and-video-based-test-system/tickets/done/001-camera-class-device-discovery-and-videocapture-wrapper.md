---
id: '001'
title: 'Camera class: device discovery and VideoCapture wrapper'
status: done
use-cases:
- SUC-001
- SUC-002
depends-on: []
github-issue: ''
todo:
- docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Camera class: device discovery and VideoCapture wrapper

## Description

Create `src/aprilcam/camera/camera.py` containing the `Camera` class.
`Camera` wraps `cv.VideoCapture` with device metadata and provides class-level
discovery methods. It replaces scattered VideoCapture management that currently
lives inside `AprilCam.__init__` and `stream.py`.

`Camera` must not open hardware in `__init__` — the device should only be opened
when `read()` is first called (lazy open). This is critical so that `VideoCamera`
can subclass `Camera` without triggering hardware access on construction.

## Acceptance Criteria

- [ ] `camera/camera.py` exists with `Camera` class.
- [ ] `Camera.list() -> list[Camera]` enumerates available cameras via `camutil.list_cameras()`.
- [ ] `Camera.find(pattern: str) -> Camera` returns first camera matching the pattern
      (case-insensitive substring). Raises `CameraNotFoundError` if no match.
- [ ] `camera.name: str` — human-readable device name.
- [ ] `camera.index: int` — OpenCV device index.
- [ ] `camera.resolution: tuple[int, int] | None` — (width, height) if open, else None.
- [ ] `camera.is_open: bool` — True if VideoCapture is open.
- [ ] `camera.read() -> np.ndarray | None` — returns BGR frame or None on failure;
      opens capture lazily on first call.
- [ ] `camera.close()` — releases VideoCapture; idempotent.
- [ ] Context manager: `with Camera.find("Brio") as cam:` calls `close()` on exit.
- [ ] `camera/__init__.py` exports `Camera`.

## Implementation Plan

### Approach

Lift camera management from `AprilCam.__init__` (lines ~80-130) and delegate
discovery to existing `camutil` functions. `Camera` is a thin wrapper.

### Files to Create

- `src/aprilcam/camera/camera.py`

### Files to Modify

- `src/aprilcam/camera/__init__.py` — add `Camera` to exports

### Key Implementation Notes

- Store capture as `self._cap: cv.VideoCapture | None = None`.
- `Camera.__init__(index: int, name: str, *, backend: int | None = None)`.
- `_open()` calls `cv.VideoCapture(self.index)` (or with backend).
- `read()` calls `_open()` if `_cap is None`, then `_cap.read()`.
- `Camera.list()` wraps each `CameraInfo` from `camutil` into a `Camera`.
- `Camera.find()` calls `camutil.select_camera_by_pattern()`.

### Testing Plan

- Smoke: `from aprilcam.camera import Camera` imports cleanly.
- Smoke: `Camera.list()` returns a list (may be empty in CI).
- Unit: `Camera.find("__no_such_camera__")` raises `CameraNotFoundError`.
- Unit: `camera.is_open` is False before `read()` on a hardware-free stub.
- Unit: context manager calls `close()` on exit.

### Documentation Updates

- Docstrings on `Camera`, `list()`, `find()`, `read()`, `close()`.
