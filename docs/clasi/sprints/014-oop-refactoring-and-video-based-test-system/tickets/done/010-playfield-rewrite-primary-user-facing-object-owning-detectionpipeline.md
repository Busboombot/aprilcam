---
id: '010'
title: 'Playfield rewrite: primary user-facing object owning DetectionPipeline'
status: done
use-cases:
  - SUC-006
  - SUC-007
  - SUC-009
depends-on:
  - "005"
  - "008"
  - "009"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Playfield rewrite: primary user-facing object owning DetectionPipeline

## Description

Rewrite `src/aprilcam/core/playfield.py` so that `Playfield` is the primary
user-facing object. It owns a `DetectionPipeline`, exposes tag access via
`tags()` and `tag(id)`, supports both streaming and callback modes, handles
calibration, and delegates all geometry/corner work to an internal
`PlayfieldBoundary`.

The existing `Playfield` geometry logic (corner detection, polygon, ArUco
detection) is renamed `PlayfieldBoundary` and made internal. The public
`Playfield` API is clean and high-level.

## Acceptance Criteria

- [ ] `Playfield(camera, *, width_cm=None, height_cm=None, family="tag36h11",
      calibration=None)` constructor.
- [ ] `Playfield.start()` — starts the `DetectionPipeline`; idempotent.
- [ ] `Playfield.stop()` — stops the pipeline; idempotent.
- [ ] `playfield.tags() -> dict[int, Tag]` — returns all currently tracked tags.
- [ ] `playfield.tag(id: int) -> Tag | None` — returns a `Tag` for the given ID,
      or None if never seen.
- [ ] `playfield.stream() -> Generator[list[Tag], None, None]` — yields list of
      `Tag` objects per frame; terminates when `stop()` is called.
- [ ] `playfield.on_frame(callback: Callable[[list[Tag]], None])` — registers
      a push callback.
- [ ] `playfield.calibrate()` — runs `calibration.calibrate()` to detect corners
      and persist homography.
- [ ] `playfield.pixel_to_world(px, py) -> tuple[float, float] | None`.
- [ ] `playfield.world_to_pixel(wx, wy) -> tuple[float, float] | None`.
- [ ] `playfield.deskew(frame) -> np.ndarray` — returns deskewed frame.
- [ ] When `calibration` file provided and it exists, loads homography automatically.
- [ ] Existing `Playfield` geometry code renamed to `PlayfieldBoundary` (private).
- [ ] The old `Playfield` class is gone from the public namespace; new `Playfield`
      replaces it.
- [ ] `core/__init__.py` exports new `Playfield`.

## Implementation Plan

### Approach

1. Rename existing `Playfield` → `PlayfieldBoundary` in `playfield.py`.
2. Write the new `Playfield` class in the same file (or a new file, keeping
   import path identical: `aprilcam.core.playfield`).
3. `Playfield.__init__` creates `TagDetector`, `OpticalFlowTracker`,
   `DetectionPipeline`, and optionally loads calibration.
4. `tags()` queries the ring buffer; `tag(id)` creates/returns a cached `Tag`.
5. `stream()` is a generator that reads from the ring buffer on each pipeline callback.
6. `on_frame()` delegates to `DetectionPipeline.on_frame()`.
7. Geometry methods delegate to `PlayfieldBoundary`.

### Files to Modify

- `src/aprilcam/core/playfield.py` — full rewrite with `PlayfieldBoundary` rename
- `src/aprilcam/core/__init__.py` — ensure `Playfield` export points to new class

### Key Implementation Notes

- Backward compatibility: `aprilcam.Playfield` still resolves; it now points
  to the new class. The old constructor signature is different — this is a
  breaking change, documented in the sprint.
- `calibration` param: if a file path, load `CameraCalibration` from JSON;
  if `None`, calibration not loaded (homography not available until `calibrate()`
  is called).
- `Tag` instances are cached by ID: `_tag_cache: dict[int, Tag]`.
- `stream()`: use `queue.Queue` filled by `on_frame` callback; `stop()` puts
  a sentinel; generator yields until sentinel received.

### Testing Plan

- Unit: `Playfield(VideoCamera(path), width_cm=100, height_cm=80)` constructs.
- Unit: `start()` starts pipeline; `stop()` stops it.
- Unit: `tags()` returns dict after pipeline processes frames.
- Unit: `tag(id)` returns `Tag` if seen, `None` if not.
- Unit: `stream()` generator terminates after `stop()`.
- Integration (system test in ticket 015): full pipeline with video.

### Documentation Updates

- Docstrings on `Playfield`, all public methods.
- Note the breaking change vs. old `Playfield` geometry class.
