---
status: in-progress
sprint: '014'
tickets:
- 014-015
---

# OOP Refactoring: Break Monolithic AprilCam into Clean Object Hierarchy

Break the monolithic AprilCam class (1013 lines, 27 constructor params) into
clean, well-bounded objects with proper separation of concerns. The goal is
an API where common operations take 10-15 lines of code and all UI is fully
separated from core logic.

## New Classes

1. **Camera** (`camera.py`) — Wraps VideoCapture with device metadata and
   discovery (`Camera.list()`, `Camera.find("Brio")`). Context manager support.

2. **Playfield** (`playfield.py`, rewritten) — Primary user-facing object.
   Owns the detection pipeline. Tag access via pull (`tag.update()`) and push
   (`field.stream()`, `field.on_frame(callback)`). Handles calibration,
   geometry, object detection. Current Playfield becomes internal
   `PlayfieldBoundary`. Constructor accepts a calibration file path (or
   "auto" to discover). The calibration file persistently stores playfield
   dimensions (width/height), per-camera homography, and other calibration
   data. When `calibrate()` is called, results are written back to this
   same file. This means `width_cm`/`height_cm` don't need to be constructor
   params if a calibration file exists — they're loaded from it.

3. **Tag** (`tag.py`) — Live tag object obtained from Playfield. Properties
   use flat names: `cx`, `cy` (pixel center), `wx`, `wy` (world center),
   `velocity` (2D + z/scale), `speed`, `heading`, `orientation`, `rotation_rate`,
   high-res capture timestamp. Methods: `update()` (pull latest),
   `position_at(t)` (extrapolation in all available dimensions, designed for
   future Kalman filter).

4. **TagDetector** (`detector.py`) — Pure stateless detection engine extracted
   from AprilCam. Absorbs `_build_detectors()`, `detect_apriltags()`,
   `_maybe_preprocess()`, and the duplicate module-level functions.

5. **OpticalFlowTracker** (`tracker.py`) — LK tracking extracted from
   AprilCam. Absorbs `lk_track()` and detect-or-track branching from
   `process_frame()`. Produces full per-tag motion decomposition from
   4-corner tracking: 2D translation, rotation rate (yaw delta), and
   scale change (z-axis proxy). Don't discard information — all 4 corners
   carry richer motion data than just center position.

6. **VelocityEstimator** (`motion.py`) — Per-tag motion estimation with EMA
   smoothing and deadband suppression. Consumes full corner-pair data from
   OpticalFlowTracker to estimate 2D velocity, rotation rate, and scale
   rate. Extracted from `Playfield.add_tag()`. With camera intrinsics
   available, can use `solvePnP` for full 6DOF pose and 3D velocity.
   Designed for future Kalman filter replacement.

7. **DetectionPipeline** (`pipeline.py`) — Background thread orchestrating
   detection, tracking, velocity, and ring buffer. Evolves from DetectionLoop,
   composes TagDetector + OpticalFlowTracker.

8. **TagTableTUI** (`tui.py`) — Rich-based TUI dashboard extracted from
   AprilCam (`_build_tui_layout`, `_print_tui`, `_stop_tui`, `_ema_smooth`).

9. **Calibration** (`calibration.py`) — `calibrate()`, `CameraCalibration`,
   `FieldSpec` split from `homography.py`. Pure math stays in `homography.py`.

10. **AprilCam.run()** interactive loop moved to CLI layer (`cli/live_cli.py`).

## Migration Strategy

- **Sprint 1**: Extract classes (non-breaking). All existing code keeps
  working — AprilCam becomes a thin wrapper.
- **Sprint 2**: Rewire consumers (stream.py, mcp_server.py, CLI). Deprecate
  AprilCam class.

## Example Target API

```python
import aprilcam

camera = aprilcam.Camera.find("Brio")
field = aprilcam.Playfield(camera, width_cm=101, height_cm=89)
field.start()

tag = field.tag(42)
if tag:
    tag.update()
    print(f"Tag 42 at ({tag.wx:.1f}, {tag.wy:.1f}) cm, pixel ({tag.cx:.0f}, {tag.cy:.0f})")
    print(f"  speed: {tag.speed:.1f} cm/s, heading: {tag.heading:.1f} rad")

field.stop()
```

```python
# Streaming mode
with aprilcam.Camera.find("Brio") as cam:
    field = aprilcam.Playfield(cam, width_cm=101, height_cm=89)
    for tags in field.stream():
        for tag in tags:
            print(f"Tag {tag.id}: ({tag.wx:.1f}, {tag.wy:.1f})")
```

## Key Design Decisions

- Playfield owns the detection pipeline; `start()` spawns background thread
- Tag.update() pulls latest from ring buffer; Tag.position_at(t) extrapolates
- Timestamp is high-res capture time (when frame was taken, not processed)
- OpticalFlowTracker produces full motion decomposition from 4-corner tracking
- VelocityEstimator consumes rich corner data; designed for Kalman filter swap
- MCP server delegates to Camera/Playfield objects instead of raw construction
- All UI (TUI, OpenCV window, overlays) accesses system through public API only
