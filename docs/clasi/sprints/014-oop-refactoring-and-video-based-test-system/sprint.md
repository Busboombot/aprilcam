---
id: "014"
title: "OOP Refactoring and Video-Based Test System"
status: planning
branch: sprint/014-oop-refactoring-and-video-based-test-system
todos:
  - docs/clasi/todo/oop-refactoring.md
  - docs/clasi/todo/video-based-test-system.md
use-cases:
  - SUC-001
  - SUC-002
  - SUC-003
  - SUC-004
  - SUC-005
  - SUC-006
  - SUC-007
  - SUC-008
  - SUC-009
  - SUC-010
---

# Sprint 014: OOP Refactoring and Video-Based Test System

## Goals

1. Break the monolithic `AprilCam` class (1013 lines, 27 constructor params)
   into clean, well-bounded objects: `Camera`, `Tag`, `TagDetector`,
   `OpticalFlowTracker`, `VelocityEstimator`, `DetectionPipeline`,
   `TagTableTUI`, and a rewritten `Playfield`.
2. Establish a three-tier video-based test system (smoke, unit, system) using
   a `VideoCamera` subclass to drive the detection pipeline against recorded
   test videos without requiring camera hardware.

## Problem

`AprilCam` has grown to 1013 lines with 27 constructor parameters, mixing
detection logic, optical flow tracking, velocity estimation, ring-buffer
management, TUI rendering, and OpenCV window management in one class. This
makes individual components impossible to test in isolation, forces callers
to understand the entire class to use any one feature, and makes the code
resistant to extension.

The test directory was cleared when the package was reorganized into submodules.
There is no test coverage for the detection pipeline, tracking, or velocity
estimation.

## Solution

### Phase 1 — OOP Refactoring

Extract each distinct responsibility into its own module with a clean interface:

- `camera/camera.py` — `Camera` class: device discovery, VideoCapture wrapper,
  context manager.
- `core/tag.py` — `Tag` class: live tag handle with flat properties and
  `update()` / `position_at(t)` methods.
- `core/detector.py` — `TagDetector`: pure stateless detection engine.
- `core/tracker.py` — `OpticalFlowTracker`: LK tracking producing full
  4-corner motion decomposition.
- `core/motion.py` — `VelocityEstimator`: EMA velocity + deadband per tag.
- `core/pipeline.py` — `DetectionPipeline`: background thread orchestrating
  detect → track → velocity → ring buffer.
- `ui/tui.py` — `TagTableTUI`: Rich-based TUI dashboard extracted from
  `AprilCam`.
- `calibration/calibration.py` — `calibrate()`, `CameraCalibration`,
  `FieldSpec` split from `homography.py`.
- `core/playfield.py` rewritten — `Playfield` becomes the primary user-facing
  object; existing geometry logic becomes `PlayfieldBoundary`.
- `aprilcam.py` shrunk to compatibility shim or deleted; `run()` moved to
  `cli/live_cli.py`.
- `stream.py` deleted; its functions replaced by `Playfield` methods.
- `__init__.py` updated to export `Camera`, `Playfield`, `Tag`, `calibrate`.

### Phase 2 — Video-Based Test System

- `camera/video_camera.py` — `VideoCamera(Camera)`: reads frames from a
  `.mov` file instead of a live device, same interface.
- `tests/smoke/` — import checks, basic object construction.
- `tests/unit/` — one test file per class, dependencies mocked.
- `tests/system/` — full pipeline tests driven by videos in `tests/movies/`.

## Success Criteria

- All new classes exist with documented public APIs.
- `AprilCam` is reduced to a thin compatibility shim (or removed).
- `stream.py` is deleted.
- `tests/smoke/`, `tests/unit/`, `tests/system/` all pass with `uv run pytest`.
- System tests confirm tag detection works across both lighting conditions
  and both camera types in the test videos.
- `uv run aprilcam --help` continues to work.
- `import aprilcam; aprilcam.Camera`, `aprilcam.Playfield`, `aprilcam.Tag`
  all resolve.

## Scope

### In Scope

- All new classes listed in Phase 1.
- `VideoCamera` subclass.
- Three-tier test hierarchy with coverage of all new classes.
- `__init__.py` updated exports.
- `stream.py` deleted.
- `AprilCam` reduced/removed.
- `run()` interactive loop moved to CLI.

### Out of Scope

- MCP server rewiring to new API (deferred — compatibility shim covers it).
- Kalman filter replacement for `VelocityEstimator`.
- `solvePnP` 6DOF pose in `VelocityEstimator`.
- Streamable HTTP transport changes.
- New MCP tools.

## Test Strategy

The refactoring phase introduces no new behavior, so tests verify that the
extracted classes reproduce the behavior of the original code. The video-based
test system tests behavior against known recordings.

- Smoke tests run in under 2 seconds and require no hardware.
- Unit tests use `VideoCamera` or mocks; no hardware required.
- System tests feed full video files through `DetectionPipeline` and assert
  detection counts, position ranges, and velocity signs are plausible.

## Architecture Notes

- `Camera` is the foundation layer; `VideoCamera` is a subclass enabling
  hardware-free testing.
- `Playfield` owns the `DetectionPipeline`. Callers start/stop it via
  `Playfield.start()` / `Playfield.stop()`.
- `Tag` holds an atomically-replaced frozen snapshot (`TagRecord`); callers
  call `tag.update()` to pull the latest.
- `TagDetector` is stateless and directly testable with a single frame.
- `OpticalFlowTracker` takes grayscale frames; its output is a list of
  `Detection` objects with full 4-corner motion data.
- `VelocityEstimator` is per-tag state; it replaces the EMA velocity code
  now scattered across `Playfield.add_tag()`.
- Dependency direction: CLI / MCP Server → Playfield → Pipeline →
  Detector + Tracker + VelocityEstimator → Camera.

## GitHub Issues

None.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [x] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

| # | Title | Depends On | Group |
|---|-------|------------|-------|
| 001 | Camera class: device discovery and VideoCapture wrapper | — | 1 |
| 002 | TagDetector: pure stateless detection engine | — | 1 |
| 003 | VelocityEstimator: per-tag EMA velocity with deadband | — | 1 |
| 004 | TagTableTUI: extract Rich TUI dashboard from AprilCam | — | 1 |
| 005 | Calibration split: extract calibrate() and types from homography.py | — | 1 |
| 006 | OpticalFlowTracker: LK tracking with 4-corner motion decomposition | 002 | 2 |
| 007 | VideoCamera: Camera subclass reading frames from a video file | 001 | 2 |
| 008 | DetectionPipeline: background thread orchestrating detect-track-velocity-buffer | 002, 003, 006, 007 | 3 |
| 009 | Tag class: live tag handle with flat properties and update/position_at | 008 | 4 |
| 010 | Playfield rewrite: primary user-facing object owning DetectionPipeline | 005, 008, 009 | 4 |
| 011 | CLI live loop: move AprilCam.run() to cli/live_cli.py | 004, 010 | 5 |
| 012 | Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim | 010, 011 | 5 |
| 013 | Smoke tests: import and construction sanity checks | 012 | 6 |
| 014 | Unit tests: per-class tests for all new OOP classes | 013 | 6 |
| 015 | System tests: video-driven pipeline tests using tests/movies/*.mov | 014 | 7 |
