---
agent_type: Explore
agent_id: a01634c1d6a193be6
sprint_id: "014"
tickets: ""
started_at: 2026-04-11T22:04:33.692152+00:00
stopped_at: 2026-04-11T22:05:17.711093+00:00
duration_seconds: 44.0
---

# Subagent: Explore
## Result

## Ticket Summary for Sprint 014: OOP Refactoring and Video-Based Test System

### **Ticket 001: Camera class: device discovery and VideoCapture wrapper**
- **Key Acceptance Criteria:**
  - `Camera.list()` enumerates available cameras
  - `Camera.find(pattern)` finds camera by substring match
  - Lazy open on first `read()` call (not in `__init__`)
  - Properties: `name`, `index`, `resolution`, `is_open`
  - Context manager support
- **Files to Create:** `src/aprilcam/camera/camera.py`
- **Files to Modify:** `src/aprilcam/camera/__init__.py`
- **Key Details:** Wraps `cv.VideoCapture`; lifts camera management from `AprilCam.__init__`; delegates discovery to `camutil`

---

### **Ticket 002: TagDetector: pure stateless detection engine**
- **Key Acceptance Criteria:**
  - `DetectorConfig` dataclass with tuning parameters
  - `TagDetector.detect(frame_bgr)` returns list of `Detection` objects
  - Stateless: calling twice on same frame returns equivalent results
  - Removes duplicate functions from `aprilcam.py`
- **Files to Create:** `src/aprilcam/core/detector.py`
- **Files to Modify:** `src/aprilcam/core/aprilcam.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** Consolidates `_build_detectors()`, `detect_apriltags()`, preprocessing logic; `Detection` dataclass with `id`, `center`, `corners`, `family`

---

### **Ticket 003: VelocityEstimator: per-tag EMA velocity with deadband**
- **Key Acceptance Criteria:**
  - `update(position, timestamp)` returns `(velocity_vec, speed)`
  - EMA smoothing: `v_ema = alpha * v_raw + (1 - alpha) * v_prev`
  - Deadband: suppress velocity if raw speed below threshold
  - `predict_position(t)` extrapolates linearly
  - `reset()` clears state
- **Files to Create:** `src/aprilcam/core/motion.py`
- **Files to Modify:** `src/aprilcam/core/playfield.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** One instance per tracked tag; pixel-space only (world transform stays in `Playfield`)

---

### **Ticket 004: TagTableTUI: extract Rich TUI dashboard from AprilCam**
- **Key Acceptance Criteria:**
  - `TagTableTUI` has no dependency on `AprilCam`
  - `update(tags: list[dict])` renders live table with columns: `id`, `cx`, `cy`, `wx`, `wy`, `speed`, `heading`, `orientation`
  - Context manager support
  - EMA smoothing for display only (not exported to business logic)
- **Files to Create:** `src/aprilcam/ui/tui.py`
- **Files to Modify:** `src/aprilcam/core/aprilcam.py`, `src/aprilcam/ui/__init__.py`
- **Key Details:** Extracts `_build_tui_layout`, `_print_tui`, `_stop_tui`, `_ema_smooth` from `AprilCam`

---

### **Ticket 005: Calibration split: extract calibrate() and types from homography.py**
- **Key Acceptance Criteria:**
  - `calibration/calibration.py` contains `calibrate()`, `CameraCalibration`, `FieldSpec`
  - `homography.py` retains only pure math functions
  - `FieldSpec` has `width_cm`, `height_cm` and unit conversion helpers
  - All callers updated to import from `calibration.calibration`
- **Files to Create:** `src/aprilcam/calibration/calibration.py`
- **Files to Modify:** `src/aprilcam/calibration/homography.py`, `src/aprilcam/calibration/__init__.py`
- **Key Details:** Split: `calibration.py` owns workflow; `homography.py` owns matrix math

---

### **Ticket 006: OpticalFlowTracker: LK tracking with 4-corner motion decomposition**
- **Key Acceptance Criteria:**
  - `update(gray, detections)` returns updated `Detection` list
  - With `detections`: reset tracked points
  - Without `detections`: propagate via Lucas-Kanade flow
  - Output includes motion fields: translation, rotation rate, scale delta
  - Fallback to center-only if LK fails
- **Files to Create:** `src/aprilcam/core/tracker.py`
- **Files to Modify:** `src/aprilcam/core/detector.py`, `src/aprilcam/core/aprilcam.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** Absorbs `AprilCam.lk_track()` and detect-or-track branching; LK params: `winSize=(21,21)`, `maxLevel=3`

---

### **Ticket 007: VideoCamera: Camera subclass reading frames from a video file**
- **Key Acceptance Criteria:**
  - `VideoCamera(path)` subclasses `Camera`; reads frames from file
  - `read()` returns frames sequentially; `None` at EOF
  - `is_open` tracks EOF state
  - `name` returns filename stem; `index` returns `-1` (sentinel)
  - Context manager support
- **Files to Create:** `src/aprilcam/camera/video_camera.py`
- **Files to Modify:** `src/aprilcam/camera/__init__.py`
- **Key Details:** Opens video file directly; bypasses `Camera._open()`; enables video-based testing without hardware

---

### **Ticket 008: DetectionPipeline: background thread orchestrating detect-track-velocity-buffer**
- **Key Acceptance Criteria:**
  - `start()` starts background thread; idempotent
  - `stop()` stops thread; idempotent
  - `is_running` property
  - `on_frame(callback)` registers callback invoked after each frame
  - Loop: read → grayscale → detect (every N frames) → track → velocity → buffer write
  - Per-tag `VelocityEstimator` instances recycled across frames
  - Thread-safe ring buffer
- **Files to Create:** `src/aprilcam/core/pipeline.py`
- **Files to Modify:** `src/aprilcam/core/__init__.py`, optionally `src/aprilcam/core/aprilcam.py`
- **Key Details:** Evolves from existing `DetectionLoop`; daemon thread; stop signal via `threading.Event`

---

### **Ticket 009: Tag class: live tag handle with flat properties and update/position_at**
- **Key Acceptance Criteria:**
  - `Tag(tag_id, pipeline)` constructor
  - Flat properties: `id`, `cx`, `cy`, `wx`, `wy`, `orientation`, `velocity`, `speed`, `heading`, `rotation_rate`, `timestamp`, `age`, `is_visible`
  - `update()` atomically replaces snapshot from ring buffer; returns self
  - `position_at(t)` extrapolates pixel position
  - `to_dict()` returns flat dict
- **Files to Create:** `src/aprilcam/core/tag.py`
- **Files to Modify:** `src/aprilcam/core/__init__.py`
- **Key Details:** Read-only view of ring buffer; thread-safe snapshot replacement; returns sensible defaults when tag not seen

---

### **Ticket 010: Playfield rewrite: primary user-facing object owning DetectionPipeline**
- **Key Acceptance Criteria:**
  - `Playfield(camera, width_cm, height_cm, family, calibration)` constructor
  - `start()`, `stop()` control pipeline
  - `tags()` returns dict[int, Tag]; `tag(id)` returns Tag | None
  - `stream()` generator yielding list[Tag] per frame
  - `on_frame(callback)` registers push callback
  - `calibrate()`, `pixel_to_world()`, `world_to_pixel()`, `deskew()`
  - Auto-loads calibration from JSON if file provided
  - Old `Playfield` geometry logic renamed to `PlayfieldBoundary` (private)
- **Files to Modify:** `src/aprilcam/core/playfield.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** New primary user-facing object; owns `DetectionPipeline`, `TagDetector`, `OpticalFlowTracker`; breaking API change from old geometry class

---

### **Ticket 011: CLI live loop: move AprilCam.run() to cli/live_cli.py**
- **Key Acceptance Criteria:**
  - `cli/live_cli.py` uses `Playfield` + `TagTableTUI` directly
  - No dependency on `AprilCam` in CLI layer
  - `aprilcam live` subcommand works end-to-end
  - TUI renders during operation
  - Keyboard interrupt (`Ctrl+C`) stops cleanly
- **Files to Modify:** `src/aprilcam/cli/live_cli.py`, `src/aprilcam/core/aprilcam.py`, `src/aprilcam/cli/__init__.py`
- **Key Details:** Rewrites CLI using new high-level API; removes dependency on `AprilCam`; argument parsing stays in CLI layer

---

### **Ticket 012: Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim**
- **Key Acceptance Criteria:**
  - `aprilcam.Camera`, `aprilcam.Tag`, `aprilcam.Playfield`, `aprilcam.calibrate` all resolve
  - Legacy names (`detect_tags`, `detect_objects`, `AprilCam`) still available (compatibility re-exports)
  - `stream.py` deleted
  - `AprilCam` reduced to minimal shim
  - `uv run pytest` passes
- **Files to Delete:** `src/aprilcam/stream.py`
- **Files to Modify:** `src/aprilcam/__init__.py`, `src/aprilcam/core/aprilcam.py`
- **Key Details:** Compatibility wrappers for old API; `detect_tags` wraps `Playfield.stream()`; `AprilCam` delegates to new classes

---

### **Ticket 013: Smoke tests: import and construction sanity checks**
- **Key Acceptance Criteria:**
  - `tests/smoke/test_imports.py` verifies all new classes import cleanly
  - `tests/smoke/test_construction.py` verifies basic construction (no hardware)
  - All smoke tests complete in under 5 seconds
  - Verify imports: `Camera`, `Tag`, `Playfield`, `calibrate`, `VideoCamera`, legacy (`detect_tags`, `AprilCam`)
  - Verify construction: `TagDetector()`, `DetectorConfig()`, `VelocityEstimator()`, `OpticalFlowTracker()`, `TagTableTUI()`, `VideoCamera(path)`
- **Files to Create:** `tests/smoke/__init__.py`, `tests/smoke/test_imports.py`, `tests/smoke/test_construction.py`
- **Files to Modify:** `tests/conftest.py` (if needed)
- **Key Details:** Fast hardware-free sanity checks; `pytest.skip` for Rich if not installed; first defense against import errors

---

### **Ticket 014: Unit tests: per-class tests for all new OOP classes**
- **Key Acceptance Criteria:**
  - Per-class test files for: `Camera`, `VideoCamera`, `TagDetector`, `OpticalFlowTracker`, `VelocityEstimator`, `DetectionPipeline`, `Tag`, `TagRecord`, `RingBuffer`
  - Test key methods: list, find, read, detect, update, reset, on_frame
  - No hardware required
  - All tests under 30 seconds total
- **Files to Create:** `tests/unit/__init__.py`, `tests/unit/test_*.py` (9 test files)
- **Key Details:** Use `VideoCamera` for pipeline tests; synthetic frames for detector/tracker; mock pipeline for `Tag` tests

---

### **Ticket 015: System tests: video-driven pipeline tests using tests/movies/*.mov**
- **Key Acceptance Criteria:**
  - `tests/system/test_detection_pipeline_videos.py` with parametrized tests
  - Test all four videos: `bright-gsc.mov`, `bright-ov9782.mov`, `dim-gsc.mov`, `dim-ov9782.mov`
  - Pipeline runs to EOF without error
  - At least one tag detected per video (or xfail for dim videos)
  - Detected positions within frame dimensions
  - Extrapolation (`estimate(t)`) sanity check
- **Files to Create:** `tests/system/__init__.py`, `tests/system/test_detection_pipeline_videos.py`
- **Key Details:** Full end-to-end pipeline tests; no hardware required; under 60 seconds total for all videos

---

## Summary

The 15 tickets are sequenced to:
1. **Tickets 001–007:** Core class extraction (Camera, TagDetector, VelocityEstimator, TUI, Calibration, OpticalFlowTracker, VideoCamera)
2. **Tickets 008–010:** Pipeline orchestration (DetectionPipeline, Tag, Playfield rewrite)
3. **Tickets 011–012:** CLI migration and API stabilization
4. **Tickets 013–015:** Testing (smoke, unit, system)

Each ticket has clear dependencies, enabling parallelizable work. The refactor transforms a monolithic `AprilCam` into composable, testable classes with a clean public API.

## Transcript

---

### user — 2026-04-11T22:04:33.603Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

Read ALL ticket files in /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/ and provide a concise summary of each ticket's acceptance criteria and implementation plan. There are 15 tickets (001 through 015). Read each one completely.

For each ticket report:
- Ticket number and title
- Key acceptance criteria (bulleted)
- Files to create/modify
- Key implementation details

---

### assistant — 2026-04-11T22:04:35.845Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001 | stop_reason: tool_use

I'll read all 15 ticket files from the specified directory. Let me start by listing the tickets and then reading each one.

---

### assistant — 2026-04-11T22:04:35.929Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001 | stop_reason: tool_use

> **Tool Use**: `Bash`
> ```json
> {
>   "command": "ls -1 /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/"
> }
> ```

---

### user — 2026-04-11T22:04:36.001Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01PrJKFUEENyHNaAXPoUtZs4`)
> ```
> 001-camera-class-device-discovery-and-videocapture-wrapper.md
> 002-tagdetector-pure-stateless-detection-engine.md
> 003-velocityestimator-per-tag-ema-velocity-with-deadband.md
> 004-tagtabletui-extract-rich-tui-dashboard-from-aprilcam.md
> 005-calibration-split-extract-calibrate-and-types-from-homography-py.md
> 006-opticalflowtracker-lk-tracking-with-4-corner-motion-decomposition.md
> 007-videocamera-camera-subclass-reading-frames-from-a-video-file.md
> 008-detectionpipeline-background-thread-orchestrating-...
> ```

---

### assistant — 2026-04-11T22:04:37.263Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

Now let me read all 15 ticket files:

---

### assistant — 2026-04-11T22:04:38.224Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/001-camera-class-device-discovery-and-videocapture-wrapper.md"
> }
> ```

---

### user — 2026-04-11T22:04:38.260Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_012kpUzh7DQzuVEAdmStENoP`)
> ```
> 1	---
> 2	id: '001'
> 3	title: 'Camera class: device discovery and VideoCapture wrapper'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-001
> 7	  - SUC-002
> 8	depends-on: []
> 9	github-issue: ''
> 10	todo:
> 11	  - docs/clasi/todo/oop-refactoring.md
> 12	---
> 13	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 14	
> 15	# Camera class: device discovery and VideoCapture wrapper
> 16	
> 17	## Description
> 18	
> 19	Create `src/aprilcam/camera/camera.py` containing the `Camera` class.
> 20	`Camer...
> ```

---

### assistant — 2026-04-11T22:04:38.819Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/002-tagdetector-pure-stateless-detection-engine.md"
> }
> ```

---

### user — 2026-04-11T22:04:38.852Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_013HxqrvZpQp5ovW62TkPW8K`)
> ```
> 1	---
> 2	id: '002'
> 3	title: 'TagDetector: pure stateless detection engine'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-003
> 7	depends-on: []
> 8	github-issue: ''
> 9	todo:
> 10	  - docs/clasi/todo/oop-refactoring.md
> 11	---
> 12	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 13	
> 14	# TagDetector: pure stateless detection engine
> 15	
> 16	## Description
> 17	
> 18	Create `src/aprilcam/core/detector.py` with `DetectorConfig` and `TagDetector`.
> 19	
> 20	`TagDetector` is a pure, stat...
> ```

---

### assistant — 2026-04-11T22:04:39.435Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/003-velocityestimator-per-tag-ema-velocity-with-deadband.md"
> }
> ```

---

### user — 2026-04-11T22:04:39.469Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01GH8Pn8RvkoJv3BttrxzSU8`)
> ```
> 1	---
> 2	id: '003'
> 3	title: 'VelocityEstimator: per-tag EMA velocity with deadband'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-005
> 7	depends-on: []
> 8	github-issue: ''
> 9	todo:
> 10	  - docs/clasi/todo/oop-refactoring.md
> 11	---
> 12	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 13	
> 14	# VelocityEstimator: per-tag EMA velocity with deadband
> 15	
> 16	## Description
> 17	
> 18	Create `src/aprilcam/core/motion.py` with `VelocityEstimator`.
> 19	
> 20	`VelocityEstimator` is per-t...
> ```

---

### assistant — 2026-04-11T22:04:40.069Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/004-tagtabletui-extract-rich-tui-dashboard-from-aprilcam.md"
> }
> ```

---

### user — 2026-04-11T22:04:40.102Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01EbMNcV7CYdWUb1J76Y9Yrw`)
> ```
> 1	---
> 2	id: '004'
> 3	title: 'TagTableTUI: extract Rich TUI dashboard from AprilCam'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-008
> 7	depends-on: []
> 8	github-issue: ''
> 9	todo:
> 10	  - docs/clasi/todo/oop-refactoring.md
> 11	---
> 12	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 13	
> 14	# TagTableTUI: extract Rich TUI dashboard from AprilCam
> 15	
> 16	## Description
> 17	
> 18	Create `src/aprilcam/ui/tui.py` with `TagTableTUI`.
> 19	
> 20	`TagTableTUI` is a Rich-based terminal ...
> ```

---

### assistant — 2026-04-11T22:04:40.550Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/005-calibration-split-extract-calibrate-and-types-from-homography-py.md"
> }
> ```

---

### user — 2026-04-11T22:04:40.580Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01BdnACBAWihjdpoMV5NQBeH`)
> ```
> 1	---
> 2	id: '005'
> 3	title: 'Calibration split: extract calibrate() and types from homography.py'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-009
> 7	depends-on: []
> 8	github-issue: ''
> 9	todo:
> 10	  - docs/clasi/todo/oop-refactoring.md
> 11	---
> 12	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 13	
> 14	# Calibration split: extract calibrate() and types from homography.py
> 15	
> 16	## Description
> 17	
> 18	Create `src/aprilcam/calibration/calibration.py` by splitting the curr...
> ```

---

### assistant — 2026-04-11T22:04:41.119Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/006-opticalflowtracker-lk-tracking-with-4-corner-motion-decomposition.md"
> }
> ```

---

### user — 2026-04-11T22:04:41.149Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_0195EmPpNeoD82XC8rU72jtT`)
> ```
> 1	---
> 2	id: '006'
> 3	title: 'OpticalFlowTracker: LK tracking with 4-corner motion decomposition'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-004
> 7	depends-on:
> 8	  - "002"
> 9	github-issue: ''
> 10	todo:
> 11	  - docs/clasi/todo/oop-refactoring.md
> 12	---
> 13	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 14	
> 15	# OpticalFlowTracker: LK tracking with 4-corner motion decomposition
> 16	
> 17	## Description
> 18	
> 19	Create `src/aprilcam/core/tracker.py` with `OpticalFlowTracker...
> ```

---

### assistant — 2026-04-11T22:04:41.751Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/007-videocamera-camera-subclass-reading-frames-from-a-video-file.md"
> }
> ```

---

### user — 2026-04-11T22:04:41.840Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01Bf4ufaxwaUC3wngssEHmwW`)
> ```
> 1	---
> 2	id: '007'
> 3	title: 'VideoCamera: Camera subclass reading frames from a video file'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-002
> 7	  - SUC-010
> 8	depends-on:
> 9	  - "001"
> 10	github-issue: ''
> 11	todo:
> 12	  - docs/clasi/todo/video-based-test-system.md
> 13	---
> 14	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 15	
> 16	# VideoCamera: Camera subclass reading frames from a video file
> 17	
> 18	## Description
> 19	
> 20	Create `src/aprilcam/camera/video_camera.py` with...
> ```

---

### assistant — 2026-04-11T22:04:42.205Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/008-detectionpipeline-background-thread-orchestrating-detect-track-velocity-buffer.md"
> }
> ```

---

### user — 2026-04-11T22:04:42.302Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_015c7DjazesL5xEwZ8BnyrTr`)
> ```
> 1	---
> 2	id: "008"
> 3	title: 'DetectionPipeline: background thread orchestrating detect-track-velocity-buffer'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-002
> 7	  - SUC-004
> 8	  - SUC-005
> 9	  - SUC-006
> 10	  - SUC-007
> 11	depends-on:
> 12	  - "002"
> 13	  - "003"
> 14	  - "006"
> 15	  - "007"
> 16	github-issue: ''
> 17	todo:
> 18	  - docs/clasi/todo/oop-refactoring.md
> 19	---
> 20	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 21	
> 22	# DetectionPipeline: background thread orchestra...
> ```

---

### assistant — 2026-04-11T22:04:42.805Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/009-tag-class-live-tag-handle-with-flat-properties-and-update-position-at.md"
> }
> ```

---

### user — 2026-04-11T22:04:42.835Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01UbQmwHoqmrgsw6ep1N3XfV`)
> ```
> 1	---
> 2	id: "009"
> 3	title: 'Tag class: live tag handle with flat properties and update/position_at'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-006
> 7	  - SUC-007
> 8	depends-on:
> 9	  - "008"
> 10	github-issue: ''
> 11	todo:
> 12	  - docs/clasi/todo/oop-refactoring.md
> 13	---
> 14	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 15	
> 16	# Tag class: live tag handle with flat properties and update/position_at
> 17	
> 18	## Description
> 19	
> 20	Create `src/aprilcam/core/tag.py` with ...
> ```

---

### assistant — 2026-04-11T22:04:43.465Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/010-playfield-rewrite-primary-user-facing-object-owning-detectionpipeline.md"
> }
> ```

---

### user — 2026-04-11T22:04:43.495Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01GShiT2AFaM9nxt8zWtrZWe`)
> ```
> 1	---
> 2	id: '010'
> 3	title: 'Playfield rewrite: primary user-facing object owning DetectionPipeline'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-006
> 7	  - SUC-007
> 8	  - SUC-009
> 9	depends-on:
> 10	  - "005"
> 11	  - "008"
> 12	  - "009"
> 13	github-issue: ''
> 14	todo:
> 15	  - docs/clasi/todo/oop-refactoring.md
> 16	---
> 17	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 18	
> 19	# Playfield rewrite: primary user-facing object owning DetectionPipeline
> 20	
> 21	## Description
> 22	
> 2...
> ```

---

### assistant — 2026-04-11T22:04:43.886Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/011-cli-live-loop-move-aprilcam-run-to-cli-live-cli-py.md"
> }
> ```

---

### user — 2026-04-11T22:04:43.915Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01Rv3jJQ8VGZVVcQ3HYJzvZL`)
> ```
> 1	---
> 2	id: '011'
> 3	title: 'CLI live loop: move AprilCam.run() to cli/live_cli.py'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-008
> 7	depends-on:
> 8	  - "004"
> 9	  - "010"
> 10	github-issue: ''
> 11	todo:
> 12	  - docs/clasi/todo/oop-refactoring.md
> 13	---
> 14	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 15	
> 16	# CLI live loop: move AprilCam.run() to cli/live_cli.py
> 17	
> 18	## Description
> 19	
> 20	Move the interactive detection loop from `AprilCam.run()` into `cli/live_cl...
> ```

---

### assistant — 2026-04-11T22:04:44.700Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/012-update-init-py-export-camera-tag-calibrate-delete-stream-py-shrink-aprilcam-shim.md"
> }
> ```

---

### user — 2026-04-11T22:04:44.737Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01SQwgdDgL4epXpyxT4VW54h`)
> ```
> 1	---
> 2	id: '012'
> 3	title: 'Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-001
> 7	  - SUC-006
> 8	  - SUC-009
> 9	depends-on:
> 10	  - "010"
> 11	  - "011"
> 12	github-issue: ''
> 13	todo:
> 14	  - docs/clasi/todo/oop-refactoring.md
> 15	---
> 16	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 17	
> 18	# Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim
> 19...
> ```

---

### assistant — 2026-04-11T22:04:45.223Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/013-smoke-tests-import-and-construction-sanity-checks.md"
> }
> ```

---

### user — 2026-04-11T22:04:45.251Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01F3esu6exSo7yb3JPiDbG8L`)
> ```
> 1	---
> 2	id: '013'
> 3	title: 'Smoke tests: import and construction sanity checks'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-010
> 7	depends-on:
> 8	  - "012"
> 9	github-issue: ''
> 10	todo:
> 11	  - docs/clasi/todo/video-based-test-system.md
> 12	---
> 13	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 14	
> 15	# Smoke tests: import and construction sanity checks
> 16	
> 17	## Description
> 18	
> 19	Create `tests/smoke/` with fast, hardware-free sanity checks. Smoke tests
> 20	verify th...
> ```

---

### assistant — 2026-04-11T22:04:45.780Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/014-unit-tests-per-class-tests-for-all-new-oop-classes.md"
> }
> ```

---

### user — 2026-04-11T22:04:45.813Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01MhgsjRycxbgyNhB5NQ2dpY`)
> ```
> 1	---
> 2	id: '014'
> 3	title: 'Unit tests: per-class tests for all new OOP classes'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-010
> 7	depends-on:
> 8	  - "013"
> 9	github-issue: ''
> 10	todo:
> 11	  - docs/clasi/todo/video-based-test-system.md
> 12	---
> 13	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 14	
> 15	# Unit tests: per-class tests for all new OOP classes
> 16	
> 17	## Description
> 18	
> 19	Create `tests/unit/` with one test file per major class. Unit tests verify
> 20	constr...
> ```

---

### assistant — 2026-04-11T22:04:46.221Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001 | stop_reason: tool_use

> **Tool Use**: `Read`
> ```json
> {
>   "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/015-system-tests-video-driven-pipeline-tests-using-tests-movies-mov.md"
> }
> ```

---

### user — 2026-04-11T22:04:46.252Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags`

> **Tool Result** (id: `toolu_01C2kCo2iW9wzBCHeX56Eqdt`)
> ```
> 1	---
> 2	id: '015'
> 3	title: 'System tests: video-driven pipeline tests using tests/movies/*.mov'
> 4	status: todo
> 5	use-cases:
> 6	  - SUC-002
> 7	  - SUC-010
> 8	depends-on:
> 9	  - "014"
> 10	github-issue: ''
> 11	todo:
> 12	  - docs/clasi/todo/oop-refactoring.md
> 13	  - docs/clasi/todo/video-based-test-system.md
> 14	---
> 15	<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->
> 16	
> 17	# System tests: video-driven pipeline tests using tests/movies/*.mov
> 18	
> 19	## Description
> 20	
> ...
> ```

---

### assistant — 2026-04-11T22:05:17.556Z

branch: `sprint/014-oop-refactoring-and-video-based-test-system` | userType: external | cwd: `/Volumes/Proj/proj/RobotProjects/AprilTags` | model: claude-haiku-4-5-20251001 | stop_reason: end_turn

## Ticket Summary for Sprint 014: OOP Refactoring and Video-Based Test System

### **Ticket 001: Camera class: device discovery and VideoCapture wrapper**
- **Key Acceptance Criteria:**
  - `Camera.list()` enumerates available cameras
  - `Camera.find(pattern)` finds camera by substring match
  - Lazy open on first `read()` call (not in `__init__`)
  - Properties: `name`, `index`, `resolution`, `is_open`
  - Context manager support
- **Files to Create:** `src/aprilcam/camera/camera.py`
- **Files to Modify:** `src/aprilcam/camera/__init__.py`
- **Key Details:** Wraps `cv.VideoCapture`; lifts camera management from `AprilCam.__init__`; delegates discovery to `camutil`

---

### **Ticket 002: TagDetector: pure stateless detection engine**
- **Key Acceptance Criteria:**
  - `DetectorConfig` dataclass with tuning parameters
  - `TagDetector.detect(frame_bgr)` returns list of `Detection` objects
  - Stateless: calling twice on same frame returns equivalent results
  - Removes duplicate functions from `aprilcam.py`
- **Files to Create:** `src/aprilcam/core/detector.py`
- **Files to Modify:** `src/aprilcam/core/aprilcam.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** Consolidates `_build_detectors()`, `detect_apriltags()`, preprocessing logic; `Detection` dataclass with `id`, `center`, `corners`, `family`

---

### **Ticket 003: VelocityEstimator: per-tag EMA velocity with deadband**
- **Key Acceptance Criteria:**
  - `update(position, timestamp)` returns `(velocity_vec, speed)`
  - EMA smoothing: `v_ema = alpha * v_raw + (1 - alpha) * v_prev`
  - Deadband: suppress velocity if raw speed below threshold
  - `predict_position(t)` extrapolates linearly
  - `reset()` clears state
- **Files to Create:** `src/aprilcam/core/motion.py`
- **Files to Modify:** `src/aprilcam/core/playfield.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** One instance per tracked tag; pixel-space only (world transform stays in `Playfield`)

---

### **Ticket 004: TagTableTUI: extract Rich TUI dashboard from AprilCam**
- **Key Acceptance Criteria:**
  - `TagTableTUI` has no dependency on `AprilCam`
  - `update(tags: list[dict])` renders live table with columns: `id`, `cx`, `cy`, `wx`, `wy`, `speed`, `heading`, `orientation`
  - Context manager support
  - EMA smoothing for display only (not exported to business logic)
- **Files to Create:** `src/aprilcam/ui/tui.py`
- **Files to Modify:** `src/aprilcam/core/aprilcam.py`, `src/aprilcam/ui/__init__.py`
- **Key Details:** Extracts `_build_tui_layout`, `_print_tui`, `_stop_tui`, `_ema_smooth` from `AprilCam`

---

### **Ticket 005: Calibration split: extract calibrate() and types from homography.py**
- **Key Acceptance Criteria:**
  - `calibration/calibration.py` contains `calibrate()`, `CameraCalibration`, `FieldSpec`
  - `homography.py` retains only pure math functions
  - `FieldSpec` has `width_cm`, `height_cm` and unit conversion helpers
  - All callers updated to import from `calibration.calibration`
- **Files to Create:** `src/aprilcam/calibration/calibration.py`
- **Files to Modify:** `src/aprilcam/calibration/homography.py`, `src/aprilcam/calibration/__init__.py`
- **Key Details:** Split: `calibration.py` owns workflow; `homography.py` owns matrix math

---

### **Ticket 006: OpticalFlowTracker: LK tracking with 4-corner motion decomposition**
- **Key Acceptance Criteria:**
  - `update(gray, detections)` returns updated `Detection` list
  - With `detections`: reset tracked points
  - Without `detections`: propagate via Lucas-Kanade flow
  - Output includes motion fields: translation, rotation rate, scale delta
  - Fallback to center-only if LK fails
- **Files to Create:** `src/aprilcam/core/tracker.py`
- **Files to Modify:** `src/aprilcam/core/detector.py`, `src/aprilcam/core/aprilcam.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** Absorbs `AprilCam.lk_track()` and detect-or-track branching; LK params: `winSize=(21,21)`, `maxLevel=3`

---

### **Ticket 007: VideoCamera: Camera subclass reading frames from a video file**
- **Key Acceptance Criteria:**
  - `VideoCamera(path)` subclasses `Camera`; reads frames from file
  - `read()` returns frames sequentially; `None` at EOF
  - `is_open` tracks EOF state
  - `name` returns filename stem; `index` returns `-1` (sentinel)
  - Context manager support
- **Files to Create:** `src/aprilcam/camera/video_camera.py`
- **Files to Modify:** `src/aprilcam/camera/__init__.py`
- **Key Details:** Opens video file directly; bypasses `Camera._open()`; enables video-based testing without hardware

---

### **Ticket 008: DetectionPipeline: background thread orchestrating detect-track-velocity-buffer**
- **Key Acceptance Criteria:**
  - `start()` starts background thread; idempotent
  - `stop()` stops thread; idempotent
  - `is_running` property
  - `on_frame(callback)` registers callback invoked after each frame
  - Loop: read → grayscale → detect (every N frames) → track → velocity → buffer write
  - Per-tag `VelocityEstimator` instances recycled across frames
  - Thread-safe ring buffer
- **Files to Create:** `src/aprilcam/core/pipeline.py`
- **Files to Modify:** `src/aprilcam/core/__init__.py`, optionally `src/aprilcam/core/aprilcam.py`
- **Key Details:** Evolves from existing `DetectionLoop`; daemon thread; stop signal via `threading.Event`

---

### **Ticket 009: Tag class: live tag handle with flat properties and update/position_at**
- **Key Acceptance Criteria:**
  - `Tag(tag_id, pipeline)` constructor
  - Flat properties: `id`, `cx`, `cy`, `wx`, `wy`, `orientation`, `velocity`, `speed`, `heading`, `rotation_rate`, `timestamp`, `age`, `is_visible`
  - `update()` atomically replaces snapshot from ring buffer; returns self
  - `position_at(t)` extrapolates pixel position
  - `to_dict()` returns flat dict
- **Files to Create:** `src/aprilcam/core/tag.py`
- **Files to Modify:** `src/aprilcam/core/__init__.py`
- **Key Details:** Read-only view of ring buffer; thread-safe snapshot replacement; returns sensible defaults when tag not seen

---

### **Ticket 010: Playfield rewrite: primary user-facing object owning DetectionPipeline**
- **Key Acceptance Criteria:**
  - `Playfield(camera, width_cm, height_cm, family, calibration)` constructor
  - `start()`, `stop()` control pipeline
  - `tags()` returns dict[int, Tag]; `tag(id)` returns Tag | None
  - `stream()` generator yielding list[Tag] per frame
  - `on_frame(callback)` registers push callback
  - `calibrate()`, `pixel_to_world()`, `world_to_pixel()`, `deskew()`
  - Auto-loads calibration from JSON if file provided
  - Old `Playfield` geometry logic renamed to `PlayfieldBoundary` (private)
- **Files to Modify:** `src/aprilcam/core/playfield.py`, `src/aprilcam/core/__init__.py`
- **Key Details:** New primary user-facing object; owns `DetectionPipeline`, `TagDetector`, `OpticalFlowTracker`; breaking API change from old geometry class

---

### **Ticket 011: CLI live loop: move AprilCam.run() to cli/live_cli.py**
- **Key Acceptance Criteria:**
  - `cli/live_cli.py` uses `Playfield` + `TagTableTUI` directly
  - No dependency on `AprilCam` in CLI layer
  - `aprilcam live` subcommand works end-to-end
  - TUI renders during operation
  - Keyboard interrupt (`Ctrl+C`) stops cleanly
- **Files to Modify:** `src/aprilcam/cli/live_cli.py`, `src/aprilcam/core/aprilcam.py`, `src/aprilcam/cli/__init__.py`
- **Key Details:** Rewrites CLI using new high-level API; removes dependency on `AprilCam`; argument parsing stays in CLI layer

---

### **Ticket 012: Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim**
- **Key Acceptance Criteria:**
  - `aprilcam.Camera`, `aprilcam.Tag`, `aprilcam.Playfield`, `aprilcam.calibrate` all resolve
  - Legacy names (`detect_tags`, `detect_objects`, `AprilCam`) still available (compatibility re-exports)
  - `stream.py` deleted
  - `AprilCam` reduced to minimal shim
  - `uv run pytest` passes
- **Files to Delete:** `src/aprilcam/stream.py`
- **Files to Modify:** `src/aprilcam/__init__.py`, `src/aprilcam/core/aprilcam.py`
- **Key Details:** Compatibility wrappers for old API; `detect_tags` wraps `Playfield.stream()`; `AprilCam` delegates to new classes

---

### **Ticket 013: Smoke tests: import and construction sanity checks**
- **Key Acceptance Criteria:**
  - `tests/smoke/test_imports.py` verifies all new classes import cleanly
  - `tests/smoke/test_construction.py` verifies basic construction (no hardware)
  - All smoke tests complete in under 5 seconds
  - Verify imports: `Camera`, `Tag`, `Playfield`, `calibrate`, `VideoCamera`, legacy (`detect_tags`, `AprilCam`)
  - Verify construction: `TagDetector()`, `DetectorConfig()`, `VelocityEstimator()`, `OpticalFlowTracker()`, `TagTableTUI()`, `VideoCamera(path)`
- **Files to Create:** `tests/smoke/__init__.py`, `tests/smoke/test_imports.py`, `tests/smoke/test_construction.py`
- **Files to Modify:** `tests/conftest.py` (if needed)
- **Key Details:** Fast hardware-free sanity checks; `pytest.skip` for Rich if not installed; first defense against import errors

---

### **Ticket 014: Unit tests: per-class tests for all new OOP classes**
- **Key Acceptance Criteria:**
  - Per-class test files for: `Camera`, `VideoCamera`, `TagDetector`, `OpticalFlowTracker`, `VelocityEstimator`, `DetectionPipeline`, `Tag`, `TagRecord`, `RingBuffer`
  - Test key methods: list, find, read, detect, update, reset, on_frame
  - No hardware required
  - All tests under 30 seconds total
- **Files to Create:** `tests/unit/__init__.py`, `tests/unit/test_*.py` (9 test files)
- **Key Details:** Use `VideoCamera` for pipeline tests; synthetic frames for detector/tracker; mock pipeline for `Tag` tests

---

### **Ticket 015: System tests: video-driven pipeline tests using tests/movies/*.mov**
- **Key Acceptance Criteria:**
  - `tests/system/test_detection_pipeline_videos.py` with parametrized tests
  - Test all four videos: `bright-gsc.mov`, `bright-ov9782.mov`, `dim-gsc.mov`, `dim-ov9782.mov`
  - Pipeline runs to EOF without error
  - At least one tag detected per video (or xfail for dim videos)
  - Detected positions within frame dimensions
  - Extrapolation (`estimate(t)`) sanity check
- **Files to Create:** `tests/system/__init__.py`, `tests/system/test_detection_pipeline_videos.py`
- **Key Details:** Full end-to-end pipeline tests; no hardware required; under 60 seconds total for all videos

---

## Summary

The 15 tickets are sequenced to:
1. **Tickets 001–007:** Core class extraction (Camera, TagDetector, VelocityEstimator, TUI, Calibration, OpticalFlowTracker, VideoCamera)
2. **Tickets 008–010:** Pipeline orchestration (DetectionPipeline, Tag, Playfield rewrite)
3. **Tickets 011–012:** CLI migration and API stabilization
4. **Tickets 013–015:** Testing (smoke, unit, system)

Each ticket has clear dependencies, enabling parallelizable work. The refactor transforms a monolithic `AprilCam` into composable, testable classes with a clean public API.

---


# Raw JSON Transcript

```json
[
  {
    "parentUuid": null,
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": "Read ALL ticket files in /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/ and provide a concise summary of each ticket's acceptance criteria and implementation plan. There are 15 tickets (001 through 015). Read each one completely.\n\nFor each ticket report:\n- Ticket number and title\n- Key acceptance criteria (bulleted)\n- Files to create/modify\n- Key implementation details"
    },
    "uuid": "0b31f5c6-d19e-40f6-9553-500c18184756",
    "timestamp": "2026-04-11T22:04:33.603Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "0b31f5c6-d19e-40f6-9553-500c18184756",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01NntoQafPogC6bibYnwHG97",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "I'll read all 15 ticket files from the specified directory. Let me start by listing the tickets and then reading each one."
        }
      ],
      "stop_reason": "tool_use",
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 3,
        "cache_creation_input_tokens": 34338,
        "cache_read_input_tokens": 0,
        "output_tokens": 132,
        "server_tool_use": {
          "web_search_requests": 0,
          "web_fetch_requests": 0
        },
        "service_tier": "standard",
        "cache_creation": {
          "ephemeral_1h_input_tokens": 0,
          "ephemeral_5m_input_tokens": 34338
        },
        "inference_geo": "",
        "iterations": [
          {
            "input_tokens": 3,
            "output_tokens": 132,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 34338,
            "cache_creation": {
              "ephemeral_5m_input_tokens": 34338,
              "ephemeral_1h_input_tokens": 0
            },
            "type": "message"
          }
        ],
        "speed": "standard"
      }
    },
    "requestId": "req_011CZxjx58Ugpm7MJh4AfszE",
    "type": "assistant",
    "uuid": "2aacd925-6708-42b5-9b41-522a8799bd23",
    "timestamp": "2026-04-11T22:04:35.845Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "2aacd925-6708-42b5-9b41-522a8799bd23",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01NntoQafPogC6bibYnwHG97",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01PrJKFUEENyHNaAXPoUtZs4",
          "name": "Bash",
          "input": {
            "command": "ls -1 /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": "tool_use",
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 3,
        "cache_creation_input_tokens": 34338,
        "cache_read_input_tokens": 0,
        "output_tokens": 132,
        "server_tool_use": {
          "web_search_requests": 0,
          "web_fetch_requests": 0
        },
        "service_tier": "standard",
        "cache_creation": {
          "ephemeral_1h_input_tokens": 0,
          "ephemeral_5m_input_tokens": 34338
        },
        "inference_geo": "",
        "iterations": [
          {
            "input_tokens": 3,
            "output_tokens": 132,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 34338,
            "cache_creation": {
              "ephemeral_5m_input_tokens": 34338,
              "ephemeral_1h_input_tokens": 0
            },
            "type": "message"
          }
        ],
        "speed": "standard"
      }
    },
    "requestId": "req_011CZxjx58Ugpm7MJh4AfszE",
    "type": "assistant",
    "uuid": "c5c814fd-8c50-4391-a0a6-40d5603b2fc5",
    "timestamp": "2026-04-11T22:04:35.929Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "c5c814fd-8c50-4391-a0a6-40d5603b2fc5",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01PrJKFUEENyHNaAXPoUtZs4",
          "type": "tool_result",
          "content": "001-camera-class-device-discovery-and-videocapture-wrapper.md\n002-tagdetector-pure-stateless-detection-engine.md\n003-velocityestimator-per-tag-ema-velocity-with-deadband.md\n004-tagtabletui-extract-rich-tui-dashboard-from-aprilcam.md\n005-calibration-split-extract-calibrate-and-types-from-homography-py.md\n006-opticalflowtracker-lk-tracking-with-4-corner-motion-decomposition.md\n007-videocamera-camera-subclass-reading-frames-from-a-video-file.md\n008-detectionpipeline-background-thread-orchestrating-detect-track-velocity-buffer.md\n009-tag-class-live-tag-handle-with-flat-properties-and-update-position-at.md\n010-playfield-rewrite-primary-user-facing-object-owning-detectionpipeline.md\n011-cli-live-loop-move-aprilcam-run-to-cli-live-cli-py.md\n012-update-init-py-export-camera-tag-calibrate-delete-stream-py-shrink-aprilcam-shim.md\n013-smoke-tests-import-and-construction-sanity-checks.md\n014-unit-tests-per-class-tests-for-all-new-oop-classes.md\n015-system-tests-video-driven-pipeline-tests-using-tests-movies-mov.md\ndone\n[rerun: b1]",
          "is_error": false
        }
      ]
    },
    "uuid": "9815c380-7ea0-439e-a613-67fe5d646437",
    "timestamp": "2026-04-11T22:04:36.001Z",
    "sourceToolAssistantUUID": "c5c814fd-8c50-4391-a0a6-40d5603b2fc5",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "9815c380-7ea0-439e-a613-67fe5d646437",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Now let me read all 15 ticket files:"
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "d3c27dbf-66da-4a90-993a-0c40c57839ba",
    "timestamp": "2026-04-11T22:04:37.263Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "d3c27dbf-66da-4a90-993a-0c40c57839ba",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_012kpUzh7DQzuVEAdmStENoP",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/001-camera-class-device-discovery-and-videocapture-wrapper.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "3439ac78-d8eb-4687-9016-cbc9539bad3d",
    "timestamp": "2026-04-11T22:04:38.224Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "3439ac78-d8eb-4687-9016-cbc9539bad3d",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_012kpUzh7DQzuVEAdmStENoP",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '001'\n3\ttitle: 'Camera class: device discovery and VideoCapture wrapper'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-001\n7\t  - SUC-002\n8\tdepends-on: []\n9\tgithub-issue: ''\n10\ttodo:\n11\t  - docs/clasi/todo/oop-refactoring.md\n12\t---\n13\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n14\t\n15\t# Camera class: device discovery and VideoCapture wrapper\n16\t\n17\t## Description\n18\t\n19\tCreate `src/aprilcam/camera/camera.py` containing the `Camera` class.\n20\t`Camera` wraps `cv.VideoCapture` with device metadata and provides class-level\n21\tdiscovery methods. It replaces scattered VideoCapture management that currently\n22\tlives inside `AprilCam.__init__` and `stream.py`.\n23\t\n24\t`Camera` must not open hardware in `__init__` \u2014 the device should only be opened\n25\twhen `read()` is first called (lazy open). This is critical so that `VideoCamera`\n26\tcan subclass `Camera` without triggering hardware access on construction.\n27\t\n28\t## Acceptance Criteria\n29\t\n30\t- [ ] `camera/camera.py` exists with `Camera` class.\n31\t- [ ] `Camera.list() -> list[Camera]` enumerates available cameras via `camutil.list_cameras()`.\n32\t- [ ] `Camera.find(pattern: str) -> Camera` returns first camera matching the pattern\n33\t      (case-insensitive substring). Raises `CameraNotFoundError` if no match.\n34\t- [ ] `camera.name: str` \u2014 human-readable device name.\n35\t- [ ] `camera.index: int` \u2014 OpenCV device index.\n36\t- [ ] `camera.resolution: tuple[int, int] | None` \u2014 (width, height) if open, else None.\n37\t- [ ] `camera.is_open: bool` \u2014 True if VideoCapture is open.\n38\t- [ ] `camera.read() -> np.ndarray | None` \u2014 returns BGR frame or None on failure;\n39\t      opens capture lazily on first call.\n40\t- [ ] `camera.close()` \u2014 releases VideoCapture; idempotent.\n41\t- [ ] Context manager: `with Camera.find(\"Brio\") as cam:` calls `close()` on exit.\n42\t- [ ] `camera/__init__.py` exports `Camera`.\n43\t\n44\t## Implementation Plan\n45\t\n46\t### Approach\n47\t\n48\tLift camera management from `AprilCam.__init__` (lines ~80-130) and delegate\n49\tdiscovery to existing `camutil` functions. `Camera` is a thin wrapper.\n50\t\n51\t### Files to Create\n52\t\n53\t- `src/aprilcam/camera/camera.py`\n54\t\n55\t### Files to Modify\n56\t\n57\t- `src/aprilcam/camera/__init__.py` \u2014 add `Camera` to exports\n58\t\n59\t### Key Implementation Notes\n60\t\n61\t- Store capture as `self._cap: cv.VideoCapture | None = None`.\n62\t- `Camera.__init__(index: int, name: str, *, backend: int | None = None)`.\n63\t- `_open()` calls `cv.VideoCapture(self.index)` (or with backend).\n64\t- `read()` calls `_open()` if `_cap is None`, then `_cap.read()`.\n65\t- `Camera.list()` wraps each `CameraInfo` from `camutil` into a `Camera`.\n66\t- `Camera.find()` calls `camutil.select_camera_by_pattern()`.\n67\t\n68\t### Testing Plan\n69\t\n70\t- Smoke: `from aprilcam.camera import Camera` imports cleanly.\n71\t- Smoke: `Camera.list()` returns a list (may be empty in CI).\n72\t- Unit: `Camera.find(\"__no_such_camera__\")` raises `CameraNotFoundError`.\n73\t- Unit: `camera.is_open` is False before `read()` on a hardware-free stub.\n74\t- Unit: context manager calls `close()` on exit.\n75\t\n76\t### Documentation Updates\n77\t\n78\t- Docstrings on `Camera`, `list()`, `find()`, `read()`, `close()`.\n79\t"
        }
      ]
    },
    "uuid": "9323697f-d126-4f11-930e-9d2f7d9854f3",
    "timestamp": "2026-04-11T22:04:38.260Z",
    "sourceToolAssistantUUID": "3439ac78-d8eb-4687-9016-cbc9539bad3d",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "9323697f-d126-4f11-930e-9d2f7d9854f3",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_013HxqrvZpQp5ovW62TkPW8K",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/002-tagdetector-pure-stateless-detection-engine.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "ca399a5a-e361-4989-8d1d-44c516fe943d",
    "timestamp": "2026-04-11T22:04:38.819Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "ca399a5a-e361-4989-8d1d-44c516fe943d",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_013HxqrvZpQp5ovW62TkPW8K",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '002'\n3\ttitle: 'TagDetector: pure stateless detection engine'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-003\n7\tdepends-on: []\n8\tgithub-issue: ''\n9\ttodo:\n10\t  - docs/clasi/todo/oop-refactoring.md\n11\t---\n12\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n13\t\n14\t# TagDetector: pure stateless detection engine\n15\t\n16\t## Description\n17\t\n18\tCreate `src/aprilcam/core/detector.py` with `DetectorConfig` and `TagDetector`.\n19\t\n20\t`TagDetector` is a pure, stateless detection engine: given a BGR frame, return\n21\tdetections. No camera, no threads, no ring buffer. This extracts and consolidates\n22\t`AprilCam._build_detectors()`, `_maybe_preprocess()`, `detect_apriltags()`, and\n23\tthe duplicate module-level versions of those functions from `aprilcam.py`.\n24\t\n25\t## Acceptance Criteria\n26\t\n27\t- [ ] `core/detector.py` exists with `DetectorConfig` dataclass and `TagDetector` class.\n28\t- [ ] `DetectorConfig` fields: `family`, `quad_decimate`, `quad_sigma`, `corner_refine`,\n29\t      `detect_inverted`, `use_clahe`, `use_sharpen`, `use_highpass`, `highpass_ksize`,\n30\t      `april_min_wb_diff`, `april_min_cluster_pixels`, `april_max_line_fit_mse`,\n31\t      `detect_aruco_4x4`, `proc_width` \u2014 all with sensible defaults.\n32\t- [ ] `TagDetector(config: DetectorConfig | None = None)` constructs detector objects.\n33\t- [ ] `TagDetector.detect(frame_bgr: np.ndarray) -> list[Detection]` runs preprocessing\n34\t      and detection; returns list of `Detection` objects.\n35\t- [ ] `Detection` dataclass: `id: int`, `center: tuple[float, float]`,\n36\t      `corners: np.ndarray` (4x2), `family: str`.\n37\t- [ ] Calling `detect()` twice on the same frame returns equivalent results (stateless).\n38\t- [ ] Duplicate module-level `build_detectors()` / `detect_apriltags()` removed from\n39\t      `aprilcam.py`; `aprilcam.py` delegates to `TagDetector` instead.\n40\t- [ ] `core/__init__.py` exports `TagDetector`, `DetectorConfig`.\n41\t\n42\t## Implementation Plan\n43\t\n44\t### Approach\n45\t\n46\t1. Define `Detection` dataclass in `detector.py`.\n47\t2. Define `DetectorConfig` dataclass with all tuning params.\n48\t3. `TagDetector.__init__` builds detector objects (lifted from `_build_detectors()`).\n49\t4. `TagDetector.detect()` runs preprocessing + detection (lifted from\n50\t   `_maybe_preprocess()` + `detect_apriltags()`).\n51\t5. Update `aprilcam.py`: create a `TagDetector` in `__init__`, delegate to it.\n52\t6. Remove duplicate module-level functions from `aprilcam.py`.\n53\t\n54\t### Files to Create\n55\t\n56\t- `src/aprilcam/core/detector.py`\n57\t\n58\t### Files to Modify\n59\t\n60\t- `src/aprilcam/core/aprilcam.py` \u2014 delegate to `TagDetector`; remove duplicates\n61\t- `src/aprilcam/core/__init__.py` \u2014 export `TagDetector`, `DetectorConfig`\n62\t\n63\t### Key Implementation Notes\n64\t\n65\t- `family=\"all\"` means build one detector per family, run all, merge results.\n66\t- The preprocessing pipeline (highpass, CLAHE, sharpen, resize) lives inside\n67\t  `detect()` \u2014 no separate public method needed.\n68\t- `TagDetector` stores the built detector objects as instance attributes; only\n69\t  `detect()` is called per-frame.\n70\t\n71\t### Testing Plan\n72\t\n73\t- Smoke: `from aprilcam.core import TagDetector` succeeds.\n74\t- Smoke: `TagDetector()` constructs without error.\n75\t- Unit: `TagDetector().detect(np.ones((480,640,3), dtype=np.uint8)*255)` returns `[]`.\n76\t- Unit: calling `detect()` twice returns equivalent results.\n77\t- Unit: `DetectorConfig()` has expected default values.\n78\t\n79\t### Documentation Updates\n80\t\n81\t- Docstrings on `DetectorConfig`, `TagDetector`, `detect()`, `Detection`.\n82\t"
        }
      ]
    },
    "uuid": "8f22babe-1e3b-4201-8c4b-219afbfaee38",
    "timestamp": "2026-04-11T22:04:38.852Z",
    "sourceToolAssistantUUID": "ca399a5a-e361-4989-8d1d-44c516fe943d",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "8f22babe-1e3b-4201-8c4b-219afbfaee38",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01GH8Pn8RvkoJv3BttrxzSU8",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/003-velocityestimator-per-tag-ema-velocity-with-deadband.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "f6d1bb2c-7cc2-41fc-b754-98d300583f66",
    "timestamp": "2026-04-11T22:04:39.435Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "f6d1bb2c-7cc2-41fc-b754-98d300583f66",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01GH8Pn8RvkoJv3BttrxzSU8",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '003'\n3\ttitle: 'VelocityEstimator: per-tag EMA velocity with deadband'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-005\n7\tdepends-on: []\n8\tgithub-issue: ''\n9\ttodo:\n10\t  - docs/clasi/todo/oop-refactoring.md\n11\t---\n12\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n13\t\n14\t# VelocityEstimator: per-tag EMA velocity with deadband\n15\t\n16\t## Description\n17\t\n18\tCreate `src/aprilcam/core/motion.py` with `VelocityEstimator`.\n19\t\n20\t`VelocityEstimator` is per-tag stateful object tracking position history with\n21\tEMA smoothing and deadband suppression. It absorbs the velocity EMA block from\n22\t`Playfield.add_tag()` in `playfield.py`. Designed so callers can swap EMA for\n23\ta Kalman filter without changing the interface.\n24\t\n25\t## Acceptance Criteria\n26\t\n27\t- [ ] `core/motion.py` exists with `VelocityEstimator` class.\n28\t- [ ] `VelocityEstimator(alpha: float = 0.3, deadband: float = 50.0)` constructor.\n29\t- [ ] `update(position: tuple[float, float], timestamp: float) -> tuple[tuple[float, float], float]`\n30\t      returns `(velocity_vec, speed)`. Returns `((0.0, 0.0), 0.0)` on first call.\n31\t- [ ] EMA: `v_ema = alpha * v_raw + (1 - alpha) * v_prev`.\n32\t- [ ] Deadband: if raw `speed < deadband`, reports `((0.0, 0.0), 0.0)`.\n33\t- [ ] `predict_position(t: float) -> tuple[float, float]` extrapolates linearly from\n34\t      last known position using last computed velocity.\n35\t- [ ] `reset()` clears all internal state.\n36\t- [ ] `Playfield.add_tag()` updated to delegate EMA velocity to `VelocityEstimator`.\n37\t- [ ] `core/__init__.py` exports `VelocityEstimator`.\n38\t\n39\t## Implementation Plan\n40\t\n41\t### Approach\n42\t\n43\tLift `_vel_ema`, `_last_seen` EMA logic from `Playfield.add_tag()` into this\n44\tclass. The world-velocity homography transform stays in `Playfield`/`pipeline.py`\n45\tsince it needs the homography matrix; `VelocityEstimator` handles pixel-space only.\n46\t\n47\t### Files to Create\n48\t\n49\t- `src/aprilcam/core/motion.py`\n50\t\n51\t### Files to Modify\n52\t\n53\t- `src/aprilcam/core/playfield.py` \u2014 delegate velocity EMA to `VelocityEstimator`\n54\t- `src/aprilcam/core/__init__.py` \u2014 export `VelocityEstimator`\n55\t\n56\t### Key Implementation Notes\n57\t\n58\t- One `VelocityEstimator` instance per tracked tag ID.\n59\t- `speed = hypot(vx, vy)`. Deadband applied to raw speed before EMA.\n60\t- Store `_last_pos`, `_last_time`, `_vel_ema` as instance attributes.\n61\t\n62\t### Testing Plan\n63\t\n64\t- Smoke: `from aprilcam.core import VelocityEstimator` succeeds.\n65\t- Unit: first `update()` returns zero velocity.\n66\t- Unit: constant velocity across two updates returns correct vector.\n67\t- Unit: sub-deadband speed returns zero.\n68\t- Unit: `predict_position(t)` extrapolates correctly.\n69\t- Unit: `reset()` clears state so next `update()` gives zero velocity.\n70\t\n71\t### Documentation Updates\n72\t\n73\t- Docstrings on `VelocityEstimator`, `update()`, `predict_position()`, `reset()`.\n74\t"
        }
      ]
    },
    "uuid": "83aa9cc9-4616-4870-82f2-955fdd632958",
    "timestamp": "2026-04-11T22:04:39.469Z",
    "sourceToolAssistantUUID": "f6d1bb2c-7cc2-41fc-b754-98d300583f66",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "83aa9cc9-4616-4870-82f2-955fdd632958",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01EbMNcV7CYdWUb1J76Y9Yrw",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/004-tagtabletui-extract-rich-tui-dashboard-from-aprilcam.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "8220a0ab-41ba-45f8-937e-9a9e38c82fda",
    "timestamp": "2026-04-11T22:04:40.069Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "8220a0ab-41ba-45f8-937e-9a9e38c82fda",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01EbMNcV7CYdWUb1J76Y9Yrw",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '004'\n3\ttitle: 'TagTableTUI: extract Rich TUI dashboard from AprilCam'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-008\n7\tdepends-on: []\n8\tgithub-issue: ''\n9\ttodo:\n10\t  - docs/clasi/todo/oop-refactoring.md\n11\t---\n12\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n13\t\n14\t# TagTableTUI: extract Rich TUI dashboard from AprilCam\n15\t\n16\t## Description\n17\t\n18\tCreate `src/aprilcam/ui/tui.py` with `TagTableTUI`.\n19\t\n20\t`TagTableTUI` is a Rich-based terminal dashboard that displays a live table of\n21\ttag IDs, positions, speeds, and headings. It is extracted from the TUI rendering\n22\tmethods inside `AprilCam` (`_build_tui_layout`, `_print_tui`, `_stop_tui`,\n23\t`_ema_smooth`). After extraction, `TagTableTUI` must have no dependency on\n24\t`AprilCam`; it reads tag data only through the public `Playfield` / `Tag` API.\n25\t\n26\tThis ticket does not wire `TagTableTUI` to the new `Playfield` \u2014 that happens\n27\tin ticket 010. This ticket extracts the code and puts it in its own module with\n28\ta clean interface.\n29\t\n30\t## Acceptance Criteria\n31\t\n32\t- [ ] `ui/tui.py` exists with `TagTableTUI` class.\n33\t- [ ] `TagTableTUI` has no import of `AprilCam`.\n34\t- [ ] `TagTableTUI.__init__(title: str = \"AprilCam\")` initializes the Rich Live display.\n35\t- [ ] `TagTableTUI.update(tags: list[dict])` renders a row per tag with columns:\n36\t      `id`, `cx`, `cy`, `wx`, `wy`, `speed`, `heading`, `orientation`.\n37\t- [ ] `TagTableTUI.start()` and `TagTableTUI.stop()` control the Live context.\n38\t- [ ] Context manager support: `with TagTableTUI() as tui:`.\n39\t- [ ] EMA smoothing for display-only values stays inside `TagTableTUI` (not\n40\t      exported to business logic).\n41\t- [ ] `AprilCam` TUI methods replaced with calls to `TagTableTUI`.\n42\t- [ ] `ui/__init__.py` exports `TagTableTUI`.\n43\t\n44\t## Implementation Plan\n45\t\n46\t### Approach\n47\t\n48\t1. Create `ui/tui.py`.\n49\t2. Lift `_build_tui_layout`, `_print_tui`, `_stop_tui`, `_ema_smooth` from\n50\t   `aprilcam.py` into `TagTableTUI` methods.\n51\t3. `update(tags)` accepts a list of dicts (not `TagRecord` objects) so the\n52\t   TUI has no dependency on `core` data types.\n53\t4. Update `AprilCam` to instantiate `TagTableTUI` and delegate to it.\n54\t\n55\t### Files to Create\n56\t\n57\t- `src/aprilcam/ui/tui.py`\n58\t\n59\t### Files to Modify\n60\t\n61\t- `src/aprilcam/core/aprilcam.py` \u2014 delegate TUI rendering to `TagTableTUI`\n62\t- `src/aprilcam/ui/__init__.py` \u2014 export `TagTableTUI`\n63\t\n64\t### Key Implementation Notes\n65\t\n66\t- `tags` parameter to `update()` is `list[dict]` with keys matching `TagRecord.to_dict()`.\n67\t- EMA display smoothing is private state inside `TagTableTUI`.\n68\t- If Rich is not installed, `TagTableTUI` should raise `ImportError` with a clear message.\n69\t\n70\t### Testing Plan\n71\t\n72\t- Smoke: `from aprilcam.ui import TagTableTUI` succeeds.\n73\t- Smoke: `TagTableTUI()` constructs without error.\n74\t- Unit: `update([])` renders an empty table without raising.\n75\t- Unit: `update([{\"id\": 1, \"cx\": 100.0, ...}])` renders a one-row table.\n76\t- (No hardware needed; TUI output can be captured to a string buffer.)\n77\t\n78\t### Documentation Updates\n79\t\n80\t- Docstrings on `TagTableTUI`, `update()`, `start()`, `stop()`.\n81\t"
        }
      ]
    },
    "uuid": "3ea2093c-e35b-4644-b2c6-338cb6743341",
    "timestamp": "2026-04-11T22:04:40.102Z",
    "sourceToolAssistantUUID": "8220a0ab-41ba-45f8-937e-9a9e38c82fda",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "3ea2093c-e35b-4644-b2c6-338cb6743341",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01BdnACBAWihjdpoMV5NQBeH",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/005-calibration-split-extract-calibrate-and-types-from-homography-py.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "f3d22416-c971-487e-a5a8-97bc0135622b",
    "timestamp": "2026-04-11T22:04:40.550Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "f3d22416-c971-487e-a5a8-97bc0135622b",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01BdnACBAWihjdpoMV5NQBeH",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '005'\n3\ttitle: 'Calibration split: extract calibrate() and types from homography.py'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-009\n7\tdepends-on: []\n8\tgithub-issue: ''\n9\ttodo:\n10\t  - docs/clasi/todo/oop-refactoring.md\n11\t---\n12\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n13\t\n14\t# Calibration split: extract calibrate() and types from homography.py\n15\t\n16\t## Description\n17\t\n18\tCreate `src/aprilcam/calibration/calibration.py` by splitting the current\n19\t`homography.py`. Pure math (homography matrix computation, coordinate\n20\ttransforms) stays in `homography.py`. The calibration workflow (`calibrate()`\n21\tfunction), `CameraCalibration` dataclass, and `FieldSpec` dataclass move to\n22\t`calibration.py`.\n23\t\n24\t`calibration.py` owns: detect ArUco corners, compute homography, persist JSON,\n25\tload JSON. `homography.py` owns: matrix math, coordinate transforms.\n26\t\n27\t## Acceptance Criteria\n28\t\n29\t- [ ] `calibration/calibration.py` exists with `calibrate()`, `CameraCalibration`,\n30\t      `FieldSpec`.\n31\t- [ ] `calibrate()` function: detect corners, compute homography, write JSON to disk.\n32\t- [ ] `CameraCalibration` dataclass: contains `homography: np.ndarray`, `field_spec: FieldSpec`,\n33\t      `device_name: str`, `resolution: tuple[int, int]`.\n34\t- [ ] `FieldSpec` dataclass: `width_cm`, `height_cm` (and unit-conversion helpers).\n35\t- [ ] `homography.py` retains only pure math functions (no `calibrate()`, no `FieldSpec`,\n36\t      no `CameraCalibration`).\n37\t- [ ] All existing callers of `CameraCalibration`, `FieldSpec`, `calibrate()`\n38\t      updated to import from `calibration.calibration` (not `calibration.homography`).\n39\t- [ ] `calibration/__init__.py` exports `calibrate`, `CameraCalibration`, `FieldSpec`.\n40\t- [ ] `from aprilcam.calibration import calibrate` works at the package level.\n41\t\n42\t## Implementation Plan\n43\t\n44\t### Approach\n45\t\n46\t1. Create `calibration/calibration.py`.\n47\t2. Move `FieldSpec`, `CameraCalibration` from `homography.py` to `calibration.py`.\n48\t3. Move the `calibrate()` function from `homography.py` (or `stream.py`) to\n49\t   `calibration.py`.\n50\t4. Add re-exports in `homography.py` for backward compat during transition:\n51\t   `from .calibration import CameraCalibration, FieldSpec`.\n52\t5. Update `calibration/__init__.py`.\n53\t\n54\t### Files to Create\n55\t\n56\t- `src/aprilcam/calibration/calibration.py`\n57\t\n58\t### Files to Modify\n59\t\n60\t- `src/aprilcam/calibration/homography.py` \u2014 remove moved items; add re-exports\n61\t- `src/aprilcam/calibration/__init__.py` \u2014 export new names\n62\t- `src/aprilcam/stream.py` \u2014 update import of `calibrate` (if needed before deletion)\n63\t\n64\t### Key Implementation Notes\n65\t\n66\t- `FieldSpec` conversion helpers (`width_cm`, `height_cm` properties from inches)\n67\t  stay on `FieldSpec`.\n68\t- `calibrate()` reads from a `Camera` (or path) and a `FieldSpec`, writes\n69\t  `CameraCalibration` to JSON.\n70\t- The existing `load_calibration_for_camera()` may stay in `homography.py` or\n71\t  move to `calibration.py` \u2014 move it to `calibration.py` since it is part of\n72\t  the calibration workflow.\n73\t\n74\t### Testing Plan\n75\t\n76\t- Smoke: `from aprilcam.calibration import calibrate, CameraCalibration, FieldSpec` succeeds.\n77\t- Unit: `FieldSpec(width_in=40, height_in=35, units=\"inch\").width_cm` returns correct value.\n78\t- Unit: `CameraCalibration` can be round-tripped to/from JSON dict.\n79\t\n80\t### Documentation Updates\n81\t\n82\t- Docstrings on `calibrate()`, `CameraCalibration`, `FieldSpec`.\n83\t"
        }
      ]
    },
    "uuid": "3e431283-46d1-41e5-94eb-15cea4b56002",
    "timestamp": "2026-04-11T22:04:40.580Z",
    "sourceToolAssistantUUID": "f3d22416-c971-487e-a5a8-97bc0135622b",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "3e431283-46d1-41e5-94eb-15cea4b56002",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_0195EmPpNeoD82XC8rU72jtT",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/006-opticalflowtracker-lk-tracking-with-4-corner-motion-decomposition.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "0e854346-a985-4ffb-b578-811912b306d7",
    "timestamp": "2026-04-11T22:04:41.119Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "0e854346-a985-4ffb-b578-811912b306d7",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_0195EmPpNeoD82XC8rU72jtT",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '006'\n3\ttitle: 'OpticalFlowTracker: LK tracking with 4-corner motion decomposition'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-004\n7\tdepends-on:\n8\t  - \"002\"\n9\tgithub-issue: ''\n10\ttodo:\n11\t  - docs/clasi/todo/oop-refactoring.md\n12\t---\n13\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n14\t\n15\t# OpticalFlowTracker: LK tracking with 4-corner motion decomposition\n16\t\n17\t## Description\n18\t\n19\tCreate `src/aprilcam/core/tracker.py` with `OpticalFlowTracker`.\n20\t\n21\t`OpticalFlowTracker` applies Lucas-Kanade optical flow to the 4 corners of each\n22\tknown tag between full-detection frames. It produces a full per-tag motion\n23\tdecomposition: 2D translation, rotation rate (yaw delta), and scale delta\n24\t(z-axis proxy). All 4-corner data is preserved \u2014 no information is discarded\n25\tto just center position.\n26\t\n27\tThis absorbs `AprilCam.lk_track()` and the detect-or-track branching logic\n28\tfrom `process_frame()` in `aprilcam.py`.\n29\t\n30\t## Acceptance Criteria\n31\t\n32\t- [ ] `core/tracker.py` exists with `OpticalFlowTracker` class.\n33\t- [ ] `OpticalFlowTracker()` default constructor.\n34\t- [ ] `update(gray: np.ndarray, detections: list[Detection] | None) -> list[Detection]`:\n35\t      - When `detections` is not None: reset tracked points to new detections.\n36\t      - When `detections` is None: propagate existing tracked points via LK flow.\n37\t      - Returns list of `Detection` objects with updated `corners` and `center`.\n38\t- [ ] Output `Detection` objects include a `motion` field (or updated corners)\n39\t      preserving 2D translation, rotation rate, and scale delta from 4-corner tracking.\n40\t- [ ] `OpticalFlowTracker.reset()` clears all tracked state.\n41\t- [ ] Absorbs `AprilCam.lk_track()` and detect-or-track branching from `process_frame()`.\n42\t- [ ] `aprilcam.py` delegates to `OpticalFlowTracker`; no duplicate LK code remains.\n43\t- [ ] `core/__init__.py` exports `OpticalFlowTracker`.\n44\t\n45\t## Implementation Plan\n46\t\n47\t### Approach\n48\t\n49\t1. Create `tracker.py`. Define `OpticalFlowTracker`.\n50\t2. Lift `lk_track()` from `aprilcam.py` (~lines 400-480) into `update()`.\n51\t3. Lift the detect-or-track branching from `process_frame()` into `update()`.\n52\t4. Extend `Detection` (from ticket 002) with optional `translation`, `rotation_rate`,\n53\t   `scale_delta` fields (all default to None for standard detections).\n54\t5. Update `aprilcam.py` to instantiate `OpticalFlowTracker` and delegate.\n55\t\n56\t### Files to Create\n57\t\n58\t- `src/aprilcam/core/tracker.py`\n59\t\n60\t### Files to Modify\n61\t\n62\t- `src/aprilcam/core/detector.py` \u2014 extend `Detection` with motion fields (or add `TrackedDetection` subtype)\n63\t- `src/aprilcam/core/aprilcam.py` \u2014 delegate tracking to `OpticalFlowTracker`\n64\t- `src/aprilcam/core/__init__.py` \u2014 export `OpticalFlowTracker`\n65\t\n66\t### Key Implementation Notes\n67\t\n68\t- LK params: `winSize=(21, 21)`, `maxLevel=3`, standard termination criteria.\n69\t- 4-corner tracking: store `prev_gray` and `prev_pts` (N\u00d74\u00d72 array) as instance state.\n70\t- Rotation rate from 4-corner: fit rigid body transform (translation + rotation)\n71\t  using `cv.estimateAffinePartial2D` on the corner pairs.\n72\t- Scale delta: ratio of mean corner span before vs. after.\n73\t- When LK fails for a point (status=0), fall back to center-only approximation\n74\t  rather than dropping the tag entirely.\n75\t\n76\t### Testing Plan\n77\t\n78\t- Smoke: `from aprilcam.core import OpticalFlowTracker` succeeds.\n79\t- Unit: `update(gray, detections=[...])` resets and returns same detections with no motion.\n80\t- Unit: `update(gray2, None)` propagates points and returns updated positions.\n81\t- Unit: `reset()` clears state so next `update(gray, None)` returns empty list.\n82\t- Use two synthetic grayscale frames with known shift for the propagation test.\n83\t\n84\t### Documentation Updates\n85\t\n86\t- Docstrings on `OpticalFlowTracker`, `update()`, `reset()`.\n87\t"
        }
      ]
    },
    "uuid": "7ed57f78-c41f-45d7-9e4c-77060cf0b08b",
    "timestamp": "2026-04-11T22:04:41.149Z",
    "sourceToolAssistantUUID": "0e854346-a985-4ffb-b578-811912b306d7",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "7ed57f78-c41f-45d7-9e4c-77060cf0b08b",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01Bf4ufaxwaUC3wngssEHmwW",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/007-videocamera-camera-subclass-reading-frames-from-a-video-file.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "f4db745f-3f5c-42d7-a24c-77326157ef87",
    "timestamp": "2026-04-11T22:04:41.751Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "f4db745f-3f5c-42d7-a24c-77326157ef87",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01Bf4ufaxwaUC3wngssEHmwW",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '007'\n3\ttitle: 'VideoCamera: Camera subclass reading frames from a video file'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-002\n7\t  - SUC-010\n8\tdepends-on:\n9\t  - \"001\"\n10\tgithub-issue: ''\n11\ttodo:\n12\t  - docs/clasi/todo/video-based-test-system.md\n13\t---\n14\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n15\t\n16\t# VideoCamera: Camera subclass reading frames from a video file\n17\t\n18\t## Description\n19\t\n20\tCreate `src/aprilcam/camera/video_camera.py` with `VideoCamera(Camera)`.\n21\t\n22\t`VideoCamera` subclasses `Camera` and reads frames sequentially from a `.mov`\n23\t(or other OpenCV-readable) video file. It provides the same `read()` / `close()` /\n24\tcontext manager interface as `Camera`. At EOF, `read()` returns `None`.\n25\t\n26\tThis enables the full detection pipeline to run against recorded test videos\n27\twithout any connected camera hardware \u2014 the foundation of the video-based test system.\n28\t\n29\t## Acceptance Criteria\n30\t\n31\t- [ ] `camera/video_camera.py` exists with `VideoCamera(Camera)` class.\n32\t- [ ] `VideoCamera(path: str | Path)` constructor opens the video file.\n33\t- [ ] `video_camera.read() -> np.ndarray | None` returns frames sequentially;\n34\t      returns `None` at EOF.\n35\t- [ ] `video_camera.is_open` is True while frames remain, False after EOF or close.\n36\t- [ ] `video_camera.close()` releases the VideoCapture; idempotent.\n37\t- [ ] Context manager: `with VideoCamera(path) as cam:` closes on exit.\n38\t- [ ] `video_camera.name` returns the filename stem.\n39\t- [ ] `video_camera.index` returns -1 (sentinel for file-based camera).\n40\t- [ ] `VideoCamera` does not call `Camera._open()` (no hardware access).\n41\t- [ ] `camera/__init__.py` exports `VideoCamera`.\n42\t- [ ] `from aprilcam.camera import VideoCamera` works at package level.\n43\t\n44\t## Implementation Plan\n45\t\n46\t### Approach\n47\t\n48\t1. Create `video_camera.py`. `VideoCamera.__init__(path)` opens a\n49\t   `cv.VideoCapture(str(path))` directly \u2014 bypasses `Camera._open()`.\n50\t2. Override `read()` to call `_cap.read()` and return `None` when\n51\t   `_cap.read()` fails (EOF).\n52\t3. Override `name` property to return `Path(path).stem`.\n53\t4. Override `index` property to return `-1`.\n54\t5. `close()` inherited from `Camera` works unchanged.\n55\t6. Add `VideoCamera` to `camera/__init__.py`.\n56\t\n57\t### Files to Create\n58\t\n59\t- `src/aprilcam/camera/video_camera.py`\n60\t\n61\t### Files to Modify\n62\t\n63\t- `src/aprilcam/camera/__init__.py` \u2014 export `VideoCamera`\n64\t\n65\t### Key Implementation Notes\n66\t\n67\t- `VideoCamera.__init__` must NOT call `super().__init__()` in a way that\n68\t  triggers hardware probing. Pass `index=-1, name=path.stem` to `super()`.\n69\t- The `_cap` attribute is set in `__init__` directly (opened from file path).\n70\t- No `_open()` override needed if `_cap` is set before `read()` is called.\n71\t- `read()` return: on EOF `ret=False`, return `None`. Caller checks for `None`.\n72\t\n73\t### Testing Plan\n74\t\n75\t- Smoke: `from aprilcam.camera import VideoCamera` succeeds.\n76\t- Unit: `VideoCamera(\"tests/movies/bright-gsc.mov\")` constructs.\n77\t- Unit: `read()` returns a BGR numpy array on first call.\n78\t- Unit: `read()` returns `None` after all frames are exhausted.\n79\t- Unit: context manager closes on exit.\n80\t- Unit: `video_camera.name == \"bright-gsc\"`.\n81\t\n82\t### Documentation Updates\n83\t\n84\t- Docstrings on `VideoCamera`, `__init__()`, `read()`.\n85\t"
        }
      ]
    },
    "uuid": "ea9e5750-9ca3-466f-a223-671bc642855b",
    "timestamp": "2026-04-11T22:04:41.840Z",
    "sourceToolAssistantUUID": "f4db745f-3f5c-42d7-a24c-77326157ef87",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "ea9e5750-9ca3-466f-a223-671bc642855b",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_015c7DjazesL5xEwZ8BnyrTr",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/008-detectionpipeline-background-thread-orchestrating-detect-track-velocity-buffer.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "d94ee5b2-9b8b-4778-a79c-f58c5002e928",
    "timestamp": "2026-04-11T22:04:42.205Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "d94ee5b2-9b8b-4778-a79c-f58c5002e928",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_015c7DjazesL5xEwZ8BnyrTr",
          "type": "tool_result",
          "content": "1\t---\n2\tid: \"008\"\n3\ttitle: 'DetectionPipeline: background thread orchestrating detect-track-velocity-buffer'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-002\n7\t  - SUC-004\n8\t  - SUC-005\n9\t  - SUC-006\n10\t  - SUC-007\n11\tdepends-on:\n12\t  - \"002\"\n13\t  - \"003\"\n14\t  - \"006\"\n15\t  - \"007\"\n16\tgithub-issue: ''\n17\ttodo:\n18\t  - docs/clasi/todo/oop-refactoring.md\n19\t---\n20\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n21\t\n22\t# DetectionPipeline: background thread orchestrating detect-track-velocity-buffer\n23\t\n24\t## Description\n25\t\n26\tCreate `src/aprilcam/core/pipeline.py` with `DetectionPipeline`.\n27\t\n28\t`DetectionPipeline` runs a background thread that continuously reads frames\n29\tfrom a `Camera` (or `VideoCamera`), runs `TagDetector` every N frames,\n30\truns `OpticalFlowTracker` on all frames, updates per-tag `VelocityEstimator`\n31\tinstances, and writes results to the `RingBuffer`. It evolves from the existing\n32\t`DetectionLoop` in `detection.py`, composing the new extracted classes.\n33\t\n34\tCallers start/stop the pipeline and register `on_frame` callbacks. The\n35\t`Playfield` class (ticket 010) owns and controls a `DetectionPipeline`.\n36\t\n37\t## Acceptance Criteria\n38\t\n39\t- [ ] `core/pipeline.py` exists with `DetectionPipeline` class.\n40\t- [ ] `DetectionPipeline(camera, detector, tracker, *, detect_interval: int = 3,\n41\t      ring_buffer_size: int = 300)` constructor.\n42\t- [ ] `start()` \u2014 starts the background thread; idempotent if already running.\n43\t- [ ] `stop()` \u2014 signals thread to stop; joins thread; idempotent.\n44\t- [ ] `is_running: bool` property.\n45\t- [ ] `ring_buffer: RingBuffer` property \u2014 the live ring buffer.\n46\t- [ ] `on_frame(callback: Callable[[list[TagRecord]], None])` \u2014 registers a\n47\t      callback invoked on the background thread after each frame is processed.\n48\t- [ ] Pipeline loop: read frame \u2192 grayscale \u2192 detect (every N frames) \u2192\n49\t      track \u2192 per-tag velocity \u2192 build `TagRecord` list \u2192 write to ring buffer \u2192\n50\t      call callbacks.\n51\t- [ ] `VelocityEstimator` instances created per tag ID; recycled across frames.\n52\t- [ ] Thread-safe: `ring_buffer` reads from the main thread are safe while\n53\t      the pipeline writes from the background thread.\n54\t- [ ] `core/__init__.py` exports `DetectionPipeline`.\n55\t\n56\t## Implementation Plan\n57\t\n58\t### Approach\n59\t\n60\t1. Create `pipeline.py` with `DetectionPipeline`.\n61\t2. Lift the core loop from `detection.py`'s `DetectionLoop._run_loop()` and\n62\t   `aprilcam.py`'s `process_frame()`.\n63\t3. Replace inline detection/tracking calls with `TagDetector.detect()` and\n64\t   `OpticalFlowTracker.update()`.\n65\t4. Replace inline velocity EMA with `VelocityEstimator.update()`.\n66\t5. Per-tag `VelocityEstimator` map: `dict[int, VelocityEstimator]`.\n67\t6. Existing `DetectionLoop` in `detection.py` is kept for now (compatibility shim).\n68\t\n69\t### Files to Create\n70\t\n71\t- `src/aprilcam/core/pipeline.py`\n72\t\n73\t### Files to Modify\n74\t\n75\t- `src/aprilcam/core/__init__.py` \u2014 export `DetectionPipeline`\n76\t- `src/aprilcam/core/aprilcam.py` \u2014 optionally delegate main loop to `DetectionPipeline`\n77\t\n78\t### Key Implementation Notes\n79\t\n80\t- Use `threading.Thread(daemon=True)` for the background thread.\n81\t- Stop signal: `threading.Event` set by `stop()`, checked in loop.\n82\t- Ring buffer writes are atomic per frame (lock held only during list append).\n83\t- `on_frame` callbacks run synchronously inside the loop; keep them fast.\n84\t- `VideoCamera.read()` returning `None` means EOF \u2014 pipeline should stop cleanly.\n85\t\n86\t### Testing Plan\n87\t\n88\t- Smoke: `from aprilcam.core import DetectionPipeline` succeeds.\n89\t- Unit: `pipeline.start()` sets `is_running = True`.\n90\t- Unit: `pipeline.stop()` after start sets `is_running = False`; thread joined.\n91\t- Unit: with `VideoCamera`, pipeline processes all frames and stops at EOF.\n92\t- Unit: `on_frame` callback called at least once for a multi-frame video.\n93\t- Unit: `ring_buffer` contains `TagRecord` entries after processing.\n94\t\n95\t### Documentation Updates\n96\t\n97\t- Docstrings on `DetectionPipeline`, `start()`, `stop()`, `on_frame()`, `ring_buffer`.\n98\t"
        }
      ]
    },
    "uuid": "7fbaef6c-d3aa-4c26-9599-7f725886935e",
    "timestamp": "2026-04-11T22:04:42.302Z",
    "sourceToolAssistantUUID": "d94ee5b2-9b8b-4778-a79c-f58c5002e928",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "7fbaef6c-d3aa-4c26-9599-7f725886935e",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01UbQmwHoqmrgsw6ep1N3XfV",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/009-tag-class-live-tag-handle-with-flat-properties-and-update-position-at.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "9e16c874-63f8-447d-91ed-b8271b311d27",
    "timestamp": "2026-04-11T22:04:42.805Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "9e16c874-63f8-447d-91ed-b8271b311d27",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01UbQmwHoqmrgsw6ep1N3XfV",
          "type": "tool_result",
          "content": "1\t---\n2\tid: \"009\"\n3\ttitle: 'Tag class: live tag handle with flat properties and update/position_at'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-006\n7\t  - SUC-007\n8\tdepends-on:\n9\t  - \"008\"\n10\tgithub-issue: ''\n11\ttodo:\n12\t  - docs/clasi/todo/oop-refactoring.md\n13\t---\n14\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n15\t\n16\t# Tag class: live tag handle with flat properties and update/position_at\n17\t\n18\t## Description\n19\t\n20\tCreate `src/aprilcam/core/tag.py` with `Tag`.\n21\t\n22\t`Tag` is the user-facing live tag object obtained from `Playfield`. It holds an\n23\tatomically-replaced frozen `TagRecord` snapshot. Callers use flat properties\n24\t(`cx`, `cy`, `wx`, `wy`, etc.) instead of navigating nested `TagRecord` fields.\n25\t`update()` pulls the latest snapshot from the pipeline; `position_at(t)` extrapolates.\n26\t\n27\t`Tag` has no detection logic \u2014 it is purely a read-only view of the ring buffer.\n28\t\n29\t## Acceptance Criteria\n30\t\n31\t- [ ] `core/tag.py` exists with `Tag` class.\n32\t- [ ] `Tag(tag_id: int, pipeline: DetectionPipeline)` constructor.\n33\t- [ ] `tag.id: int` \u2014 tag ID.\n34\t- [ ] `tag.cx: float` \u2014 pixel center x (from latest snapshot).\n35\t- [ ] `tag.cy: float` \u2014 pixel center y.\n36\t- [ ] `tag.wx: float | None` \u2014 world x (cm), or None if uncalibrated.\n37\t- [ ] `tag.wy: float | None` \u2014 world y (cm), or None if uncalibrated.\n38\t- [ ] `tag.orientation: float` \u2014 yaw in radians.\n39\t- [ ] `tag.velocity: tuple[float, float] | None` \u2014 (vx, vy) in world units/s.\n40\t- [ ] `tag.speed: float | None` \u2014 scalar speed.\n41\t- [ ] `tag.heading: float | None` \u2014 `atan2(vy, vx)` in radians.\n42\t- [ ] `tag.rotation_rate: float | None` \u2014 yaw delta per second.\n43\t- [ ] `tag.timestamp: float` \u2014 capture time of latest snapshot.\n44\t- [ ] `tag.age: float` \u2014 seconds since last detected.\n45\t- [ ] `tag.is_visible: bool` \u2014 True if `age == 0.0`.\n46\t- [ ] `tag.update() -> Tag` \u2014 atomically replaces internal snapshot with latest\n47\t      from ring buffer; returns self for chaining.\n48\t- [ ] `tag.position_at(t: float) -> tuple[float, float]` \u2014 extrapolates pixel\n49\t      position; delegates to `TagRecord.estimate(t)`.\n50\t- [ ] `tag.to_dict() -> dict` \u2014 returns flat dict matching `TagRecord.to_dict()` keys.\n51\t- [ ] `core/__init__.py` exports `Tag`.\n52\t\n53\t## Implementation Plan\n54\t\n55\t### Approach\n56\t\n57\t1. Create `core/tag.py`. `Tag` stores a reference to `DetectionPipeline`\n58\t   (or to `RingBuffer` directly) and a cached `TagRecord`.\n59\t2. `update()`: search ring buffer for latest `TagRecord` with `id == self.id`;\n60\t   atomically assign to `self._snapshot` using `threading.Lock`.\n61\t3. All flat properties read from `self._snapshot`; return sensible defaults\n62\t   if `_snapshot is None` (tag not yet seen).\n63\t4. `position_at(t)` calls `self._snapshot.estimate(t).center_px`.\n64\t\n65\t### Files to Create\n66\t\n67\t- `src/aprilcam/core/tag.py`\n68\t\n69\t### Files to Modify\n70\t\n71\t- `src/aprilcam/core/__init__.py` \u2014 export `Tag`\n72\t\n73\t### Key Implementation Notes\n74\t\n75\t- `Tag` is created by `Playfield`, not by callers directly.\n76\t- `_snapshot: TagRecord | None` \u2014 None if tag never seen.\n77\t- Thread safety: `_snapshot` replaced atomically via assignment (GIL protects\n78\t  single assignment in CPython; use a lock for correctness in edge cases).\n79\t- `is_visible`: True when `age == 0.0` (detected in the latest frame).\n80\t\n81\t### Testing Plan\n82\t\n83\t- Smoke: `from aprilcam.core import Tag` succeeds.\n84\t- Unit: `Tag(id=42, pipeline=mock_pipeline)` constructs.\n85\t- Unit: properties return None/zero before first `update()`.\n86\t- Unit: after `update()` with a seeded ring buffer, properties match `TagRecord` values.\n87\t- Unit: `position_at(t)` extrapolates correctly from a known velocity snapshot.\n88\t\n89\t### Documentation Updates\n90\t\n91\t- Docstrings on `Tag`, all properties, `update()`, `position_at()`, `to_dict()`.\n92\t"
        }
      ]
    },
    "uuid": "8ce76b8a-e405-460c-bf4d-e5fd6880e202",
    "timestamp": "2026-04-11T22:04:42.835Z",
    "sourceToolAssistantUUID": "9e16c874-63f8-447d-91ed-b8271b311d27",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "8ce76b8a-e405-460c-bf4d-e5fd6880e202",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01GShiT2AFaM9nxt8zWtrZWe",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/010-playfield-rewrite-primary-user-facing-object-owning-detectionpipeline.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "48db7812-afbf-407f-a555-32b3fe07e074",
    "timestamp": "2026-04-11T22:04:43.465Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "48db7812-afbf-407f-a555-32b3fe07e074",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01GShiT2AFaM9nxt8zWtrZWe",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '010'\n3\ttitle: 'Playfield rewrite: primary user-facing object owning DetectionPipeline'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-006\n7\t  - SUC-007\n8\t  - SUC-009\n9\tdepends-on:\n10\t  - \"005\"\n11\t  - \"008\"\n12\t  - \"009\"\n13\tgithub-issue: ''\n14\ttodo:\n15\t  - docs/clasi/todo/oop-refactoring.md\n16\t---\n17\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n18\t\n19\t# Playfield rewrite: primary user-facing object owning DetectionPipeline\n20\t\n21\t## Description\n22\t\n23\tRewrite `src/aprilcam/core/playfield.py` so that `Playfield` is the primary\n24\tuser-facing object. It owns a `DetectionPipeline`, exposes tag access via\n25\t`tags()` and `tag(id)`, supports both streaming and callback modes, handles\n26\tcalibration, and delegates all geometry/corner work to an internal\n27\t`PlayfieldBoundary`.\n28\t\n29\tThe existing `Playfield` geometry logic (corner detection, polygon, ArUco\n30\tdetection) is renamed `PlayfieldBoundary` and made internal. The public\n31\t`Playfield` API is clean and high-level.\n32\t\n33\t## Acceptance Criteria\n34\t\n35\t- [ ] `Playfield(camera, *, width_cm=None, height_cm=None, family=\"tag36h11\",\n36\t      calibration=None)` constructor.\n37\t- [ ] `Playfield.start()` \u2014 starts the `DetectionPipeline`; idempotent.\n38\t- [ ] `Playfield.stop()` \u2014 stops the pipeline; idempotent.\n39\t- [ ] `playfield.tags() -> dict[int, Tag]` \u2014 returns all currently tracked tags.\n40\t- [ ] `playfield.tag(id: int) -> Tag | None` \u2014 returns a `Tag` for the given ID,\n41\t      or None if never seen.\n42\t- [ ] `playfield.stream() -> Generator[list[Tag], None, None]` \u2014 yields list of\n43\t      `Tag` objects per frame; terminates when `stop()` is called.\n44\t- [ ] `playfield.on_frame(callback: Callable[[list[Tag]], None])` \u2014 registers\n45\t      a push callback.\n46\t- [ ] `playfield.calibrate()` \u2014 runs `calibration.calibrate()` to detect corners\n47\t      and persist homography.\n48\t- [ ] `playfield.pixel_to_world(px, py) -> tuple[float, float] | None`.\n49\t- [ ] `playfield.world_to_pixel(wx, wy) -> tuple[float, float] | None`.\n50\t- [ ] `playfield.deskew(frame) -> np.ndarray` \u2014 returns deskewed frame.\n51\t- [ ] When `calibration` file provided and it exists, loads homography automatically.\n52\t- [ ] Existing `Playfield` geometry code renamed to `PlayfieldBoundary` (private).\n53\t- [ ] The old `Playfield` class is gone from the public namespace; new `Playfield`\n54\t      replaces it.\n55\t- [ ] `core/__init__.py` exports new `Playfield`.\n56\t\n57\t## Implementation Plan\n58\t\n59\t### Approach\n60\t\n61\t1. Rename existing `Playfield` \u2192 `PlayfieldBoundary` in `playfield.py`.\n62\t2. Write the new `Playfield` class in the same file (or a new file, keeping\n63\t   import path identical: `aprilcam.core.playfield`).\n64\t3. `Playfield.__init__` creates `TagDetector`, `OpticalFlowTracker`,\n65\t   `DetectionPipeline`, and optionally loads calibration.\n66\t4. `tags()` queries the ring buffer; `tag(id)` creates/returns a cached `Tag`.\n67\t5. `stream()` is a generator that reads from the ring buffer on each pipeline callback.\n68\t6. `on_frame()` delegates to `DetectionPipeline.on_frame()`.\n69\t7. Geometry methods delegate to `PlayfieldBoundary`.\n70\t\n71\t### Files to Modify\n72\t\n73\t- `src/aprilcam/core/playfield.py` \u2014 full rewrite with `PlayfieldBoundary` rename\n74\t- `src/aprilcam/core/__init__.py` \u2014 ensure `Playfield` export points to new class\n75\t\n76\t### Key Implementation Notes\n77\t\n78\t- Backward compatibility: `aprilcam.Playfield` still resolves; it now points\n79\t  to the new class. The old constructor signature is different \u2014 this is a\n80\t  breaking change, documented in the sprint.\n81\t- `calibration` param: if a file path, load `CameraCalibration` from JSON;\n82\t  if `None`, calibration not loaded (homography not available until `calibrate()`\n83\t  is called).\n84\t- `Tag` instances are cached by ID: `_tag_cache: dict[int, Tag]`.\n85\t- `stream()`: use `queue.Queue` filled by `on_frame` callback; `stop()` puts\n86\t  a sentinel; generator yields until sentinel received.\n87\t\n88\t### Testing Plan\n89\t\n90\t- Unit: `Playfield(VideoCamera(path), width_cm=100, height_cm=80)` constructs.\n91\t- Unit: `start()` starts pipeline; `stop()` stops it.\n92\t- Unit: `tags()` returns dict after pipeline processes frames.\n93\t- Unit: `tag(id)` returns `Tag` if seen, `None` if not.\n94\t- Unit: `stream()` generator terminates after `stop()`.\n95\t- Integration (system test in ticket 015): full pipeline with video.\n96\t\n97\t### Documentation Updates\n98\t\n99\t- Docstrings on `Playfield`, all public methods.\n100\t- Note the breaking change vs. old `Playfield` geometry class.\n101\t"
        }
      ]
    },
    "uuid": "0d428609-21e2-418b-aef1-011488a8af8b",
    "timestamp": "2026-04-11T22:04:43.495Z",
    "sourceToolAssistantUUID": "48db7812-afbf-407f-a555-32b3fe07e074",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "0d428609-21e2-418b-aef1-011488a8af8b",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01Rv3jJQ8VGZVVcQ3HYJzvZL",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/011-cli-live-loop-move-aprilcam-run-to-cli-live-cli-py.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "fd337f6d-6d4a-4493-8140-45bec0e7923e",
    "timestamp": "2026-04-11T22:04:43.886Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "fd337f6d-6d4a-4493-8140-45bec0e7923e",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01Rv3jJQ8VGZVVcQ3HYJzvZL",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '011'\n3\ttitle: 'CLI live loop: move AprilCam.run() to cli/live_cli.py'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-008\n7\tdepends-on:\n8\t  - \"004\"\n9\t  - \"010\"\n10\tgithub-issue: ''\n11\ttodo:\n12\t  - docs/clasi/todo/oop-refactoring.md\n13\t---\n14\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n15\t\n16\t# CLI live loop: move AprilCam.run() to cli/live_cli.py\n17\t\n18\t## Description\n19\t\n20\tMove the interactive detection loop from `AprilCam.run()` into `cli/live_cli.py`.\n21\t`live_cli.py` should use `Playfield` and `TagTableTUI` directly, removing all\n22\tdependency on `AprilCam` from the CLI layer.\n23\t\n24\tAfter this ticket, `aprilcam live` (or equivalent CLI command) works identically\n25\tfrom the user's perspective, but the implementation no longer touches `AprilCam`.\n26\t\n27\t## Acceptance Criteria\n28\t\n29\t- [ ] `cli/live_cli.py` contains the live detection loop using `Playfield` +\n30\t      `TagTableTUI` + `PlayfieldDisplay`.\n31\t- [ ] `AprilCam.run()` is removed or reduced to a one-line shim calling the\n32\t      CLI-layer function.\n33\t- [ ] `uv run aprilcam live` (or the equivalent subcommand) works end-to-end.\n34\t- [ ] TUI renders tag data during live camera operation.\n35\t- [ ] Keyboard interrupt (`Ctrl+C`) stops the pipeline cleanly.\n36\t- [ ] No import of `AprilCam` in `cli/live_cli.py`.\n37\t\n38\t## Implementation Plan\n39\t\n40\t### Approach\n41\t\n42\t1. Read the existing `AprilCam.run()` implementation in `aprilcam.py`.\n43\t2. Rewrite the equivalent logic in `cli/live_cli.py` using:\n44\t   - `Camera.find(pattern)` to open the camera.\n45\t   - `Playfield(camera, ...)` to set up the pipeline.\n46\t   - `TagTableTUI` for display.\n47\t   - `PlayfieldDisplay` for the OpenCV window (if applicable).\n48\t3. Wire to the CLI argument parser in `cli/__init__.py` or `init_cli.py`.\n49\t4. In `aprilcam.py`, `run()` becomes: raise `DeprecationWarning` or simply\n50\t   call the new CLI function.\n51\t\n52\t### Files to Modify\n53\t\n54\t- `src/aprilcam/cli/live_cli.py` \u2014 full rewrite using Playfield API\n55\t- `src/aprilcam/core/aprilcam.py` \u2014 remove/shim `run()`\n56\t- `src/aprilcam/cli/__init__.py` (or `init_cli.py`) \u2014 ensure routing unchanged\n57\t\n58\t### Key Implementation Notes\n59\t\n60\t- Argument parsing for camera pattern, calibration file, family, etc. stays in\n61\t  CLI layer.\n62\t- `PlayfieldDisplay` (OpenCV window) is wired to the `on_frame` callback.\n63\t- Signal handling: `try/finally` with `field.stop()` guarantees cleanup.\n64\t\n65\t### Testing Plan\n66\t\n67\t- Smoke: `uv run aprilcam --help` includes the live subcommand.\n68\t- Smoke: `uv run aprilcam live --help` shows expected options.\n69\t- (End-to-end live test requires hardware; deferred to manual verification.)\n70\t\n71\t### Documentation Updates\n72\t\n73\t- Update `AGENT_GUIDE.md` if it references the CLI command.\n74\t- Docstring on the new `live_cli` entry point function.\n75\t"
        }
      ]
    },
    "uuid": "12c9c211-8769-4fc9-b8d9-e2fc7bf0cc55",
    "timestamp": "2026-04-11T22:04:43.915Z",
    "sourceToolAssistantUUID": "fd337f6d-6d4a-4493-8140-45bec0e7923e",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "12c9c211-8769-4fc9-b8d9-e2fc7bf0cc55",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01SQwgdDgL4epXpyxT4VW54h",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/012-update-init-py-export-camera-tag-calibrate-delete-stream-py-shrink-aprilcam-shim.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "39de6c33-dc20-4d6d-b0a5-ceed34e37791",
    "timestamp": "2026-04-11T22:04:44.700Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "39de6c33-dc20-4d6d-b0a5-ceed34e37791",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01SQwgdDgL4epXpyxT4VW54h",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '012'\n3\ttitle: 'Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-001\n7\t  - SUC-006\n8\t  - SUC-009\n9\tdepends-on:\n10\t  - \"010\"\n11\t  - \"011\"\n12\tgithub-issue: ''\n13\ttodo:\n14\t  - docs/clasi/todo/oop-refactoring.md\n15\t---\n16\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n17\t\n18\t# Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim\n19\t\n20\t## Description\n21\t\n22\tUpdate `src/aprilcam/__init__.py` to export the new public API (`Camera`, `Tag`,\n23\t`calibrate`, `Playfield`). Delete `stream.py` and replace its top-level exports\n24\twith re-exports pointing to `Playfield`-based implementations. Shrink `AprilCam`\n25\tin `core/aprilcam.py` to a minimal compatibility shim.\n26\t\n27\tAfter this ticket, `import aprilcam; aprilcam.Camera`, `aprilcam.Tag`,\n28\t`aprilcam.Playfield`, `aprilcam.calibrate` all resolve. Legacy names\n29\t`detect_tags`, `detect_objects`, `AprilCam` remain available but may emit\n30\t`DeprecationWarning`.\n31\t\n32\t## Acceptance Criteria\n33\t\n34\t- [ ] `import aprilcam; aprilcam.Camera` resolves to `camera.Camera`.\n35\t- [ ] `import aprilcam; aprilcam.Tag` resolves to `core.tag.Tag`.\n36\t- [ ] `import aprilcam; aprilcam.calibrate` resolves to `calibration.calibrate`.\n37\t- [ ] `import aprilcam; aprilcam.Playfield` resolves to new `core.playfield.Playfield`.\n38\t- [ ] `from aprilcam import detect_tags` still works (compatibility re-export).\n39\t- [ ] `from aprilcam import detect_objects` still works (compatibility re-export).\n40\t- [ ] `from aprilcam import AprilCam` still works (shim, deprecated).\n41\t- [ ] `stream.py` is deleted from the source tree.\n42\t- [ ] `core/aprilcam.py` is reduced: all logic extracted; only shim wrappers remain\n43\t      that delegate to the new classes.\n44\t- [ ] `uv run pytest` passes (no import errors from removals).\n45\t\n46\t## Implementation Plan\n47\t\n48\t### Approach\n49\t\n50\t1. Update `__init__.py`: add `Camera`, `Tag`, `calibrate` imports; update\n51\t   `Playfield` import; add `__all__` entries.\n52\t2. For `detect_tags`/`detect_objects` compatibility, write thin wrappers in\n53\t   `__init__.py` that open a `Playfield` and stream results \u2014 or point to\n54\t   `stream.py`'s replacement function.\n55\t3. Delete `stream.py`.\n56\t4. `AprilCam` in `core/aprilcam.py`: strip all extracted logic; keep constructor\n57\t   signature for backward compat; delegate methods to new classes.\n58\t\n59\t### Files to Delete\n60\t\n61\t- `src/aprilcam/stream.py`\n62\t\n63\t### Files to Modify\n64\t\n65\t- `src/aprilcam/__init__.py` \u2014 full update\n66\t- `src/aprilcam/core/aprilcam.py` \u2014 strip to shim\n67\t\n68\t### Key Implementation Notes\n69\t\n70\t- The compatibility `detect_tags()` in `__init__.py` needs to match the old\n71\t  generator signature: `detect_tags(camera, ...) -> Generator[list[TagRecord]]`.\n72\t  Implement as a thin wrapper over `Playfield.stream()`.\n73\t- `AprilCam` shim: keep `__init__` signature; in body, construct `TagDetector`,\n74\t  `OpticalFlowTracker`, `DetectionPipeline`; delegate `process_frame()`,\n75\t  `detect_apriltags()`, etc. to those objects.\n76\t- Verify all CLI and MCP server paths still import cleanly after deletion.\n77\t\n78\t### Testing Plan\n79\t\n80\t- Smoke: `import aprilcam` succeeds.\n81\t- Smoke: `aprilcam.Camera`, `aprilcam.Tag`, `aprilcam.Playfield`,\n82\t  `aprilcam.calibrate` all resolve.\n83\t- Smoke: `from aprilcam import detect_tags, detect_objects, AprilCam` succeeds.\n84\t- Smoke: `uv run aprilcam --help` works.\n85\t- Unit: `uv run pytest` passes with no import errors.\n86\t\n87\t### Documentation Updates\n88\t\n89\t- `AGENT_GUIDE.md`: update API section to show new exports.\n90\t- `__init__.py`: update module docstring to list public API.\n91\t"
        }
      ]
    },
    "uuid": "6aed098b-771f-4ad7-845e-957fc796fd11",
    "timestamp": "2026-04-11T22:04:44.737Z",
    "sourceToolAssistantUUID": "39de6c33-dc20-4d6d-b0a5-ceed34e37791",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "6aed098b-771f-4ad7-845e-957fc796fd11",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01F3esu6exSo7yb3JPiDbG8L",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/013-smoke-tests-import-and-construction-sanity-checks.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "da008c13-30a9-428d-aaf0-8f27ffa6821b",
    "timestamp": "2026-04-11T22:04:45.223Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "da008c13-30a9-428d-aaf0-8f27ffa6821b",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01F3esu6exSo7yb3JPiDbG8L",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '013'\n3\ttitle: 'Smoke tests: import and construction sanity checks'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-010\n7\tdepends-on:\n8\t  - \"012\"\n9\tgithub-issue: ''\n10\ttodo:\n11\t  - docs/clasi/todo/video-based-test-system.md\n12\t---\n13\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n14\t\n15\t# Smoke tests: import and construction sanity checks\n16\t\n17\t## Description\n18\t\n19\tCreate `tests/smoke/` with fast, hardware-free sanity checks. Smoke tests\n20\tverify that the package imports cleanly, all new public classes are accessible,\n21\tand basic construction does not raise. They must run in under 5 seconds total.\n22\t\n23\tThese tests are the first line of defense against import errors, missing\n24\tmodules, and broken `__init__.py` wiring after the refactor.\n25\t\n26\t## Acceptance Criteria\n27\t\n28\t- [ ] `tests/smoke/` directory exists with `__init__.py` and at least one test file.\n29\t- [ ] `tests/smoke/test_imports.py` verifies:\n30\t      - `import aprilcam` succeeds.\n31\t      - `aprilcam.Camera` is accessible.\n32\t      - `aprilcam.Tag` is accessible.\n33\t      - `aprilcam.Playfield` is accessible.\n34\t      - `aprilcam.calibrate` is callable.\n35\t      - `aprilcam.VideoCamera` is accessible (via `aprilcam.camera` or top-level).\n36\t      - `from aprilcam import TagRecord, AprilCam, detect_tags` succeeds (legacy).\n37\t- [ ] `tests/smoke/test_construction.py` verifies:\n38\t      - `TagDetector()` constructs without error.\n39\t      - `DetectorConfig()` constructs with defaults.\n40\t      - `VelocityEstimator()` constructs without error.\n41\t      - `OpticalFlowTracker()` constructs without error.\n42\t      - `TagTableTUI()` constructs without error.\n43\t      - `VideoCamera(\"tests/movies/bright-gsc.mov\")` constructs and is open.\n44\t- [ ] All smoke tests pass with `uv run pytest tests/smoke/`.\n45\t- [ ] Total smoke test runtime under 5 seconds.\n46\t- [ ] No hardware required to run smoke tests.\n47\t\n48\t## Implementation Plan\n49\t\n50\t### Approach\n51\t\n52\tWrite minimal `pytest` tests \u2014 no fixtures, no mocks. Each test is a single\n53\t`assert` or `import` statement. The point is to catch wiring errors quickly.\n54\t\n55\t### Files to Create\n56\t\n57\t- `tests/smoke/__init__.py`\n58\t- `tests/smoke/test_imports.py`\n59\t- `tests/smoke/test_construction.py`\n60\t\n61\t### Files to Modify\n62\t\n63\t- `tests/conftest.py` (create if needed) \u2014 add path to `tests/movies/` fixture.\n64\t\n65\t### Key Implementation Notes\n66\t\n67\t- Use `pytest.importorskip` or `importlib` for optional dependencies.\n68\t- `VideoCamera` construction test needs the path to `tests/movies/bright-gsc.mov`;\n69\t  use `pathlib.Path(__file__).parent.parent / \"movies\" / \"bright-gsc.mov\"`.\n70\t- Do not call `read()` in smoke tests \u2014 construction only.\n71\t- If Rich is not installed, `TagTableTUI` construction should be `pytest.skip`-ed\n72\t  rather than fail.\n73\t\n74\t### Testing Plan\n75\t\n76\t- These tests are themselves the test plan.\n77\t- Verification: `uv run pytest tests/smoke/ -v` all green.\n78\t\n79\t### Documentation Updates\n80\t\n81\t- Add comment at top of each smoke test file explaining its purpose.\n82\t"
        }
      ]
    },
    "uuid": "75229f9c-a26a-4692-baf9-f5a06bc59cdc",
    "timestamp": "2026-04-11T22:04:45.251Z",
    "sourceToolAssistantUUID": "da008c13-30a9-428d-aaf0-8f27ffa6821b",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "75229f9c-a26a-4692-baf9-f5a06bc59cdc",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01MhgsjRycxbgyNhB5NQ2dpY",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/014-unit-tests-per-class-tests-for-all-new-oop-classes.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": null,
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 1736,
          "ephemeral_1h_input_tokens": 0
        },
        "output_tokens": 1,
        "service_tier": "standard",
        "inference_geo": "not_available"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "b20ac298-8fc5-4240-b2c0-1389adda1d55",
    "timestamp": "2026-04-11T22:04:45.780Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "b20ac298-8fc5-4240-b2c0-1389adda1d55",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01MhgsjRycxbgyNhB5NQ2dpY",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '014'\n3\ttitle: 'Unit tests: per-class tests for all new OOP classes'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-010\n7\tdepends-on:\n8\t  - \"013\"\n9\tgithub-issue: ''\n10\ttodo:\n11\t  - docs/clasi/todo/video-based-test-system.md\n12\t---\n13\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n14\t\n15\t# Unit tests: per-class tests for all new OOP classes\n16\t\n17\t## Description\n18\t\n19\tCreate `tests/unit/` with one test file per major class. Unit tests verify\n20\tconstructor, methods, properties, and edge cases in isolation. Hardware is not\n21\trequired; `VideoCamera` or synthetic frames are used where needed.\n22\t\n23\tThese tests form the regression safety net for the OOP refactor.\n24\t\n25\t## Acceptance Criteria\n26\t\n27\t- [ ] `tests/unit/` directory exists with `__init__.py`.\n28\t- [ ] `tests/unit/test_camera.py` \u2014 tests `Camera.list()`, `Camera.find()`,\n29\t      `is_open`, `close()`, context manager.\n30\t- [ ] `tests/unit/test_video_camera.py` \u2014 tests `VideoCamera` construction,\n31\t      sequential `read()`, EOF returns `None`, context manager.\n32\t- [ ] `tests/unit/test_tag_detector.py` \u2014 tests `DetectorConfig` defaults,\n33\t      `TagDetector()` construction, `detect()` on blank frame returns `[]`,\n34\t      stateless (same result twice).\n35\t- [ ] `tests/unit/test_optical_flow_tracker.py` \u2014 tests `update()` with\n36\t      detections, `update()` without (propagate), `reset()`.\n37\t- [ ] `tests/unit/test_velocity_estimator.py` \u2014 tests first-call zero, constant\n38\t      velocity, deadband suppression, `predict_position()`, `reset()`.\n39\t- [ ] `tests/unit/test_detection_pipeline.py` \u2014 tests `start()`, `stop()`,\n40\t      `is_running`, `on_frame` callback firing, ring buffer populated; uses\n41\t      `VideoCamera`.\n42\t- [ ] `tests/unit/test_tag.py` \u2014 tests properties when snapshot is None,\n43\t      properties after `update()` with seeded ring buffer, `position_at(t)`.\n44\t- [ ] `tests/unit/test_tag_record.py` \u2014 tests `TagRecord.to_dict()`,\n45\t      `TagRecord.estimate(t)` extrapolation.\n46\t- [ ] `tests/unit/test_ring_buffer.py` \u2014 tests write, read latest, history,\n47\t      capacity wrapping.\n48\t- [ ] All unit tests pass with `uv run pytest tests/unit/`.\n49\t- [ ] No test requires a connected camera.\n50\t\n51\t## Implementation Plan\n52\t\n53\t### Approach\n54\t\n55\tWrite focused pytest functions. Use `VideoCamera` for pipeline tests. Use\n56\tsynthetic NumPy frames for detector and tracker tests. Mock `DetectionPipeline`\n57\tfor `Tag` tests (inject a fake `RingBuffer` with seeded records).\n58\t\n59\t### Files to Create\n60\t\n61\t- `tests/unit/__init__.py`\n62\t- `tests/unit/test_camera.py`\n63\t- `tests/unit/test_video_camera.py`\n64\t- `tests/unit/test_tag_detector.py`\n65\t- `tests/unit/test_optical_flow_tracker.py`\n66\t- `tests/unit/test_velocity_estimator.py`\n67\t- `tests/unit/test_detection_pipeline.py`\n68\t- `tests/unit/test_tag.py`\n69\t- `tests/unit/test_tag_record.py`\n70\t- `tests/unit/test_ring_buffer.py`\n71\t\n72\t### Key Implementation Notes\n73\t\n74\t- `conftest.py` at `tests/` level: provide `video_path(name)` fixture returning\n75\t  `Path(\"tests/movies/<name>\")`.\n76\t- `TagDetector` test: `np.ones((480, 640, 3), dtype=np.uint8) * 255` is a\n77\t  blank white frame \u2014 no tags will be detected.\n78\t- `OpticalFlowTracker` test: create two identical grayscale frames (zero shift)\n79\t  and verify `update()` returns same number of detections.\n80\t- `DetectionPipeline` test: use `bright-gsc.mov` via `VideoCamera`; call\n81\t  `start()`; wait for pipeline to process all frames; call `stop()`; assert\n82\t  `ring_buffer` is non-empty.\n83\t- `Tag` test: instantiate with a stub `DetectionPipeline` whose `ring_buffer`\n84\t  contains a seeded `TagRecord`; call `update()`; assert properties match.\n85\t\n86\t### Testing Plan\n87\t\n88\t- Verification: `uv run pytest tests/unit/ -v` all green.\n89\t- Aim for under 30 seconds total (video processing in pipeline test may take a\n90\t  few seconds).\n91\t\n92\t### Documentation Updates\n93\t\n94\t- `tests/unit/conftest.py` if created: docstring explaining shared fixtures.\n95\t"
        }
      ]
    },
    "uuid": "b4177cf1-a2dc-4c4b-a036-6264abb5f28b",
    "timestamp": "2026-04-11T22:04:45.813Z",
    "sourceToolAssistantUUID": "b20ac298-8fc5-4240-b2c0-1389adda1d55",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "b4177cf1-a2dc-4c4b-a036-6264abb5f28b",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01W8BFs8rv5JrFFeAw3w9rz5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01C2kCo2iW9wzBCHeX56Eqdt",
          "name": "Read",
          "input": {
            "file_path": "/Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/014-oop-refactoring-and-video-based-test-system/tickets/015-system-tests-video-driven-pipeline-tests-using-tests-movies-mov.md"
          },
          "caller": {
            "type": "direct"
          }
        }
      ],
      "stop_reason": "tool_use",
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 1736,
        "cache_read_input_tokens": 34338,
        "output_tokens": 1652,
        "server_tool_use": {
          "web_search_requests": 0,
          "web_fetch_requests": 0
        },
        "service_tier": "standard",
        "cache_creation": {
          "ephemeral_1h_input_tokens": 0,
          "ephemeral_5m_input_tokens": 1736
        },
        "inference_geo": "",
        "iterations": [
          {
            "input_tokens": 5,
            "output_tokens": 1652,
            "cache_read_input_tokens": 34338,
            "cache_creation_input_tokens": 1736,
            "cache_creation": {
              "ephemeral_5m_input_tokens": 1736,
              "ephemeral_1h_input_tokens": 0
            },
            "type": "message"
          }
        ],
        "speed": "standard"
      }
    },
    "requestId": "req_011CZxjxFW4iGQJL4RvxE5b4",
    "type": "assistant",
    "uuid": "290e5346-9385-4c0b-a8be-13d640e96bc3",
    "timestamp": "2026-04-11T22:04:46.221Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "290e5346-9385-4c0b-a8be-13d640e96bc3",
    "isSidechain": true,
    "promptId": "8eae9b5c-add5-4dbb-b462-e1da10044f90",
    "agentId": "a01634c1d6a193be6",
    "type": "user",
    "message": {
      "role": "user",
      "content": [
        {
          "tool_use_id": "toolu_01C2kCo2iW9wzBCHeX56Eqdt",
          "type": "tool_result",
          "content": "1\t---\n2\tid: '015'\n3\ttitle: 'System tests: video-driven pipeline tests using tests/movies/*.mov'\n4\tstatus: todo\n5\tuse-cases:\n6\t  - SUC-002\n7\t  - SUC-010\n8\tdepends-on:\n9\t  - \"014\"\n10\tgithub-issue: ''\n11\ttodo:\n12\t  - docs/clasi/todo/oop-refactoring.md\n13\t  - docs/clasi/todo/video-based-test-system.md\n14\t---\n15\t<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->\n16\t\n17\t# System tests: video-driven pipeline tests using tests/movies/*.mov\n18\t\n19\t## Description\n20\t\n21\tCreate `tests/system/` with full pipeline tests driven by the four test videos\n22\tin `tests/movies/`. Each test constructs a `VideoCamera`, builds a `Playfield`,\n23\truns the `DetectionPipeline` to completion, and asserts on detection results.\n24\t\n25\tSystem tests verify end-to-end behavior: tags are detected, positions are\n26\tplausible, velocity is computed, `estimate()` extrapolation is consistent.\n27\tNo camera hardware required.\n28\t\n29\t## Acceptance Criteria\n30\t\n31\t- [ ] `tests/system/` directory exists with `__init__.py`.\n32\t- [ ] `tests/system/test_detection_pipeline_videos.py` exists with parametrized\n33\t      tests for all four video files.\n34\t- [ ] For each of `bright-gsc.mov`, `bright-ov9782.mov`, `dim-gsc.mov`,\n35\t      `dim-ov9782.mov`:\n36\t      - [ ] Pipeline runs to completion (EOF) without raising.\n37\t      - [ ] At least one tag is detected in the video.\n38\t      - [ ] Detected tag `center_px` values are within frame dimensions.\n39\t      - [ ] If velocity is non-zero, `estimate(t)` returns a position within\n40\t            2\u00d7 the frame dimensions (sanity check on extrapolation).\n41\t- [ ] Bright lighting tests (`bright-*.mov`) detect at least one tag.\n42\t- [ ] Dim lighting tests (`dim-*.mov`) detect at least one tag (may require\n43\t      relaxed assertion or `pytest.mark.xfail` if dim videos are genuinely very dark).\n44\t- [ ] Both camera types (`*-gsc.mov`, `*-ov9782.mov`) produce detections.\n45\t- [ ] All system tests pass with `uv run pytest tests/system/`.\n46\t- [ ] No test requires a connected camera.\n47\t\n48\t## Implementation Plan\n49\t\n50\t### Approach\n51\t\n52\t1. Create `tests/system/__init__.py`.\n53\t2. Create `tests/system/test_detection_pipeline_videos.py`.\n54\t3. Use `@pytest.mark.parametrize` over the four video filenames.\n55\t4. Each parametrized test:\n56\t   a. Construct `VideoCamera(path)`.\n57\t   b. Construct `Playfield(video_cam, width_cm=100, height_cm=80)`.\n58\t   c. Call `field.start()`.\n59\t   d. Collect frames via `field.stream()` until the generator terminates (EOF).\n60\t   e. Call `field.stop()`.\n61\t   f. Assert detection results.\n62\t5. Use a shared `conftest.py` fixture for the movies directory path.\n63\t\n64\t### Files to Create\n65\t\n66\t- `tests/system/__init__.py`\n67\t- `tests/system/test_detection_pipeline_videos.py`\n68\t- `tests/conftest.py` (if not already created in ticket 013/014) \u2014 movies path fixture.\n69\t\n70\t### Key Implementation Notes\n71\t\n72\t- `field.stream()` terminates when `VideoCamera.read()` returns `None` (EOF);\n73\t  the generator loop must check for this.\n74\t- Collect all `Tag` objects across all frames into a set by ID.\n75\t- For dim videos: if detection genuinely fails (dim videos may be very dark),\n76\t  mark with `@pytest.mark.xfail(strict=False, reason=\"dim lighting may not detect\")`.\n77\t  Update once actual detection results are known.\n78\t- Frame count: videos may have hundreds of frames; the test should complete in\n79\t  under 60 seconds total for all four.\n80\t- Detection assertions: assert `len(all_detected_ids) >= 1` \u2014 at least one tag\n81\t  seen across all frames of the video.\n82\t\n83\t### Testing Plan\n84\t\n85\t- Verification: `uv run pytest tests/system/ -v` green (or xfail for dim if expected).\n86\t- Run time target: under 60 seconds total for all four videos.\n87\t\n88\t### Documentation Updates\n89\t\n90\t- Comment in test file explaining the parametrize strategy and the dim-lighting\n91\t  xfail rationale.\n92\t"
        }
      ]
    },
    "uuid": "fc05a3b4-1c74-40d7-905e-2b381b75d642",
    "timestamp": "2026-04-11T22:04:46.252Z",
    "sourceToolAssistantUUID": "290e5346-9385-4c0b-a8be-13d640e96bc3",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  },
  {
    "parentUuid": "fc05a3b4-1c74-40d7-905e-2b381b75d642",
    "isSidechain": true,
    "agentId": "a01634c1d6a193be6",
    "message": {
      "model": "claude-haiku-4-5-20251001",
      "id": "msg_01CzE3F4wkEaXWAnsPNEXnQq",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "## Ticket Summary for Sprint 014: OOP Refactoring and Video-Based Test System\n\n### **Ticket 001: Camera class: device discovery and VideoCapture wrapper**\n- **Key Acceptance Criteria:**\n  - `Camera.list()` enumerates available cameras\n  - `Camera.find(pattern)` finds camera by substring match\n  - Lazy open on first `read()` call (not in `__init__`)\n  - Properties: `name`, `index`, `resolution`, `is_open`\n  - Context manager support\n- **Files to Create:** `src/aprilcam/camera/camera.py`\n- **Files to Modify:** `src/aprilcam/camera/__init__.py`\n- **Key Details:** Wraps `cv.VideoCapture`; lifts camera management from `AprilCam.__init__`; delegates discovery to `camutil`\n\n---\n\n### **Ticket 002: TagDetector: pure stateless detection engine**\n- **Key Acceptance Criteria:**\n  - `DetectorConfig` dataclass with tuning parameters\n  - `TagDetector.detect(frame_bgr)` returns list of `Detection` objects\n  - Stateless: calling twice on same frame returns equivalent results\n  - Removes duplicate functions from `aprilcam.py`\n- **Files to Create:** `src/aprilcam/core/detector.py`\n- **Files to Modify:** `src/aprilcam/core/aprilcam.py`, `src/aprilcam/core/__init__.py`\n- **Key Details:** Consolidates `_build_detectors()`, `detect_apriltags()`, preprocessing logic; `Detection` dataclass with `id`, `center`, `corners`, `family`\n\n---\n\n### **Ticket 003: VelocityEstimator: per-tag EMA velocity with deadband**\n- **Key Acceptance Criteria:**\n  - `update(position, timestamp)` returns `(velocity_vec, speed)`\n  - EMA smoothing: `v_ema = alpha * v_raw + (1 - alpha) * v_prev`\n  - Deadband: suppress velocity if raw speed below threshold\n  - `predict_position(t)` extrapolates linearly\n  - `reset()` clears state\n- **Files to Create:** `src/aprilcam/core/motion.py`\n- **Files to Modify:** `src/aprilcam/core/playfield.py`, `src/aprilcam/core/__init__.py`\n- **Key Details:** One instance per tracked tag; pixel-space only (world transform stays in `Playfield`)\n\n---\n\n### **Ticket 004: TagTableTUI: extract Rich TUI dashboard from AprilCam**\n- **Key Acceptance Criteria:**\n  - `TagTableTUI` has no dependency on `AprilCam`\n  - `update(tags: list[dict])` renders live table with columns: `id`, `cx`, `cy`, `wx`, `wy`, `speed`, `heading`, `orientation`\n  - Context manager support\n  - EMA smoothing for display only (not exported to business logic)\n- **Files to Create:** `src/aprilcam/ui/tui.py`\n- **Files to Modify:** `src/aprilcam/core/aprilcam.py`, `src/aprilcam/ui/__init__.py`\n- **Key Details:** Extracts `_build_tui_layout`, `_print_tui`, `_stop_tui`, `_ema_smooth` from `AprilCam`\n\n---\n\n### **Ticket 005: Calibration split: extract calibrate() and types from homography.py**\n- **Key Acceptance Criteria:**\n  - `calibration/calibration.py` contains `calibrate()`, `CameraCalibration`, `FieldSpec`\n  - `homography.py` retains only pure math functions\n  - `FieldSpec` has `width_cm`, `height_cm` and unit conversion helpers\n  - All callers updated to import from `calibration.calibration`\n- **Files to Create:** `src/aprilcam/calibration/calibration.py`\n- **Files to Modify:** `src/aprilcam/calibration/homography.py`, `src/aprilcam/calibration/__init__.py`\n- **Key Details:** Split: `calibration.py` owns workflow; `homography.py` owns matrix math\n\n---\n\n### **Ticket 006: OpticalFlowTracker: LK tracking with 4-corner motion decomposition**\n- **Key Acceptance Criteria:**\n  - `update(gray, detections)` returns updated `Detection` list\n  - With `detections`: reset tracked points\n  - Without `detections`: propagate via Lucas-Kanade flow\n  - Output includes motion fields: translation, rotation rate, scale delta\n  - Fallback to center-only if LK fails\n- **Files to Create:** `src/aprilcam/core/tracker.py`\n- **Files to Modify:** `src/aprilcam/core/detector.py`, `src/aprilcam/core/aprilcam.py`, `src/aprilcam/core/__init__.py`\n- **Key Details:** Absorbs `AprilCam.lk_track()` and detect-or-track branching; LK params: `winSize=(21,21)`, `maxLevel=3`\n\n---\n\n### **Ticket 007: VideoCamera: Camera subclass reading frames from a video file**\n- **Key Acceptance Criteria:**\n  - `VideoCamera(path)` subclasses `Camera`; reads frames from file\n  - `read()` returns frames sequentially; `None` at EOF\n  - `is_open` tracks EOF state\n  - `name` returns filename stem; `index` returns `-1` (sentinel)\n  - Context manager support\n- **Files to Create:** `src/aprilcam/camera/video_camera.py`\n- **Files to Modify:** `src/aprilcam/camera/__init__.py`\n- **Key Details:** Opens video file directly; bypasses `Camera._open()`; enables video-based testing without hardware\n\n---\n\n### **Ticket 008: DetectionPipeline: background thread orchestrating detect-track-velocity-buffer**\n- **Key Acceptance Criteria:**\n  - `start()` starts background thread; idempotent\n  - `stop()` stops thread; idempotent\n  - `is_running` property\n  - `on_frame(callback)` registers callback invoked after each frame\n  - Loop: read \u2192 grayscale \u2192 detect (every N frames) \u2192 track \u2192 velocity \u2192 buffer write\n  - Per-tag `VelocityEstimator` instances recycled across frames\n  - Thread-safe ring buffer\n- **Files to Create:** `src/aprilcam/core/pipeline.py`\n- **Files to Modify:** `src/aprilcam/core/__init__.py`, optionally `src/aprilcam/core/aprilcam.py`\n- **Key Details:** Evolves from existing `DetectionLoop`; daemon thread; stop signal via `threading.Event`\n\n---\n\n### **Ticket 009: Tag class: live tag handle with flat properties and update/position_at**\n- **Key Acceptance Criteria:**\n  - `Tag(tag_id, pipeline)` constructor\n  - Flat properties: `id`, `cx`, `cy`, `wx`, `wy`, `orientation`, `velocity`, `speed`, `heading`, `rotation_rate`, `timestamp`, `age`, `is_visible`\n  - `update()` atomically replaces snapshot from ring buffer; returns self\n  - `position_at(t)` extrapolates pixel position\n  - `to_dict()` returns flat dict\n- **Files to Create:** `src/aprilcam/core/tag.py`\n- **Files to Modify:** `src/aprilcam/core/__init__.py`\n- **Key Details:** Read-only view of ring buffer; thread-safe snapshot replacement; returns sensible defaults when tag not seen\n\n---\n\n### **Ticket 010: Playfield rewrite: primary user-facing object owning DetectionPipeline**\n- **Key Acceptance Criteria:**\n  - `Playfield(camera, width_cm, height_cm, family, calibration)` constructor\n  - `start()`, `stop()` control pipeline\n  - `tags()` returns dict[int, Tag]; `tag(id)` returns Tag | None\n  - `stream()` generator yielding list[Tag] per frame\n  - `on_frame(callback)` registers push callback\n  - `calibrate()`, `pixel_to_world()`, `world_to_pixel()`, `deskew()`\n  - Auto-loads calibration from JSON if file provided\n  - Old `Playfield` geometry logic renamed to `PlayfieldBoundary` (private)\n- **Files to Modify:** `src/aprilcam/core/playfield.py`, `src/aprilcam/core/__init__.py`\n- **Key Details:** New primary user-facing object; owns `DetectionPipeline`, `TagDetector`, `OpticalFlowTracker`; breaking API change from old geometry class\n\n---\n\n### **Ticket 011: CLI live loop: move AprilCam.run() to cli/live_cli.py**\n- **Key Acceptance Criteria:**\n  - `cli/live_cli.py` uses `Playfield` + `TagTableTUI` directly\n  - No dependency on `AprilCam` in CLI layer\n  - `aprilcam live` subcommand works end-to-end\n  - TUI renders during operation\n  - Keyboard interrupt (`Ctrl+C`) stops cleanly\n- **Files to Modify:** `src/aprilcam/cli/live_cli.py`, `src/aprilcam/core/aprilcam.py`, `src/aprilcam/cli/__init__.py`\n- **Key Details:** Rewrites CLI using new high-level API; removes dependency on `AprilCam`; argument parsing stays in CLI layer\n\n---\n\n### **Ticket 012: Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim**\n- **Key Acceptance Criteria:**\n  - `aprilcam.Camera`, `aprilcam.Tag`, `aprilcam.Playfield`, `aprilcam.calibrate` all resolve\n  - Legacy names (`detect_tags`, `detect_objects`, `AprilCam`) still available (compatibility re-exports)\n  - `stream.py` deleted\n  - `AprilCam` reduced to minimal shim\n  - `uv run pytest` passes\n- **Files to Delete:** `src/aprilcam/stream.py`\n- **Files to Modify:** `src/aprilcam/__init__.py`, `src/aprilcam/core/aprilcam.py`\n- **Key Details:** Compatibility wrappers for old API; `detect_tags` wraps `Playfield.stream()`; `AprilCam` delegates to new classes\n\n---\n\n### **Ticket 013: Smoke tests: import and construction sanity checks**\n- **Key Acceptance Criteria:**\n  - `tests/smoke/test_imports.py` verifies all new classes import cleanly\n  - `tests/smoke/test_construction.py` verifies basic construction (no hardware)\n  - All smoke tests complete in under 5 seconds\n  - Verify imports: `Camera`, `Tag`, `Playfield`, `calibrate`, `VideoCamera`, legacy (`detect_tags`, `AprilCam`)\n  - Verify construction: `TagDetector()`, `DetectorConfig()`, `VelocityEstimator()`, `OpticalFlowTracker()`, `TagTableTUI()`, `VideoCamera(path)`\n- **Files to Create:** `tests/smoke/__init__.py`, `tests/smoke/test_imports.py`, `tests/smoke/test_construction.py`\n- **Files to Modify:** `tests/conftest.py` (if needed)\n- **Key Details:** Fast hardware-free sanity checks; `pytest.skip` for Rich if not installed; first defense against import errors\n\n---\n\n### **Ticket 014: Unit tests: per-class tests for all new OOP classes**\n- **Key Acceptance Criteria:**\n  - Per-class test files for: `Camera`, `VideoCamera`, `TagDetector`, `OpticalFlowTracker`, `VelocityEstimator`, `DetectionPipeline`, `Tag`, `TagRecord`, `RingBuffer`\n  - Test key methods: list, find, read, detect, update, reset, on_frame\n  - No hardware required\n  - All tests under 30 seconds total\n- **Files to Create:** `tests/unit/__init__.py`, `tests/unit/test_*.py` (9 test files)\n- **Key Details:** Use `VideoCamera` for pipeline tests; synthetic frames for detector/tracker; mock pipeline for `Tag` tests\n\n---\n\n### **Ticket 015: System tests: video-driven pipeline tests using tests/movies/*.mov**\n- **Key Acceptance Criteria:**\n  - `tests/system/test_detection_pipeline_videos.py` with parametrized tests\n  - Test all four videos: `bright-gsc.mov`, `bright-ov9782.mov`, `dim-gsc.mov`, `dim-ov9782.mov`\n  - Pipeline runs to EOF without error\n  - At least one tag detected per video (or xfail for dim videos)\n  - Detected positions within frame dimensions\n  - Extrapolation (`estimate(t)`) sanity check\n- **Files to Create:** `tests/system/__init__.py`, `tests/system/test_detection_pipeline_videos.py`\n- **Key Details:** Full end-to-end pipeline tests; no hardware required; under 60 seconds total for all videos\n\n---\n\n## Summary\n\nThe 15 tickets are sequenced to:\n1. **Tickets 001\u2013007:** Core class extraction (Camera, TagDetector, VelocityEstimator, TUI, Calibration, OpticalFlowTracker, VideoCamera)\n2. **Tickets 008\u2013010:** Pipeline orchestration (DetectionPipeline, Tag, Playfield rewrite)\n3. **Tickets 011\u2013012:** CLI migration and API stabilization\n4. **Tickets 013\u2013015:** Testing (smoke, unit, system)\n\nEach ticket has clear dependencies, enabling parallelizable work. The refactor transforms a monolithic `AprilCam` into composable, testable classes with a clean public API."
        }
      ],
      "stop_reason": "end_turn",
      "stop_sequence": null,
      "stop_details": null,
      "usage": {
        "input_tokens": 7,
        "cache_creation_input_tokens": 23537,
        "cache_read_input_tokens": 33818,
        "output_tokens": 3496,
        "server_tool_use": {
          "web_search_requests": 0,
          "web_fetch_requests": 0
        },
        "service_tier": "standard",
        "cache_creation": {
          "ephemeral_1h_input_tokens": 0,
          "ephemeral_5m_input_tokens": 23537
        },
        "inference_geo": "",
        "iterations": [
          {
            "input_tokens": 7,
            "output_tokens": 3496,
            "cache_read_input_tokens": 33818,
            "cache_creation_input_tokens": 23537,
            "cache_creation": {
              "ephemeral_5m_input_tokens": 23537,
              "ephemeral_1h_input_tokens": 0
            },
            "type": "message"
          }
        ],
        "speed": "standard"
      }
    },
    "requestId": "req_011CZxjy16fSEG2dm8MBXqCN",
    "type": "assistant",
    "uuid": "463dc9a0-b137-43cb-bbfd-189ee082f5f9",
    "timestamp": "2026-04-11T22:05:17.556Z",
    "userType": "external",
    "entrypoint": "claude-vscode",
    "cwd": "/Volumes/Proj/proj/RobotProjects/AprilTags",
    "sessionId": "8e6d89d0-3801-4cda-b844-7f6a4a37d724",
    "version": "2.1.101",
    "gitBranch": "sprint/014-oop-refactoring-and-video-based-test-system",
    "slug": "reactive-knitting-moore"
  }
]
```
