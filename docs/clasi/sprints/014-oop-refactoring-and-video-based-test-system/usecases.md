---
status: draft
---

# Sprint 014 Use Cases

## SUC-001: Discover and Open a Camera by Name Pattern

- **Actor**: Library consumer (Python code or AI agent)
- **Preconditions**: A USB camera is attached to the host machine.
- **Main Flow**:
  1. Consumer calls `Camera.list()` to enumerate available cameras.
  2. Consumer calls `Camera.find("Brio")` to find a camera by name pattern.
  3. `Camera` returns an object with `name`, `index`, `resolution` properties.
  4. Consumer calls `camera.read()` to obtain a BGR frame.
  5. Consumer calls `camera.close()` or uses camera as a context manager.
- **Postconditions**: Frame captured without managing VideoCapture directly.
- **Acceptance Criteria**:
  - [ ] `Camera.list()` returns a list of `Camera` objects.
  - [ ] `Camera.find(pattern)` raises `CameraNotFoundError` if no match.
  - [ ] `camera.read()` returns a BGR numpy array.
  - [ ] Context manager (`with Camera.find(...) as cam`) closes on exit.

---

## SUC-002: Run Full Tag Detection Pipeline Against a Video File

- **Actor**: Test suite / developer
- **Preconditions**: A `.mov` test video exists in `tests/movies/`.
- **Main Flow**:
  1. Developer constructs `VideoCamera("tests/movies/bright-gsc.mov")`.
  2. Developer constructs `Playfield(video_camera, ...)` and calls `start()`.
  3. `DetectionPipeline` reads frames from `VideoCamera` and runs detection.
  4. Developer calls `playfield.tags()` or consumes `playfield.stream()`.
  5. `playfield.stop()` joins the background thread.
- **Postconditions**: Tag detections returned from a recording; no hardware needed.
- **Acceptance Criteria**:
  - [ ] `VideoCamera` reads all frames from the `.mov` file sequentially.
  - [ ] `VideoCamera` raises `StopIteration` (or `read()` returns None) at EOF.
  - [ ] Tags detected in the bright-gsc video match expected IDs.

---

## SUC-003: Detect AprilTags in a Single Frame (Pure Detection)

- **Actor**: Library consumer
- **Preconditions**: A BGR frame is available.
- **Main Flow**:
  1. Consumer constructs `TagDetector(family="tag36h11")`.
  2. Consumer calls `detector.detect(frame_bgr)`.
  3. `TagDetector` returns a list of `Detection` objects with id, corners, center.
- **Postconditions**: Detection result available without opening a camera or pipeline.
- **Acceptance Criteria**:
  - [ ] `TagDetector` is stateless; calling `detect()` twice returns same result.
  - [ ] `DetectorConfig` provides sensible defaults.
  - [ ] `TagDetector` absorbs duplicate module-level `build_detectors()` functions.

---

## SUC-004: Track Tags Between Detection Frames (Optical Flow)

- **Actor**: `DetectionPipeline` (internal)
- **Preconditions**: Two consecutive grayscale frames are available; prior detections known.
- **Main Flow**:
  1. `OpticalFlowTracker.update(gray, detections)` called on new frame.
  2. Tracker uses LK optical flow on 4 corners of each known tag.
  3. Returns updated `Detection` list with 2D translation, rotation rate, scale delta.
- **Postconditions**: Tag positions updated at full frame rate; detection runs less frequently.
- **Acceptance Criteria**:
  - [ ] Tracker returns a list of `Detection` objects per frame.
  - [ ] When `detections` is not None, tracker resets with new detections.
  - [ ] 4-corner data preserved in output (no information discarded).

---

## SUC-005: Estimate Tag Velocity with EMA Smoothing

- **Actor**: `DetectionPipeline` (internal)
- **Preconditions**: Per-tag position history is available.
- **Main Flow**:
  1. `VelocityEstimator.update(position, timestamp)` called each frame.
  2. Estimator applies EMA smoothing and deadband suppression.
  3. Returns `(velocity_vec, speed)` tuple.
  4. `predict_position(t)` extrapolates position linearly.
- **Postconditions**: Smoothed velocity available for each tracked tag.
- **Acceptance Criteria**:
  - [ ] EMA alpha and deadband threshold are configurable.
  - [ ] Velocity is zero when position change is below deadband.
  - [ ] `predict_position(t)` returns plausible extrapolated position.

---

## SUC-006: Access a Live Tag Object from a Running Playfield

- **Actor**: Library consumer (Python code or AI agent)
- **Preconditions**: `Playfield` is started and tags are being detected.
- **Main Flow**:
  1. Consumer calls `field.tag(42)` to obtain a `Tag` handle.
  2. Consumer calls `tag.update()` to pull the latest snapshot.
  3. Consumer reads `tag.cx`, `tag.cy`, `tag.wx`, `tag.wy`, `tag.speed`,
     `tag.heading`, `tag.orientation`.
  4. Consumer calls `tag.position_at(t)` to extrapolate to a future time.
- **Postconditions**: Tag state accessed with simple flat properties.
- **Acceptance Criteria**:
  - [ ] `Tag` properties match the underlying `TagRecord` values.
  - [ ] `tag.update()` atomically replaces internal snapshot.
  - [ ] `tag.position_at(t)` delegates to `TagRecord.estimate(t)`.
  - [ ] `field.tags()` returns a dict mapping tag id to `Tag`.

---

## SUC-007: Stream Tag Updates via Callback or Generator

- **Actor**: Library consumer
- **Preconditions**: `Playfield` is started.
- **Main Flow**:
  1. Consumer calls `field.stream()` which yields `list[Tag]` per frame.
  2. Or consumer calls `field.on_frame(callback)` for push-based notification.
- **Postconditions**: Consumer receives tag updates without polling a ring buffer.
- **Acceptance Criteria**:
  - [ ] `field.stream()` is a generator that terminates when `stop()` is called.
  - [ ] `field.on_frame(callback)` invokes callback on background thread.
  - [ ] Both modes deliver the same data as `field.tags()`.

---

## SUC-008: Display a Rich TUI Dashboard of Live Tags

- **Actor**: Developer / operator
- **Preconditions**: A `Playfield` is started; terminal supports Rich.
- **Main Flow**:
  1. `TagTableTUI` is constructed with a `Playfield` reference.
  2. TUI renders a table of tag IDs, positions, speeds, headings.
  3. TUI updates on each frame callback.
- **Postconditions**: Live tag data displayed in terminal without mixing UI code into detection logic.
- **Acceptance Criteria**:
  - [ ] `TagTableTUI` has no dependency on `AprilCam`.
  - [ ] TUI reads tag data only through the public `Playfield` / `Tag` API.
  - [ ] TUI can be started and stopped independently of the detection pipeline.

---

## SUC-009: Calibrate a Playfield and Persist Results

- **Actor**: Developer / operator
- **Preconditions**: ArUco corner markers are visible; field dimensions known.
- **Main Flow**:
  1. Developer constructs `Playfield(camera, width_cm=101, height_cm=89)`.
  2. Developer calls `playfield.calibrate()`.
  3. Calibration detects corner ArUco markers, computes homography, persists to file.
  4. On next construction, calibration file is loaded automatically.
- **Postconditions**: World-coordinate mapping available; no manual JSON editing needed.
- **Acceptance Criteria**:
  - [ ] `calibrate()` writes a calibration file loadable by `CameraCalibration`.
  - [ ] `Playfield` loads calibration automatically when file exists.
  - [ ] `width_cm` / `height_cm` are not required if calibration file contains them.

---

## SUC-010: Run Smoke, Unit, and System Tests Without Hardware

- **Actor**: Developer / CI
- **Preconditions**: `tests/movies/*.mov` files present; no camera attached.
- **Main Flow**:
  1. Developer runs `uv run pytest tests/smoke/` — passes in seconds.
  2. Developer runs `uv run pytest tests/unit/` — each class tested in isolation.
  3. Developer runs `uv run pytest tests/system/` — pipeline tested against videos.
- **Postconditions**: Full test suite passes without camera hardware.
- **Acceptance Criteria**:
  - [ ] `tests/smoke/` passes in under 5 seconds.
  - [ ] `tests/unit/` covers Camera, Tag, TagDetector, OpticalFlowTracker,
        VelocityEstimator, DetectionPipeline, TagRecord, RingBuffer.
  - [ ] `tests/system/` detects tags in all four test videos.
  - [ ] No test requires a connected camera.
