---
id: '004'
title: Tag Detection Loop & Ring Buffer
status: done
branch: sprint/004-tag-detection-loop-ring-buffer
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 004: Tag Detection Loop & Ring Buffer

## Goals

Add persistent, background AprilTag detection to the MCP server so that
AI agents can start a detection loop on a camera or playfield, then query
tag state on demand without blocking. Store per-frame tag records in a
ring buffer for history and motion analysis.

Specifically:

1. Provide MCP tools to start and stop a background detection loop.
2. Store tag records in a 300-frame circular buffer with timestamps.
3. Provide MCP tools to query current tag state and historical frames.
4. Compute velocity, speed, and heading for each detected tag.
5. Refactor the existing `AprilCam.run()` loop and `AprilTagFlow` to
   work headlessly in the MCP server context.

## Problem

The existing `AprilCam.run()` method is a blocking, interactive loop
that opens a window, processes keyboard input, and renders overlays.
It cannot be used by an MCP server that needs to run detection in a
background thread while serving tool requests on the main thread. There
is also no structured storage of frame history -- the `AprilTagFlow`
class keeps only 5 observations per tag, which is too few for trajectory
analysis, and there is no per-frame snapshot that captures the state of
all tags simultaneously.

AI agents need to:
- Start detection and walk away (non-blocking).
- Query the latest tag positions at any time.
- Retrieve a window of recent frames to analyze motion patterns.

None of this is possible with the current interactive loop.

## Solution

1. **DetectionLoop class** (`src/aprilcam/detection.py`): A new class
   that encapsulates the frame-grab / detect / track cycle from
   `AprilCam.run()`, running in a daemon thread. It owns an `AprilCam`
   instance configured for headless operation and writes results into a
   thread-safe ring buffer. The loop supports start/stop lifecycle and
   exposes current state via thread-safe accessors.

2. **FrameRecord and TagRecord dataclasses** (`src/aprilcam/models.py`):
   A `TagRecord` captures one tag's full state at a point in time (id,
   center_px, corners_px, orientation_yaw, world_xy, in_playfield,
   vel_px, vel_world, speed_px, speed_world, heading_rad). A
   `FrameRecord` bundles a timestamp, frame index, and a list of
   `TagRecord` objects for that frame.

3. **RingBuffer** (`src/aprilcam/detection.py`): A `collections.deque`
   with `maxlen=300` holding `FrameRecord` objects, protected by a
   `threading.Lock`. Provides `get_latest()` and `get_last_n(n)`
   accessors.

4. **MCP tools** (registered in the MCP server module):
   - `start_detection(source_id, family?, proc_width?, ...)` -- creates
     a `DetectionLoop`, starts the thread, returns a `detection_id`.
   - `stop_detection(detection_id)` -- signals the loop to stop, joins
     the thread, cleans up.
   - `get_tags(source_id)` -- returns the latest `FrameRecord` as
     structured JSON.
   - `get_tag_history(source_id, num_frames?)` -- returns the last N
     `FrameRecord` objects from the ring buffer.

5. **Refactor AprilCam** -- extract the detection/tracking core from
   `run()` into a `process_frame()` method that takes a frame and
   returns a list of `TagRecord` objects. The existing `run()` calls
   `process_frame()` internally so CLI behavior is preserved.

## Success Criteria

- An agent can call `start_detection` on an open camera or playfield
  and receive a detection_id without blocking.
- The detection loop runs continuously in the background, processing
  frames at the camera's native frame rate.
- `get_tags` returns current tag state within one frame period of the
  most recent detection.
- `get_tag_history` returns up to 300 frames of historical tag records.
- Each tag record includes: id, center_px, corners_px, orientation_yaw,
  world_xy (when homography is available), in_playfield, velocity
  (px/s and world units/s), speed, and heading.
- `stop_detection` cleanly terminates the loop and releases camera
  resources.
- The existing `aprilcam` CLI continues to work via the refactored
  `run()` method.
- All new code has unit tests with >80% coverage.

## Scope

### In Scope

- `DetectionLoop` class with background thread lifecycle.
- `FrameRecord` and `TagRecord` dataclasses.
- Ring buffer (300-frame `deque`) with thread-safe access.
- Four MCP tools: `start_detection`, `stop_detection`, `get_tags`,
  `get_tag_history`.
- Velocity computation in both pixel and world coordinate systems.
- Refactoring `AprilCam.run()` to extract `process_frame()`.
- Unit tests for DetectionLoop, ring buffer, TagRecord serialization,
  and MCP tool handlers.
- Headless operation (no OpenCV windows in the detection loop).

### Out of Scope

- Image processing tools (Sprint 5).
- Multi-camera compositing (Sprint 6).
- Display, overlays, or GUI rendering in the detection loop.
- Streaming or push-based notifications (agents poll via `get_tags`).
- Persistence of ring buffer across server restarts.
- Configurable ring buffer size (hardcoded at 300 for now).

## Test Strategy

**Unit tests** (`tests/test_detection.py`, `tests/test_models.py`):

- `TagRecord` and `FrameRecord` construction and JSON serialization.
- Ring buffer: adding frames, overflow at capacity, `get_latest()`,
  `get_last_n()` with various N values including 0 and >capacity.
- `DetectionLoop` lifecycle: start, process frames, stop. Use a mock
  `VideoCapture` that yields captured test images from `tests/data/`
  (`playfield_cam3.jpg`, `playfield_cam3_moved.jpg`) to simulate a
  real camera feed without hardware.
- Velocity computation: feed the two test images (tags in different
  positions) through `AprilTagFlow` and verify px/s calculations
  match expected displacements between the known tag positions.
  Also verify near-zero velocity when the same image is fed twice.
- Thread safety: concurrent reads during active detection loop.

**Integration tests** (`tests/test_mcp_detection.py`):

- MCP tool round-trip: `start_detection` -> `get_tags` -> `get_tag_history`
  -> `stop_detection` using a synthetic video source.
- Error cases: `get_tags` before starting detection, `stop_detection`
  on an invalid ID, starting detection on an already-active source.

**Regression**:

- Verify `aprilcam` CLI `run` subcommand still works after the
  `process_frame()` refactor (manual smoke test).

## Architecture Notes

- **Threading model**: One daemon thread per detection loop. The MCP
  server's main thread handles tool requests. Shared state (the ring
  buffer) is protected by a `threading.Lock`. The lock is held only
  during buffer reads/writes (microseconds), so contention is minimal.

- **DetectionLoop owns AprilCam**: Each `DetectionLoop` creates its own
  `AprilCam` instance configured for headless mode. This avoids shared
  mutable state between loops.

- **Source ID mapping**: The MCP server maintains a registry mapping
  `source_id` (camera_id or playfield_id from Sprints 2-3) to active
  `DetectionLoop` instances. Only one detection loop per source.

- **Frame indexing**: Frame indices are local to each detection loop,
  starting at 0 when the loop begins. Timestamps use
  `time.monotonic()` for consistent interval measurement.

- **Velocity computation**: Computed per-tag between consecutive frames
  where the tag is visible. Uses the center_px delta divided by the
  timestamp delta. World velocity is computed by transforming the
  pixel velocity vector through the homography Jacobian at the tag's
  position, or by differencing consecutive world_xy positions.

- **Graceful degradation**: If homography is not available (no
  calibration), world_xy, vel_world, and speed_world are null in the
  tag record. The tools still return valid data for pixel-space fields.

## GitHub Issues

(None linked yet.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
