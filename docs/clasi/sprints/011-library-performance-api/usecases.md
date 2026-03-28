---
sprint: "011"
status: done
---

# Use Cases — Sprint 011: Library Performance & API

## SUC-011-001: Library Tag Detection Loop

**Actor**: Python developer importing aprilcam

**Precondition**: Camera is connected; homography file exists in `data/`
from a previous calibration session.

**Flow**:
1. Developer calls `detect_tags(camera=0)` (or with device name).
2. System opens camera, discovers matching homography file by device
   name + resolution, loads it.
3. System yields `List[TagRecord]` per frame — each record includes
   tag ID, pixel center, world coordinates, orientation, velocity.
4. Developer iterates the generator, processing tag data.
5. On generator close or KeyboardInterrupt, camera is released.

**Postcondition**: Tags detected at maximum frame rate with world
coordinates and velocities.

**Variations**:
- No homography file found → tags returned with pixel coordinates
  only (world_xy=None).
- `homography="auto"` (default) → auto-discover from `data/`.
- `homography=<path>` → load specific file.
- `homography=None` → skip, pixel-only mode.

## SUC-011-002: Per-Camera Homography Persistence

**Actor**: Developer or AI agent calibrating cameras

**Precondition**: Camera is open and ArUco corner markers are visible.

**Flow**:
1. Actor runs calibration (via MCP `calibrate_playfield` or library).
2. System saves homography to `data/homography-<device-slug>-<WxH>.json`.
3. On next session, `detect_tags(camera=0)` auto-loads the matching
   file without recalibration.

**Postcondition**: Homography persists across sessions for each camera.

**Variations**:
- Two identical cameras (same name + resolution) → append index as
  tiebreaker.
- Fallback: `data/homography.json` is loaded if no per-camera file
  matches and the global file exists.

## SUC-011-003: Camera Contention Diagnosis

**Actor**: Developer trying to open a camera

**Precondition**: Camera is already in use by another process.

**Flow**:
1. Developer calls `detect_tags(camera=0)` or `open_camera(0)`.
2. OpenCV fails to open the camera.
3. System detects contention: queries OS for processes using the
   camera device.
4. System raises `CameraInUseError` with message:
   "Camera 0 (Brio 501) is in use by process 'python3' (PID 12345).
   Kill it with: kill 12345"

**Postcondition**: Developer knows exactly which process to kill.

**Variations**:
- Camera doesn't exist → `CameraNotFoundError`.
- Permission denied → `CameraPermissionError`.
- Cannot determine blocking process → generic message with
  suggestion to check `lsof` / `fuser`.

## SUC-011-004: Clean Library Imports

**Actor**: Python developer

**Precondition**: aprilcam is installed.

**Flow**:
1. Developer writes `from aprilcam import detect_tags, AprilCam, TagRecord`.
2. Import succeeds with no internal module path knowledge required.

**Postcondition**: Key classes and functions available from top-level package.
