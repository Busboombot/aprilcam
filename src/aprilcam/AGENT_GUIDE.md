# AprilCam Agent Guide

You are using **AprilCam**, a Python library and MCP server for camera
management, AprilTag/ArUco fiducial marker detection, playfield
homography, and image processing on robotics playfields.

## Quick Start (Library)

```python
from aprilcam import detect_tags

for tags in detect_tags(camera=0):
    for t in tags:
        print(f"Tag {t.id} at pixel ({t.center_px[0]:.0f}, {t.center_px[1]:.0f})")
        if t.world_xy:
            print(f"  world ({t.world_xy[0]:.1f}, {t.world_xy[1]:.1f}) cm")
```

## Quick Start (MCP)

```json
{ "mcpServers": { "aprilcam": { "command": "aprilcam", "args": ["mcp"] } } }
```

Then call `list_cameras` → `open_camera` → `start_detection` → `get_tags`.

## Core Concepts

### Cameras
- **Index**: Integer (0, 1, 2...) identifying a camera device.
- **camera_id**: Handle string returned by `open_camera` (e.g., `cam_0`).
- Cameras are opened by index or by name pattern (substring match).
- Screen capture is also available as a camera source (`source="screen"`).

### Playfields
- A **playfield** is a camera view with ArUco corner markers (IDs 0-3)
  defining a rectangular region.
- **Deskew**: Perspective-corrects the playfield to a top-down rectangle
  using the corner markers. No calibration needed.
- **Calibration**: Maps pixel coordinates to real-world units (cm) by
  providing physical measurements between corner markers.
- A `playfield_id` can be used anywhere a `camera_id` is accepted.

### Tag Detection
- **Detection loop**: A persistent background loop that detects
  AprilTag 36h11 and ArUco markers on every frame, storing results
  in a 300-frame ring buffer (~10 seconds at 30fps).
- **TagRecord** fields: `id`, `center_px`, `corners_px`,
  `orientation_yaw`, `world_xy` (if calibrated), `vel_px`,
  `speed_px`, `vel_world`, `speed_world`, `heading_rad`,
  `in_playfield`, `timestamp`, `frame_index`.
- Velocities are EMA-smoothed with dead-band suppression.

### Homography Files
- Calibration data is saved to `data/` as JSON files.
- **Per-camera naming**: `data/homography-<device-slug>-<WxH>.json`
  (e.g., `data/homography-brio-501-1920x1080.json`).
- **Global fallback**: `data/homography.json` is used if no per-camera
  file exists.
- The `detect_tags()` generator auto-discovers the right homography
  file when `homography="auto"` (default).
- Homography only needs to be computed once per camera position.

## Library API

### Primary Interface

```python
from aprilcam import detect_tags

# Simple: open camera 0, auto-load homography, yield tags per frame
for tags in detect_tags(camera=0):
    ...

# With options
for tags in detect_tags(
    camera=0,              # index or device name pattern
    homography="auto",     # "auto", path, or None
    family="36h11",        # AprilTag family
    data_dir="data",       # where homography files live
    proc_width=0,          # processing width (0 = native)
):
    ...
```

### Available Imports

```python
from aprilcam import (
    AprilCam,          # Core detection engine
    detect_tags,       # Generator API (recommended)
    TagRecord,         # Per-tag detection result
    DetectionLoop,     # Background detection thread
    AprilTag,          # Tag model with tracking state
    Playfield,         # Playfield polygon and deskew
)
```

### Lower-Level Usage

```python
from aprilcam import AprilCam
from aprilcam.config import AppConfig
import cv2, time

cfg = AppConfig.load()
H = cfg.load_homography(device_name="Brio 501", resolution=(1920, 1080))

cam = AprilCam(index=0, homography=H, headless=True)
cap = cam._init_capture()
cam.reset_state()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    tags = cam.process_frame(frame, time.monotonic())
    for tr in tags:
        print(tr.id, tr.center_px, tr.world_xy)

cap.release()
```

## MCP Tools Reference

### Camera Management
- `list_cameras()` → list of `{index, name, backend}`
- `open_camera(index?, pattern?, source?, backend?)` → `{camera_id}`
- `close_camera(camera_id)` → confirmation
- `capture_frame(source_id, format?, quality?)` → image

### Playfield & Homography
- `create_playfield(camera_id, max_frames?)` → `{playfield_id}`
- `create_playfield_from_image(path)` → `{playfield_id}`
- `calibrate_playfield(playfield_id, width, height, units?)` → calibration result
- `deskew_image(source_id, format?, quality?)` → deskewed image
- `get_playfield_info(playfield_id)` → polygon, calibration, dimensions

### Tag Detection & Tracking
- `start_detection(source_id, family?, proc_width?, detect_interval?)` → `{status: "started"}`
- `stop_detection(source_id)` → confirmation
- `get_tags(source_id)` → `{tags: [{id, center_px, world_xy, vel_px, ...}]}`
- `get_tag_history(source_id, num_frames?)` → ring buffer frames
- `stream_tags(source_id)` → SSE stream of tag updates

### Image Processing
- `get_frame(source_id, format?, quality?, annotate?)` → raw frame
- `crop_region(source_id, x, y, w, h)` → cropped image
- `detect_lines(source_id)` → line segments
- `detect_circles(source_id)` → circles with centers/radii
- `detect_contours(source_id)` → contour polygons
- `detect_motion(source_id)` → motion regions
- `detect_qr_codes(source_id)` → decoded QR data
- `apply_transform(source_id, operation, ...)` → transformed image

### Frame Model (batch processing)
- `create_frame(source_id)` / `create_frame_from_image(path)` → `{frame_id}`
- `process_frame(frame_id, operations)` → processed frame
- `get_frame_image(frame_id)` → image data
- `save_frame(frame_id, path)` → saved file path
- `release_frame(frame_id)` → confirmation
- `list_frames()` → active frames

### Multi-Camera Compositing
- `create_composite(name, sources)` → `{composite_id}`
- `get_composite_frame(composite_id)` → merged frame
- `get_composite_tags(composite_id)` → merged tag detections

### Live View (web UI)
- `start_live_view(source_id, annotate?)` → `{view_id, url}`
- `stop_live_view(view_id)` → confirmation

## Common Workflows

### Workflow 1: Detect tags on a playfield with world coordinates

```
1. list_cameras          → find camera index
2. open_camera(index=0)  → cam_0
3. create_playfield(camera_id="cam_0")  → pf_0
4. calibrate_playfield(playfield_id="pf_0", width=102, height=89, units="cm")
5. start_detection(source_id="pf_0")
6. get_tags(source_id="pf_0")  → tags with world_xy in cm
```

### Workflow 2: Quick visual inspection

```
1. open_camera(index=0)  → cam_0
2. get_frame(source_id="cam_0", format="base64", annotate=true)
```

### Workflow 3: Track tag movement over time

```
1. open_camera + create_playfield + calibrate + start_detection
2. get_tag_history(source_id="pf_0", num_frames=60)
   → last 60 frames of tag positions, velocities, headings
```

## CLI Commands

```
aprilcam mcp          # Start MCP server (stdio)
aprilcam web          # Start HTTP server with REST API, MCP SSE, WebSocket, and web UI
aprilcam cameras      # List available cameras
aprilcam taggen       # Generate AprilTag/ArUco marker images
aprilcam live         # Open live camera view with tag overlays
aprilcam init         # Configure MCP entries for Claude Code / VS Code
aprilcam tool         # List, inspect, and run MCP tools from CLI
```

## Web Server

```
aprilcam web [--port 17439] [--host 0.0.0.0]
```

Provides:
- **REST API**: `POST /api/<tool_name>` — same tools as MCP
- **MCP SSE**: `/mcp/sse` — MCP protocol over Server-Sent Events
- **WebSocket**: `/ws/tags/<source_id>` — real-time tag streaming
- **Web UI**: `/` — live camera view with tag table

## Error Handling

- Camera not found → `CameraNotFoundError`
- Camera in use by another process → `CameraInUseError` (includes PID)
- Permission denied → `CameraPermissionError`
- MCP tools return `{"error": "message"}` on failure

## Tips for Agents

1. **Always call `list_cameras` first** to discover available cameras
   and their indices.
2. **Playfield source IDs work everywhere** camera IDs work — prefer
   playfields when you need deskewed views or world coordinates.
3. **Start detection once**, then poll `get_tags` repeatedly — don't
   start/stop the detection loop per query.
4. **Homography persists** — calibrate once, and future sessions
   auto-load the per-camera file from `data/`.
5. **Use `annotate=true`** on `get_frame` to see tag overlays for
   visual debugging.
6. **Tag velocities** are in pixels/second (`vel_px`) and world
   units/second (`vel_world`, if calibrated). They are EMA-smoothed
   with a dead-band to suppress jitter on stationary tags.
