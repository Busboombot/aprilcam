---
status: pending
---

# AprilCam: Add Object Detection for Colored Cubes

## Summary

The robot navigation system needs to detect and locate small colored
cubes on the playfield without switching to a separate color camera
or running custom OpenCV code. This should be built into AprilCam so
that `detect_tags` (or a companion function) returns cube positions
alongside tag positions, all in world coordinates.

## Context

We're doing pick-and-place tasks: a robot with a gripper navigates to
colored cubes, picks them up, and moves them to target positions
(marked by AprilTags). Currently, detecting cubes requires:

1. Opening camera 2 (color, 1920×1080, ~2fps) separately
2. Running custom HSV thresholding in OpenCV
3. Building a separate homography for camera 2
4. Matching color detections to B&W camera positions by nearest-neighbor
5. Using the B&W position (more accurate) with the color label

This is slow (~3-5 seconds per detection cycle), unreliable (camera
open/close conflicts, false positives on tag bases), and requires
the agent to write and debug vision code every session.

## What We Need

### 1. Colored Object Detection in detect_tags

A way to detect non-tag objects (cubes, balls, etc.) on the playfield
and return them alongside tag detections. Each detected object should
include:

- **World position** (cm) — using the playfield homography
- **Color** — classified from the color camera (red, green, blue,
  yellow, orange, purple, white)
- **Bounding box** or contour in pixel coordinates
- **Confidence** or area (to filter false positives)
- **Object type** — "cube", "ball", or generic "object"

Ideally this would be a parameter on `detect_tags`:

```python
for tags in detect_tags(camera=3, homography="auto",
                        detect_objects=True, color_camera=2):
    for tag in tags:
        ...  # AprilTag as usual
    # New: objects detected on this frame
    for obj in tags.objects:  # or however it's structured
        print(f"{obj.color} {obj.type} at ({obj.world_xy[0]:.1f}, {obj.world_xy[1]:.1f})")
```

### 2. Dual-Camera Fusion

The B&W camera (camera 3, 1280×800, ~49fps) is used for navigation
because it's fast. The color camera (camera 2, 1920×1080, ~2fps) is
needed for color identification. These need to be fused:

- **B&W camera** provides precise positions at high frame rate
- **Color camera** provides color classification (can run less frequently)
- Objects should be matched between cameras by world-coordinate proximity
- Color labels should persist across frames (don't need to re-classify
  every frame — once a cube is identified as "red", it stays "red"
  until it moves significantly)

### 3. Filtering Out Tags and Robot

Detected objects should NOT include:
- AprilTag markers (already reported as tags)
- ArUco corner markers
- The robot body (known position from tag 1)

The current approach filters by distance from known tag positions, but
this is fragile. A better approach:
- Objects that overlap a detected tag → not an object
- Objects larger than a cube's expected size → not an object
- Objects within N cm of the robot tag → not an object (configurable)

### 4. Object Tracking Across Frames

When the robot picks up a cube and moves it, the system should:
- Notice that a cube disappeared from one location
- Notice that a cube appeared at a new location (or in the gripper area)
- Track object identity across frames (same red cube, just moved)

This helps the agent verify: "Did I actually pick up the cube?"
by checking if the cube is no longer at the pickup location and is
now near the gripper.

## Current Detection Method (for reference)

This is what we're doing manually in OpenCV. It works but is slow
and error-prone:

### Color Detection (camera 2)

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

color_ranges = {
    "red":    [((0, 50, 50), (12, 255, 255)),
               ((165, 50, 50), (180, 255, 255))],  # wraps around
    "green":  [((35, 50, 50), (85, 255, 255))],
    "blue":   [((90, 50, 50), (130, 255, 255))],
    "yellow": [((15, 50, 50), (35, 255, 255))],
    "orange": [((12, 80, 80), (22, 255, 255))],
    "purple": [((125, 40, 40), (160, 255, 255))],
}

for color_name, ranges in color_ranges.items():
    mask = np.zeros(...)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(mask, ...)
    # Filter by area (200-5000px), aspect ratio (<2.5), etc.
```

### B&W Square Detection (camera 3)

```python
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, ...)
# Filter: area 50-1200, aspect ratio < 2, not near known tags
```

### Matching

Match color detections to B&W detections by nearest world-coordinate
position (max distance 5cm). Use the B&W position as the authoritative
position.

## Nice-to-Haves

- **MCP tool**: `get_objects(source_id)` returning detected objects
  with color, position, and type — similar to `get_tags`.
- **Annotated frame**: `get_frame(annotate=True)` already draws tags;
  also draw detected objects with color-coded outlines and labels.
- **Configurable color ranges**: Allow tuning HSV thresholds per
  deployment (lighting varies between setups).
- **Composite source**: If a composite already merges multiple cameras,
  object detection on the composite could handle the fusion internally.

## Hardware Context

- **Camera 2**: "HD USB CAMERA", 1920×1080, color, ~2fps for detection
- **Camera 3**: "Brio 501" (actually a B&W overhead cam), 1280×800, ~49fps
- **Field**: 102×89cm black surface, ArUco corner markers, per-camera
  homography files in `data/`
- **Cubes**: ~2cm foam cubes in red, green, blue, yellow, orange, purple, white
- **Tags on cubes**: None — cubes are detected by color/brightness only
