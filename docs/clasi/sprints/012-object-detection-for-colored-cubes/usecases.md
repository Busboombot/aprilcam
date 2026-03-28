---
sprint: "012"
status: done
---

# Use Cases — Sprint 012: Object Detection for Colored Cubes

## SUC-012-001: Detect Colored Cubes at High Frame Rate

**Actor**: Python developer or AI agent

**Precondition**: B&W camera (cam 3) and color camera (cam 2) are
connected. Both have homography files in `data/`.

**Flow**:
1. Call `detect_tags(camera=3, detect_objects=True, color_camera=2)`.
2. System opens both cameras. B&W runs at full frame rate.
3. On each B&W frame: detect AprilTags and bright square contours.
4. Color camera runs in background at ~2fps, classifying colors.
5. System fuses B&W positions with color labels by world proximity.
6. Generator yields `FrameResult` with `tags` and `objects` per frame.
7. Objects include world_xy, color, bbox, type.

**Postcondition**: Objects detected at >40fps with color labels.

**Variations**:
- No color camera → objects detected but color="unknown"
- Color camera much slower → labels arrive after a few frames,
  objects start as "unknown" and get colored once classified
- `detect_objects=False` (default) → backward compatible, tags only

## SUC-012-002: Filter Out Tags and Robot

**Actor**: System (automatic)

**Precondition**: Detection loop running with `detect_objects=True`.

**Flow**:
1. Square detection finds a contour on B&W frame.
2. System checks if contour overlaps any detected AprilTag corners.
3. System checks if contour overlaps any ArUco corner marker.
4. System checks if contour is within `robot_exclusion_radius` of
   the robot tag (configurable, default tag ID 1).
5. If any check matches, contour is excluded from objects.

**Postcondition**: Only non-tag, non-robot objects are reported.

## SUC-012-003: Color Label Persistence

**Actor**: System (automatic)

**Precondition**: Color camera has classified an object as "red".

**Flow**:
1. B&W camera detects a square at world position (50.2, 30.1).
2. Last color classification mapped a red object at (50.5, 30.3).
3. Distance < 5cm → label "red" assigned to the B&W object.
4. Next 20 frames: B&W sees the object; color camera hasn't
   refreshed yet.
5. Object retains "red" label from the cached classification.
6. Color camera eventually refreshes → confirms "red" or updates.

**Postcondition**: Color labels persist at full B&W frame rate.
