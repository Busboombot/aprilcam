---
id: "003"
title: "Dual-camera fusion and color label persistence"
status: done
use-cases: ["SUC-012-002", "SUC-012-003"]
depends-on: ["001", "002"]
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Dual-camera fusion and color label persistence

## Description

Implement the dual-camera fusion layer that bridges the B&W camera (fast,
reliable tag/shape detection) and the color camera (color classification).
The key insight is that the B&W camera runs the main detection loop at high
FPS, while the color camera runs in a separate thread providing color labels
that are spatially matched and persisted.

### ObjectFuser class

Add `ObjectFuser` to `src/aprilcam/objects.py`. This class maintains a
spatial color map and fuses color information onto B&W-detected objects.

**Internal state**:
- `_color_map: dict[tuple[float, float], tuple[str, float]]` -- maps quantized
  world positions to `(color_name, timestamp)` pairs. Positions are quantized
  by rounding to nearest 1cm (round to 1 decimal place in cm units).
- `match_radius: float` -- maximum distance in world units (cm) to match a
  B&W object to a color map entry. Default 5.0.

**`update_colors(color_objects: list[ObjectRecord])`**:
- For each ObjectRecord with a non-None `world_xy` and `color != "unknown"`,
  quantize the position and insert/update the color map entry with the current
  timestamp.

**`fuse(bw_objects: list[ObjectRecord]) -> list[ObjectRecord]`**:
- For each B&W ObjectRecord:
  - If `world_xy` is None, keep as-is (no fusion possible without world coords)
  - Otherwise, search the color map for the nearest entry within `match_radius`
  - If found, create a new ObjectRecord (frozen, so replace) with the matched
    color label
  - If not found, keep `color="unknown"`
- Return the list of (possibly color-enhanced) ObjectRecords

**`clear_stale(max_age_seconds: float = 5.0)`**:
- Remove all color map entries whose timestamp is older than
  `time.time() - max_age_seconds`
- Called periodically to prevent stale color labels from persisting
  indefinitely when objects are removed from the playfield

**Color label persistence**: Once a world position has a color in the map,
it retains that color across frames until either overwritten by a newer
classification at the same position or cleared by `clear_stale()`. This
means the color camera does not need to run at the same FPS as the B&W
camera -- occasional updates are sufficient.

### ColorCameraThread class

Add `ColorCameraThread` to `src/aprilcam/objects.py`. This runs the color
camera and classifier in a daemon thread, feeding results into an ObjectFuser.

**Constructor**: `ColorCameraThread(camera_id, fuser: ObjectFuser, classifier: ColorClassifier, homography=None, fps=5.0)`

- `camera_id`: camera index or name for `open_camera`
- `fuser`: the shared ObjectFuser instance
- `classifier`: a configured ColorClassifier
- `homography`: optional homography matrix for world coordinate mapping
- `fps`: target classification rate (default 5.0 -- color classification
  does not need high FPS)

**`start()`**: open the camera, launch a daemon thread that:
1. Captures a frame
2. Runs `classifier.classify(frame, homography)`
3. Calls `fuser.update_colors(results)`
4. Calls `fuser.clear_stale()`
5. Sleeps to maintain target FPS

**`stop()`**: set a stop event, join the thread, close the camera.

The thread is a daemon so it does not prevent process exit.

## Acceptance Criteria

- [ ] `ObjectFuser` class exists in `src/aprilcam/objects.py`
- [ ] `ObjectFuser.update_colors()` inserts color entries into the spatial map keyed by quantized world position
- [ ] `ObjectFuser.fuse()` matches B&W objects to color map entries by nearest position within `match_radius`
- [ ] `ObjectFuser.fuse()` returns new ObjectRecords with assigned color labels (does not mutate originals)
- [ ] Objects outside `match_radius` retain `color="unknown"`
- [ ] Objects without `world_xy` are passed through unchanged
- [ ] Color labels persist across multiple `fuse()` calls without re-running `update_colors()`
- [ ] `ObjectFuser.clear_stale()` removes entries older than `max_age_seconds`
- [ ] `ColorCameraThread` runs color classification in a daemon thread
- [ ] `ColorCameraThread.start()` and `stop()` manage the thread lifecycle
- [ ] `ColorCameraThread` updates the shared ObjectFuser's color map
- [ ] All new code has type hints
- [ ] All tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- verify no regressions
- **New tests to write**: `tests/test_fusion.py` containing:
  - `test_fuser_update_and_fuse_basic` -- update with one colored object at (10.0, 20.0), fuse a B&W object at the same position, verify color assigned
  - `test_fuser_match_radius` -- B&W object at (10.0, 20.0), color entry at (10.3, 20.2), verify match within 5cm radius
  - `test_fuser_no_match_outside_radius` -- B&W object at (10.0, 20.0), color entry at (20.0, 30.0), verify color stays "unknown"
  - `test_fuser_persistence` -- update colors once, call fuse() multiple times on subsequent frames, verify color persists each time
  - `test_fuser_overwrite` -- update with "red" at position, then update with "blue" at same position, verify fuse returns "blue"
  - `test_fuser_clear_stale` -- insert entry, advance time past max_age, call clear_stale, verify entry removed and fuse returns "unknown"
  - `test_fuser_no_world_xy` -- B&W object with world_xy=None, verify passed through unchanged
  - `test_fuser_quantization` -- objects at (10.04, 20.06) and (10.0, 20.1) should map to the same quantized key
  - `test_color_camera_thread_lifecycle` -- mock camera and classifier, start thread, verify fuser gets updated, stop thread, verify thread joins
  - `test_color_camera_thread_is_daemon` -- verify the thread is a daemon thread
- **Verification command**: `uv run pytest tests/test_fusion.py -v`
