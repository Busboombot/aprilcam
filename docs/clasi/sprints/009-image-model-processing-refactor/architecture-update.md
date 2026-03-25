---
sprint: "009"
status: draft
---

# Architecture Update -- Sprint 009: Image Model & Processing Refactor

## What Changed

### New: `ImageFrame` dataclass (in `models.py`)

A unified image container that separates raw data from processing results:

```python
@dataclass
class ImageFrame:
    raw: np.ndarray                    # Original captured image (BGR)
    source: str                        # e.g. "cam_0", "file:path.jpg"
    timestamp: float                   # Time of capture
    width: int                         # Derived from raw.shape
    height: int                        # Derived from raw.shape

    # Homography / calibration (optional)
    homography: Optional[np.ndarray] = None
    calibration_units: Optional[str] = None

    # Processed variant
    processed: Optional[np.ndarray] = None
    processing_steps: List[str] = field(default_factory=list)

    # Detection results
    aruco_corners: Optional[Dict[int, np.ndarray]] = None  # ID -> 4x2 corners
    apriltags: Optional[List[AprilTag]] = None
```

### Modified: `AprilTag` dataclass (in `models.py`)

- Retains: `id`, `corners_px`, `center_px`, `top_dir_px`, `orientation_yaw`,
  `world_xy`, `frame`, `in_playfield`, `last_ts`
- Removes: velocity fields from per-detection data. Velocity is now
  exclusively computed by `Playfield` from flow history.
- Adds: `family: str` field (e.g. "36h11", "25h9")

### Modified: `AprilTagFlow` (in `models.py`)

- Still maintains position history deque
- `vel_px` and `speed_px` properties remain but are set by Playfield
  when it computes velocity, not self-computed from internal history
- May be simplified or merged into Playfield's internal tracking

### Modified: `image_processing.py`

All `process_*` functions already accept `np.ndarray`. This sprint
ensures no function reaches back to a camera or source_id. The module
becomes a pure function library: `f(ndarray, params) -> result`.

No signature changes needed — the functions already take arrays. The
change is ensuring callers pass arrays (not capturing frames internally).

### Modified: `Playfield` (in `playfield.py`)

- Owns tag flow tracking (already does via `_flows` dict)
- Adds velocity computation: compares current and previous tag positions
  in flow history, computes `vel_px`, `speed_px`, `vel_world`, `speed_world`
- ArUco corner detection (`update()`) remains one-time setup — called once
  during `create_playfield`, not every frame
- `add_tag()` now computes and stores velocity on the flow

### Modified: `AprilCam` (in `aprilcam.py`)

- `process_frame()` returns an `ImageFrame` populated with detection results
  instead of directly returning `List[TagRecord]`
- Velocity computation removed from `process_frame()` — delegated to Playfield
- Frame preprocessing (CLAHE, sharpening) recorded in `ImageFrame.processing_steps`

### Modified: `mcp_server.py`

- All image-returning tools resolve `source_id` to a captured frame early,
  construct an `ImageFrame`, then pass `frame.raw` to processing functions
- Detection loop feeds `ImageFrame` objects to Playfield
- No external API changes — same tool names, same parameters, same response shapes

### Modified: `detection.py`

- `DetectionLoop` produces `ImageFrame` objects internally
- `TagRecord` construction uses data from `ImageFrame.apriltags` + velocity
  from Playfield flow
- `RingBuffer` and `FrameRecord` unchanged

## Why

The current architecture tightly couples camera capture with image processing
and tag tracking. This prevents:

1. **Testing without cameras** — processing functions need a `source_id` to
   capture a frame, making unit tests require camera mocks or hardware
2. **Static image analysis** — no way to run the detection pipeline on an
   image file without opening a fake camera
3. **Clear ownership of velocity** — velocity is computed in multiple places
   (AprilTag, AprilTagFlow, AprilCam) when it conceptually belongs in Playfield
   which has the temporal context

This refactor establishes clear data flow:
`Camera -> ImageFrame -> Processing (pure functions on arrays) -> Playfield (temporal tracking)`

## Impact on Existing Components

- **MCP external API**: No changes. Tools accept the same parameters and
  return the same response shapes.
- **Detection loop**: Internal change only. Produces ImageFrame instead of
  raw arrays, but RingBuffer/FrameRecord interface unchanged.
- **Tests**: Existing tests need updates for new function signatures and
  return types. New tests added for ImageFrame and array-based processing.
- **CLI tools**: No changes (they don't use image processing directly).

## Migration Concerns

- All changes are internal refactoring — no external API changes
- No data migration needed
- Existing test images in `tests/data/` used for new tests
- Backward compatibility: MCP tool responses unchanged

## Open Questions

1. Should `ImageFrame` be immutable (frozen dataclass) or mutable? Mutable
   allows progressive enrichment (detect corners, then detect tags, then
   add metadata). Frozen requires creating new instances at each step.

2. Should `AprilTagFlow` remain a separate class, or should its tracking
   be fully absorbed into `Playfield._flows`? Currently `AprilTagFlow` is
   a thin wrapper around a deque of `AprilTag` snapshots — Playfield could
   own this directly.

3. Where should `TagRecord` construction happen? Currently in `DetectionLoop`.
   Options: (a) keep in DetectionLoop, (b) move to Playfield since it now
   owns velocity, (c) construct in ImageFrame as a convenience method.
