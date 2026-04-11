---
sprint: "012"
status: done
---

# Architecture Update -- Sprint 012: Object Detection for Colored Cubes

## What Changed

### New Files

- **`aprilcam/objects.py`** — `ObjectRecord` frozen dataclass,
  `FrameResult` class (wraps tags + objects), `SquareDetector`
  (contour-based square detection on grayscale), `ColorClassifier`
  (HSV thresholding), `ObjectFuser` (matches B&W objects to color
  labels by world proximity, maintains persistent color map).

### Modified Files

- **`aprilcam/aprilcam.py`** — `process_frame()` optionally runs
  square detection on the same grayscale image (no extra conversion),
  returns squares alongside tags.
- **`aprilcam/stream.py`** — `detect_tags()` gains `detect_objects`
  and `color_camera` parameters. When enabled, runs color camera in
  background thread, fuses results, yields `FrameResult` instead
  of plain list.
- **`aprilcam/mcp_server.py`** — New `get_objects` MCP tool.
  Updated `get_frame` annotation to draw objects.
- **`aprilcam/__init__.py`** — Exports `ObjectRecord`, `FrameResult`.

### New Dataclasses

```python
@dataclass(frozen=True)
class ObjectRecord:
    center_px: tuple[float, float]
    world_xy: tuple[float, float] | None
    color: str              # "red", "green", ..., "unknown"
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    area_px: float
    object_type: str        # "cube", "object"
    confidence: float       # 0.0 - 1.0

class FrameResult:
    tags: list[TagRecord]
    objects: list[ObjectRecord]
    timestamp: float
    frame_index: int
    # Backward compatible: iter(frame_result) yields tags
```

### Performance Design

- Square detection uses the **same grayscale** already computed for
  AprilTag detection — zero extra conversion cost.
- Color camera runs in a **background thread** with its own
  VideoCapture, updating a shared dict. Main loop never blocks on it.
- Fusion is O(N*M) where N=B&W objects, M=color objects — both
  are small (<20), so this is negligible.
- Color labels are **cached** in the fuser. A B&W object near a
  previously-colored position gets that label without waiting for
  the color camera.

## Why

Pick-and-place tasks need to detect colored cubes alongside AprilTags.
The current manual approach (3-5 seconds per cycle, custom OpenCV code
each session) is too slow and fragile. Building detection into the
library at >40fps makes robot navigation practical.

## Impact on Existing Components

- `detect_tags()` return type changes from `list[TagRecord]` to
  `FrameResult` when `detect_objects=True`. `FrameResult` is iterable
  as `list[TagRecord]` for backward compatibility.
- When `detect_objects=False` (default), behavior is unchanged.
- `process_frame()` gains an optional `detect_squares` parameter.
  Default False, no impact on existing callers.

## Migration Concerns

None — all new parameters default to off. Existing code is unaffected.
