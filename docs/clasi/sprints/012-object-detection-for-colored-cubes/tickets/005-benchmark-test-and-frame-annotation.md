---
id: "005"
title: "Benchmark test and frame annotation"
status: todo
use-cases: ["SUC-012-001"]
depends-on: ["004"]
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Benchmark test and frame annotation

## Description

Add a performance benchmark for the object detection pipeline and extend
the frame annotation logic to draw detected objects. This ticket ensures
the detection pipeline meets performance targets and that visual debugging
is available through annotated frames.

### Benchmark: tests/bench_objects.py

Create a benchmark script modeled on the existing `bench_tags.py`. The
benchmark measures end-to-end FPS of the detection pipeline with object
detection enabled.

**Structure**:

1. Open a B&W camera (or accept camera index as CLI argument)
2. Call `detect_tags()` with `detect_objects=True`
3. Run for a configurable number of frames (default 300, ~10 seconds at 30fps)
4. Track per-frame timing: total frame time, tag detection time, square
   detection time
5. Report:
   - Average FPS (total)
   - Average tag detection time (ms)
   - Average square detection time (ms)
   - Number of tags and objects detected per frame (min/avg/max)
6. Performance target: > 40 FPS on a B&W global shutter camera with
   square detection enabled (tag detection + square detection combined)

**Usage**: `uv run python tests/bench_objects.py [--camera INDEX] [--frames N]`

The benchmark is a manual script (not a pytest test) since it requires
live camera hardware. However, include a pytest-compatible wrapper that
skips if no camera is available.

### Frame annotation updates

Update the `_handle_get_frame` annotation logic in `src/aprilcam/mcp_server.py`
to include detected objects when `annotate=True`:

**For each detected object**:
- Draw a colored rectangle around the bounding box using the object's color
  label. Color mapping for drawing:
  - "red" -> (0, 0, 255)
  - "green" -> (0, 255, 0)
  - "blue" -> (255, 0, 0)
  - "yellow" -> (0, 255, 255)
  - "orange" -> (0, 165, 255)
  - "purple" -> (255, 0, 128)
  - "unknown" -> (200, 200, 200) (gray)
- Draw the color label text above the bounding box (same color, smaller font
  than tag labels)
- Draw a small filled circle at the object center point
- Rectangle line thickness: 2px
- If `world_xy` is available, show world coordinates below the label

The annotation should integrate cleanly with existing tag annotation -- tags
and objects should both appear when annotate=True.

## Acceptance Criteria

- [ ] `tests/bench_objects.py` exists and runs with `--camera` and `--frames` arguments
- [ ] Benchmark reports average FPS, per-component timing, and detection counts
- [ ] Benchmark achieves > 40 FPS on B&W camera with square detection (manual verification)
- [ ] Frame annotation draws colored rectangles around detected objects
- [ ] Rectangle color matches the object's color label
- [ ] Color label text is drawn above each object bounding box
- [ ] Center point is marked with a filled circle
- [ ] World coordinates shown when available
- [ ] Annotation works alongside existing tag annotation (both visible)
- [ ] "unknown" color objects are drawn in gray
- [ ] All tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- verify annotation changes do not break existing get_frame tests
- **New tests to write**:
  - `tests/test_object_annotation.py`:
    - `test_annotate_objects_draws_rectangles` -- create a mock frame with ObjectRecords, call annotation logic, verify cv2.rectangle was called with correct coordinates and colors
    - `test_annotate_objects_draws_labels` -- verify cv2.putText called with color name
    - `test_annotate_objects_draws_center` -- verify cv2.circle called at center point
    - `test_annotate_objects_color_mapping` -- verify each color name maps to correct BGR tuple
    - `test_annotate_unknown_color_gray` -- object with color="unknown" drawn in gray (200, 200, 200)
    - `test_annotate_objects_with_world_xy` -- verify world coordinates displayed when available
    - `test_annotate_mixed_tags_and_objects` -- verify both tags and objects annotated on same frame
  - `tests/test_bench_objects.py`:
    - `test_bench_script_importable` -- verify bench_objects.py can be imported without errors
    - `test_bench_skips_without_camera` -- verify pytest wrapper skips gracefully when no camera available
- **Verification command**: `uv run pytest tests/test_object_annotation.py tests/test_bench_objects.py -v`
