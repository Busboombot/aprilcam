---
status: in-progress
sprint: '014'
tickets:
- 014-015
---

# Rebuild Test System with Video-Based Testing

## Description

Replace ad-hoc test fixtures with a structured, video-driven test system.

### VideoCamera Class

Subclass of Camera that reads frames from a video file instead of a live
camera. Same interface as Camera — construct with a path, call `read()` to
get frames sequentially. This allows the full detection pipeline to run
against recorded video without needing hardware.

### Test Videos

Located in `tests/movies/`:
- `bright-gsc.mov` — bright lighting, global shutter camera
- `bright-ov9782.mov` — bright lighting, OV9782 sensor
- `dim-gsc.mov` — dim lighting, global shutter camera
- `dim-ov9782.mov` — dim lighting, OV9782 sensor

Two lighting conditions, two camera types.

### Three-Tier Test Structure

1. **Smoke tests** — fast, basic sanity checks. Verify imports work, objects
   construct, nothing is fundamentally broken. Should run in seconds.

2. **Unit tests** — test each major class in isolation. Camera, Playfield,
   Tag, TagDetector, OpticalFlowTracker, VelocityEstimator, DetectionPipeline,
   TagRecord, RingBuffer, SquareDetector, ColorClassifier, etc. Mock
   dependencies where needed. Verify constructors, methods, properties,
   edge cases.

3. **System tests** — full tag detection pipeline tests driven by the test
   videos. Use VideoCamera to feed frames through the detection pipeline.
   Verify:
   - Tags are detected in each video
   - Detected positions are reasonable
   - Velocity computation produces sensible values
   - Detection works across both lighting conditions
   - Detection works across both camera types
   - estimate() produces extrapolated positions consistent with velocity
