---
id: "009"
title: "Image Model & Processing Refactor"
status: planning
branch: sprint/009-image-model-processing-refactor
use-cases: [SUC-001, SUC-002, SUC-003, SUC-004]
---

# Sprint 009: Image Model & Processing Refactor

## Goals

Introduce a robust `ImageFrame` model that cleanly separates image data,
metadata, and processing results from camera and playfield concerns. Refactor
all image processing functions to operate on NumPy arrays (not camera handles),
and refactor the `Playfield` to own tag tracking and velocity computation
rather than embedding it in per-frame image data.

## Problem

The current architecture tightly couples camera capture, image processing,
tag detection, and playfield tracking. MCP tools take a `source_id` (camera
handle) and internally capture frames, process them, and return results in
one step. This makes it impossible to:

- Test image processing without a live camera
- Reuse processing pipelines on static images from disk
- Separate the concerns of "what's in this image" from "where did this image
  come from" and "how have tags moved over time"

Velocity computation currently lives in `AprilTag`/`AprilTagFlow` models and
`AprilCam.process_frame()`, but conceptually belongs in the `Playfield` which
maintains temporal context across frames.

## Solution

1. **ImageFrame model**: A new dataclass holding raw image (ndarray), metadata
   (source, timestamp, resolution), optional homography, a processed image
   variant with transformation metadata, detected ArUco corners, and detected
   AprilTags.

2. **AprilTag model cleanup**: Ensure the AprilTag class carries family, ID,
   pixel location, orientation. Remove velocity from per-tag per-frame data —
   velocity is computed by Playfield from history.

3. **Processing on arrays**: Refactor `image_processing.py` functions to accept
   `np.ndarray` inputs directly. Higher-level functions accept/return
   `ImageFrame` objects.

4. **Camera separation**: Camera operations (open, close, capture) produce
   `ImageFrame` objects. All downstream processing operates on `ImageFrame`
   or raw arrays — no camera handle needed.

5. **Playfield owns flow**: Playfield maintains tag position history and
   computes velocities. ArUco corner detection for playfield creation is a
   one-time setup, not per-frame.

6. **MCP tool refactor**: MCP tools resolve `source_id` to a frame early, then
   pass the frame (or its ndarray) to processing functions.

## Success Criteria

- All image processing functions accept `np.ndarray` — no camera dependency
- `ImageFrame` model exists with raw image, metadata, processed variant,
  detected tags
- Playfield computes velocities from tag history; individual images do not
- All existing tests pass (updated as needed for new interfaces)
- New unit tests cover ImageFrame model and refactored processing functions
  using static test images (no camera required)
- MCP tools continue to work with same external API

## Scope

### In Scope

- New `ImageFrame` dataclass in `models.py`
- Refactor `AprilTag` / `AprilTagFlow` — clean separation of detection vs flow
- Refactor `image_processing.py` to operate on `np.ndarray`
- Refactor `Playfield` to own tag flow and velocity computation
- Refactor `mcp_server.py` tools to use ImageFrame internally
- Refactor `AprilCam.process_frame()` to produce ImageFrame
- Update all tests for new interfaces
- New tests using static test images

### Out of Scope

- New MCP tool APIs (external interface stays the same)
- Camera hardware changes
- New image processing algorithms
- Recording new test data (deferred — stakeholder will assist)
- Live view changes

## Test Strategy

- **Unit tests**: ImageFrame construction, metadata, transformation tracking.
  AprilTag model fields. Processing functions with synthetic and real test
  images (`tests/data/playfield_cam3.jpg`, `playfield_cam3_moved.jpg`).
- **Integration tests**: MCP tools produce correct results via ImageFrame
  pipeline (existing MCP tests updated).
- **Regression**: All existing tests must pass after refactor.

## Architecture Notes

- `ImageFrame` is a dataclass, not a heavy framework object. It holds ndarray
  references (no copies unless explicitly requested).
- Processing functions are pure: `f(ndarray, params) -> result`. No side effects.
- The MCP server remains the only place that resolves `source_id` to a camera
  and captures a frame. Everything below that layer is camera-agnostic.
- Playfield's `add_tag()` and flow tracking become the single source of truth
  for velocity. `AprilTagFlow.vel_px` stays but is computed by Playfield, not
  by the tag itself.

## GitHub Issues

None.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
