---
id: '005'
title: Image Processing Tools
status: done
branch: sprint/005-image-processing-tools
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
- SUC-006
- SUC-007
- SUC-008
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 005: Image Processing Tools

## Goals

Implement eight MCP tools that perform image processing operations on
live frames from cameras and playfields. These tools give AI agents
the ability to analyze visual features beyond fiducial markers --
lines, circles, contours, motion, QR codes -- and to capture, crop,
and transform frames programmatically.

## Problem

After Sprints 2-4, the MCP server can manage cameras, establish
playfields with homography, and run tag detection loops. However, AI
agents have no way to perform general-purpose image analysis on live
frames. An agent that needs to find lines on a game board, detect
circular tokens, identify QR codes on objects, track motion between
frames, or simply grab a raw frame for its own analysis has no tools
available. The agent is limited to fiducial marker data only.

## Solution

Add eight new MCP tools to the AprilCam server, all sharing a common
`source_id` parameter that accepts either a `camera_id` or
`playfield_id`. Each tool captures a live frame from the specified
source, performs its processing using OpenCV, and returns structured
results (JSON) and/or images (base64 or file path, caller's choice).

A new `image_processing.py` module in `src/aprilcam/` will contain
the pure-CV logic for each operation. A `source_resolver` utility
will centralize the pattern of resolving a `source_id` to a frame,
so all tools share the same lookup and error handling. The MCP tool
handlers will be thin wrappers that resolve the source, call the
processing function, and format the response.

## Success Criteria

- All eight tools are registered in the MCP server and callable.
- Each tool accepts `source_id` that works with both camera and
  playfield sources.
- Image-returning tools support both `base64` and `file` formats.
- Each tool returns well-structured JSON results.
- Unit tests cover the CV processing functions with synthetic images.
- Integration tests verify end-to-end tool invocation through MCP.
- `detect_motion` correctly maintains per-source previous-frame state.

## Scope

### In Scope

- `get_frame(source_id, format?)` -- raw frame capture, no processing.
- `detect_lines(source_id, ...)` -- Hough line detection returning
  line segments as `[(x1, y1, x2, y2)]`.
- `detect_circles(source_id, ...)` -- Hough circle detection returning
  `[{center, radius}]`.
- `detect_contours(source_id, min_area?, ...)` -- contour detection
  with optional area filtering, returning contour polygons.
- `detect_motion(source_id)` -- frame differencing between current and
  previous frame, returning motion mask regions.
- `detect_qr_codes(source_id)` -- QR code detection and decoding,
  returning `[{data, corners}]`.
- `crop_region(source_id, x, y, w, h, format?)` -- crop a rectangular
  region, return sub-image as base64 or file.
- `apply_transform(source_id, operation, params?, format?)` -- apply
  one of: rotate, scale, threshold, edge-detect (Canny), blur.
- Source resolution utility (camera_id or playfield_id to frame).
- Image output formatting utility (base64 or temp file).
- Unit tests for all CV processing functions.
- Integration tests for MCP tool registration and invocation.

### Out of Scope

- Processing arbitrary image files from disk (tools operate on live
  frames only, per project spec).
- Video recording or streaming.
- ML-based detection (object detection, segmentation, OCR beyond QR).
- Batch processing of multiple frames in a single tool call.
- Chaining or pipelining multiple operations in one request.
- GPU acceleration or CUDA-specific code paths.

## Test Strategy

**Unit tests** (`tests/test_image_processing.py`): Test each CV
processing function in isolation using synthetic images generated
with OpenCV drawing functions. For example, draw known lines on a
blank image and verify `detect_lines` returns segments near those
coordinates. Draw circles and verify `detect_circles` returns
matching centers and radii. Tests for `detect_motion` will use two
frames with known differences.

**Integration tests** (`tests/test_mcp_image_tools.py`): Verify that
each tool is registered with the MCP server, accepts the documented
parameters, and returns the expected response schema. Use a mock
camera source that returns synthetic frames.

**Manual testing**: Verify each tool against a real camera or
playfield to confirm end-to-end behavior with real-world images.

## Architecture Notes

- **Source resolution**: A `resolve_source(source_id)` function will
  look up the source in the server's camera registry and playfield
  registry, grab a frame, and return the BGR numpy array. This
  avoids duplicating lookup logic across eight tools.

- **Image output**: A shared `format_image_output(frame, format)`
  utility encodes the frame as base64 PNG or writes to a temp file
  and returns the path. Reused by `get_frame`, `crop_region`, and
  `apply_transform`.

- **Motion state**: `detect_motion` needs to remember the previous
  frame per source. Store this in a dict keyed by `source_id` in
  the server state. First call for a source returns empty motion
  (no previous frame to compare).

- **QR detection**: Use OpenCV's `QRCodeDetector` (available in
  opencv-contrib). No additional dependencies needed.

- **Parameter validation**: Each tool validates its parameters and
  returns clear error messages (e.g., invalid source_id, crop region
  out of bounds, unknown transform operation).

## GitHub Issues

None.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
