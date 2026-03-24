---
id: '006'
title: Multi-Camera Compositing
status: done
branch: sprint/006-multi-camera-compositing
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 006: Multi-Camera Compositing

## Goals

Add multi-camera compositing to the AprilCam MCP server so that an AI
agent can combine feeds from two cameras viewing the same playfield.
The primary use case is pairing a color camera (for visual context) with
a B&W global shutter camera (for reliable high-speed tag detection),
overlaying tag positions from the B&W camera onto the color camera's
frame.

Deliver three new MCP tools:

1. `create_composite` -- register a composite view from two cameras.
2. `get_composite_frame` -- return the color frame with tag overlay data.
3. `get_composite_tags` -- return tags detected on the secondary camera,
   mapped to primary camera coordinates.

## Problem

Single-camera setups force a trade-off: color cameras provide rich visual
context but often have rolling shutters that blur fast-moving tags, while
B&W global shutter cameras detect tags reliably at speed but lack color
information. There is currently no way to combine the strengths of both
cameras through the MCP interface. An agent that needs both reliable tag
detection and color context must manage two separate camera sessions and
perform its own coordinate mapping -- work that belongs in the server.

## Solution

Introduce a `Composite` abstraction that pairs a primary camera (color,
visual frames) with a secondary camera (B&W, tag detection). The
composite computes a cross-camera homography using shared ArUco corner
markers (IDs 0-3) visible to both cameras, or falls back to manually
supplied correspondence points. Tag detections from the secondary camera
are transformed into primary camera pixel coordinates via this homography.
The MCP server exposes the composite through three tools, and individual
cameras remain independently queryable.

## Success Criteria

- An agent can call `create_composite(primary_camera_id, secondary_camera_id)`
  and receive a `composite_id`.
- `get_composite_frame(composite_id)` returns a color frame from the
  primary camera with tag bounding boxes overlaid from the secondary
  camera's detections, correctly positioned via the cross-camera
  homography.
- `get_composite_tags(composite_id)` returns tag detections with
  positions mapped to primary camera coordinates.
- Cross-camera homography achieves sub-10px reprojection error on the
  four ArUco corner markers when both cameras can see them.
- Both cameras remain independently accessible via existing
  `capture_frame` / tag query tools.
- All new functionality has unit tests with synthetic image pairs.

## Scope

### In Scope

- `Composite` dataclass/class holding primary camera ID, secondary
  camera ID, cross-camera homography matrix, and optional playfield
  reference.
- `CompositeManager` for lifecycle management (create, query, destroy).
- Cross-camera homography computation from shared ArUco 4x4 corner
  markers detected in both cameras simultaneously.
- Manual correspondence point fallback for homography when markers are
  not visible to both cameras.
- MCP tool `create_composite(primary_camera_id, secondary_camera_id,
  playfield_id?, correspondence_points?)`.
- MCP tool `get_composite_frame(composite_id, format?)` returning the
  primary camera's frame with secondary camera tag overlays.
- MCP tool `get_composite_tags(composite_id)` returning secondary camera
  tags with positions mapped to primary camera pixel space.
- Homography re-estimation: ability to recalibrate the cross-camera
  transform on demand.
- Unit tests using synthetic image pairs (programmatically generated
  frames with known ArUco markers at known positions).

### Out of Scope

- More than two cameras in a single composite (future extension).
- Automatic camera role detection (the agent explicitly assigns
  primary vs. secondary).
- Temporal synchronization / genlock between cameras (frames are
  grabbed independently; slight timing differences are accepted).
- Blending or stitching the two camera images into a panorama.
- Lens distortion correction / intrinsic camera calibration.
- Streamable HTTP transport for composite frames.

## Test Strategy

- **Unit tests** for the `Composite` class: homography computation from
  synthetic point correspondences, tag coordinate mapping, error cases
  (insufficient points, degenerate geometry).
- **Unit tests** for `CompositeManager`: create, query, destroy
  lifecycle; duplicate composite rejection; invalid camera ID handling.
- **Integration tests** using two synthetic image sources (numpy arrays
  with programmatically placed ArUco markers) to verify end-to-end
  `create_composite` -> `get_composite_tags` pipeline.
- **MCP tool tests** verifying tool registration, input validation, and
  response schema for all three new tools.
- All tests run without physical cameras using mock/synthetic captures.

## Architecture Notes

- The `Composite` class lives in a new module `src/aprilcam/composite.py`.
- Cross-camera homography is computed with `cv2.findHomography` using
  the same approach as the existing `homography.py` module, but mapping
  secondary camera pixels to primary camera pixels (not pixels to world
  coordinates).
- The `Composite` holds references to camera IDs (not camera objects
  directly) to avoid ownership issues; it resolves cameras through the
  existing camera manager at query time.
- Tag overlay rendering reuses the annotation drawing code from
  `display.py` adapted for headless/base64 output.
- The three MCP tools are registered alongside existing camera and
  playfield tools in the MCP server entry point.

## GitHub Issues

None.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
