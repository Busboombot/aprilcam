---
sprint: "006"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 006: Multi-Camera Compositing

## What Changed

### New Module: `src/aprilcam/composite.py`

A new module containing:

- **`Composite` dataclass** -- Represents a pairing of two cameras with
  a cross-camera homography. Fields:
  - `composite_id: str` -- unique identifier (generated UUID or
    deterministic from camera IDs).
  - `primary_camera_id: str` -- the camera providing visual frames
    (typically color).
  - `secondary_camera_id: str` -- the camera providing tag detections
    (typically B&W global shutter).
  - `homography: np.ndarray` -- 3x3 matrix mapping secondary camera
    pixel coordinates to primary camera pixel coordinates.
  - `playfield_id: Optional[str]` -- optional associated playfield for
    world-coordinate mapping.
  - `reprojection_error: float` -- RMS reprojection error from the
    homography computation.

- **`CompositeManager` class** -- Manages the lifecycle of composites.
  Methods:
  - `create(primary_id, secondary_id, playfield_id?, correspondence_points?)
    -> Composite` -- Captures frames from both cameras, detects shared
    ArUco markers or uses supplied correspondences, computes the
    cross-camera homography, and stores the composite.
  - `get(composite_id) -> Composite` -- Retrieves a composite by ID.
  - `destroy(composite_id)` -- Removes a composite.
  - `list() -> List[Composite]` -- Lists active composites.

- **`compute_cross_camera_homography(primary_points, secondary_points)
  -> Tuple[np.ndarray, float]`** -- Computes the 3x3 homography mapping
  secondary pixel coords to primary pixel coords using
  `cv2.findHomography`. Returns the matrix and RMS reprojection error.

- **`map_tags_to_primary(tags, homography) -> List[MappedTag]`** --
  Transforms tag corner points and centers from secondary camera space
  to primary camera space using the homography.

- **`render_tag_overlay(frame, mapped_tags) -> np.ndarray`** -- Draws
  tag bounding boxes and IDs onto a frame at the mapped positions.
  Reuses drawing conventions from `display.py`.

### New MCP Tools (registered in MCP server entry point)

Three tools added to the MCP server tool registry:

1. **`create_composite`**
   - Inputs: `primary_camera_id` (str, required), `secondary_camera_id`
     (str, required), `playfield_id` (str, optional),
     `correspondence_points` (list of point pairs, optional).
   - Output: `{ composite_id, reprojection_error, num_correspondences }`.
   - Behavior: If `correspondence_points` is omitted, auto-detects
     ArUco 4x4 markers in both cameras. Requires at least 4 shared
     marker detections.

2. **`get_composite_frame`**
   - Inputs: `composite_id` (str, required), `format` (str, optional,
     default `"base64"`).
   - Output: Annotated color frame (base64 string or file path).
   - Behavior: Captures from both cameras, detects tags on secondary,
     maps to primary coords, renders overlays on primary frame.

3. **`get_composite_tags`**
   - Inputs: `composite_id` (str, required).
   - Output: List of tag objects with `id`, `center_px`, `corners_px`,
     `orientation_yaw`, and optional `world_xy`.
   - Behavior: Captures from secondary camera, detects tags, maps
     coordinates to primary camera space via stored homography.

### Changes to Existing Modules

- **`homography.py`** -- No changes. The existing `compute_homography`
  function maps pixels to world coordinates. The new cross-camera
  homography is a separate function in `composite.py` that maps pixels
  to pixels between two camera views. Both use `cv2.findHomography`
  internally but serve different purposes.

- **`playfield.py`** -- No changes. Playfield remains a single-camera
  abstraction. When a composite has an associated `playfield_id`, the
  playfield's world-coordinate homography is applied after the
  cross-camera coordinate mapping to provide `world_xy` values on
  mapped tags.

- **`models.py`** -- Minor addition: a `MappedTag` dataclass (or reuse
  of `AprilTag` with a `source_camera` field) to represent a tag whose
  coordinates have been transformed from one camera space to another.

- **MCP server entry point** -- Three new tool registrations added
  alongside existing tools.

## Why

The project overview (approved) lists multi-camera compositing as a
high-level requirement and Sprint 6 deliverable. The primary motivation
is the common robotics setup where a color camera provides visual context
(what does the scene look like?) while a B&W global shutter camera
provides reliable tag detection at high speeds (where are the tags?).
Without server-side compositing, agents must manage two camera sessions
and perform their own coordinate transforms, duplicating logic that
belongs in the vision server.

## Impact on Existing Components

- **Camera manager**: No interface changes. Composites reference cameras
  by ID and resolve them through the existing camera manager. Both
  cameras in a composite remain independently accessible.
- **Playfield**: No changes. A playfield can optionally be associated
  with a composite for world-coordinate enrichment, but the playfield
  itself is unaware of composites.
- **Detection loop / ring buffer**: The secondary camera's detection
  loop (if running) is used opportunistically. If no detection loop is
  active, `get_composite_frame` and `get_composite_tags` perform
  one-shot detection on the secondary camera's current frame.
- **MCP tool registry**: Three new tools are added. No existing tool
  signatures change. The tool namespace grows but remains flat.
- **Dependencies**: No new external dependencies. All functionality
  uses existing OpenCV and NumPy capabilities (`findHomography`,
  `perspectiveTransform`, `polylines`).

## Migration Concerns

None. This sprint adds new functionality without modifying existing
interfaces or data formats. No database migrations, configuration
changes, or breaking API changes are introduced.
