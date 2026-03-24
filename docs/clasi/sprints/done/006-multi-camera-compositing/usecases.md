---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 006 Use Cases

## SUC-001: Create Composite from Two Cameras with Automatic Alignment
Parent: UC-MULTI-CAM

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - Two cameras are opened and accessible via their camera IDs.
  - Both cameras can see at least four shared ArUco 4x4 corner markers
    (IDs 0-3) on the playfield.
- **Main Flow**:
  1. Agent calls `create_composite(primary_camera_id, secondary_camera_id)`.
  2. Server captures a frame from each camera.
  3. Server detects ArUco 4x4 markers (IDs 0-3) in both frames.
  4. Server computes a homography mapping secondary camera pixels to
     primary camera pixels using the matched marker centers.
  5. Server stores the composite and returns a `composite_id`.
- **Postconditions**:
  - A composite exists with a valid cross-camera homography.
  - The composite is queryable via `get_composite_frame` and
    `get_composite_tags`.
- **Acceptance Criteria**:
  - [ ] `create_composite` returns a unique `composite_id` string.
  - [ ] The cross-camera homography is computed from at least 4 shared
        ArUco marker correspondences.
  - [ ] Reprojection error on the corner markers is below 10px.
  - [ ] Error is returned if fewer than 4 shared markers are detected.

## SUC-002: Create Composite with Manual Correspondence Points
Parent: UC-MULTI-CAM

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - Two cameras are opened and accessible via their camera IDs.
  - The agent has identified corresponding points in both camera views
    (e.g., from prior frame inspection).
- **Main Flow**:
  1. Agent calls `create_composite(primary_camera_id,
     secondary_camera_id, correspondence_points=[[p1, s1], ...])` with
     at least 4 point pairs, each pair being a primary pixel coordinate
     and its corresponding secondary pixel coordinate.
  2. Server computes the homography from the supplied correspondences.
  3. Server stores the composite and returns a `composite_id`.
- **Postconditions**:
  - A composite exists with a homography derived from the manual points.
- **Acceptance Criteria**:
  - [ ] `create_composite` accepts a `correspondence_points` parameter.
  - [ ] Homography is computed from the supplied points without requiring
        ArUco marker detection.
  - [ ] Error is returned if fewer than 4 point pairs are provided.
  - [ ] Error is returned if the points produce a degenerate homography.

## SUC-003: Get Composite Frame with Tag Overlays
Parent: UC-MULTI-CAM

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - A composite has been created via `create_composite`.
  - Both cameras are still accessible.
- **Main Flow**:
  1. Agent calls `get_composite_frame(composite_id, format="base64")`.
  2. Server captures a frame from the primary (color) camera.
  3. Server captures a frame from the secondary (B&W) camera and runs
     tag detection on it.
  4. Server maps detected tag positions from secondary camera coordinates
     to primary camera coordinates using the stored homography.
  5. Server draws tag bounding boxes and IDs onto the primary camera
     frame at the mapped positions.
  6. Server encodes the annotated frame and returns it in the requested
     format (base64 or file path).
- **Postconditions**:
  - The agent receives a color frame showing tag positions detected by
    the secondary camera, correctly positioned in the primary camera's
    view.
- **Acceptance Criteria**:
  - [ ] Returned frame is from the primary (color) camera.
  - [ ] Tag overlays reflect detections from the secondary camera.
  - [ ] Tag bounding box positions are correctly transformed via the
        cross-camera homography.
  - [ ] Both `base64` and `file` format options work.
  - [ ] Error is returned if the composite ID is invalid.

## SUC-004: Get Composite Tags in Primary Camera Coordinates
Parent: UC-MULTI-CAM

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - A composite has been created via `create_composite`.
  - The secondary camera can detect at least one tag.
- **Main Flow**:
  1. Agent calls `get_composite_tags(composite_id)`.
  2. Server captures a frame from the secondary camera and runs tag
     detection.
  3. Server transforms each detected tag's corner points and center
     from secondary camera pixel coordinates to primary camera pixel
     coordinates using the stored homography.
  4. Server returns the list of tags with mapped coordinates, including
     tag ID, mapped center, mapped corners, and orientation.
- **Postconditions**:
  - The agent receives structured tag data in primary camera coordinates.
- **Acceptance Criteria**:
  - [ ] Each tag includes: id, center_px (in primary coords),
        corners_px (in primary coords), orientation_yaw.
  - [ ] Positions are correctly transformed via the homography.
  - [ ] World coordinates are included if a playfield with calibrated
        homography is associated.
  - [ ] Empty list is returned when no tags are detected (not an error).

## SUC-005: Query Individual Cameras While Composite Exists
Parent: UC-MULTI-CAM

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - A composite has been created from two cameras.
- **Main Flow**:
  1. Agent calls existing camera tools (`capture_frame`,
     `get_current_tags`, etc.) using either the primary or secondary
     camera ID directly.
  2. Server processes the request against the individual camera as
     usual, independent of the composite.
- **Postconditions**:
  - The agent receives the individual camera's response, unaffected by
    the composite configuration.
- **Acceptance Criteria**:
  - [ ] `capture_frame(primary_camera_id)` returns the raw color frame
        without composite overlays.
  - [ ] `capture_frame(secondary_camera_id)` returns the raw B&W frame.
  - [ ] Tag detection on an individual camera returns tags in that
        camera's own coordinate space.
  - [ ] Creating or destroying a composite does not disrupt individual
        camera access.
