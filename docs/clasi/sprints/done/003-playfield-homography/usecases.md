---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 003 Use Cases

## SUC-001: Create Playfield from Camera
Parent: UC-003 (Playfield Setup)

- **Actor**: LLM agent (MCP client)
- **Preconditions**:
  - A camera is open and registered (`camera_id` exists).
  - Four ArUco 4x4 markers (IDs 0, 1, 2, 3) are visible in the camera's
    field of view, placed at the corners of the playing surface.
- **Main Flow**:
  1. Agent calls `create_playfield(camera_id="cam_0")`.
  2. Server captures one or more frames from the camera.
  3. Server runs ArUco 4x4 detection on each frame, accumulating
     corner marker positions (IDs 0-3).
  4. Once all four corners are found, server orders them geometrically
     into UL, UR, LR, LL using `_order_poly()`.
  5. Server creates a `Playfield` instance with the locked polygon.
  6. Server registers the playfield in `PlayfieldRegistry` under
     `playfield_id = "pf_cam_0"`.
  7. Server returns `{playfield_id: "pf_cam_0", corners: [[x,y],...],
     calibrated: false}`.
- **Alternative Flow**:
  - 3a. Fewer than four markers detected after max frame attempts:
    server returns an error with the list of missing corner IDs.
- **Postconditions**:
  - A playfield is registered and associated with the camera.
  - The playfield polygon is locked (will not change on subsequent frames).
- **Acceptance Criteria**:
  - [ ] `create_playfield` returns a valid `playfield_id` when all 4 markers are visible.
  - [ ] `create_playfield` returns an error listing missing IDs when fewer than 4 markers are found.
  - [ ] The returned corners are in UL, UR, LR, LL order.
  - [ ] Calling `create_playfield` twice for the same camera returns the same `playfield_id` (or replaces the old one).

## SUC-002: Capture Deskewed Image via Playfield
Parent: UC-003 (Playfield Setup)

- **Actor**: LLM agent (MCP client)
- **Preconditions**:
  - A playfield has been created (`playfield_id` exists in registry).
  - The underlying camera is still open.
- **Main Flow**:
  1. Agent calls a capture tool (e.g., `capture_image`) passing
     `playfield_id` instead of `camera_id`.
  2. Server resolves the playfield_id to its underlying camera.
  3. Server captures a raw frame from the camera.
  4. Server applies `Playfield.deskew()` to produce a top-down
     rectangular view using pixel-only perspective transform.
  5. Server returns the deskewed image (base64 PNG or file path).
- **Alternative Flow**:
  - 1a. Agent passes an unknown `playfield_id`: server returns
    "unknown playfield" error.
  - 3a. Camera read fails: server returns camera error.
- **Postconditions**:
  - The returned image is a perspective-corrected rectangle whose
    dimensions match the pixel distances between the detected corners.
- **Acceptance Criteria**:
  - [ ] Passing a `playfield_id` to a capture tool returns a deskewed image.
  - [ ] The deskewed image dimensions are derived from corner distances (not the raw camera resolution).
  - [ ] An unknown `playfield_id` produces a clear error message.

## SUC-003: Calibrate Playfield with Real-World Measurements
Parent: UC-004 (Homography Calibration)

- **Actor**: LLM agent (MCP client)
- **Preconditions**:
  - A playfield exists (`playfield_id` in registry) with a locked polygon.
  - The agent knows the physical dimensions of the playing surface.
- **Main Flow**:
  1. Agent calls `calibrate_playfield(playfield_id="pf_cam_0",
     measurements={width: 40, height: 35, units: "inch"})`.
  2. Server retrieves the playfield's pixel corner positions.
  3. Server builds world-coordinate correspondences:
     UL=(0,0), UR=(width_cm,0), LR=(width_cm,height_cm), LL=(0,height_cm).
  4. Server calls `compute_homography(pixel_pts, world_pts_cm)` to
     get the 3x3 homography matrix.
  5. Server stores the homography matrix and field spec with the
     playfield entry.
  6. Server returns `{playfield_id, calibrated: true,
     width_cm: 101.6, height_cm: 88.9, units: "cm"}`.
- **Alternative Flow**:
  - 1a. Invalid playfield_id: error "unknown playfield".
  - 1b. Missing or invalid measurements: error listing required fields.
  - 4a. Homography computation fails (degenerate points): error with
    details.
- **Postconditions**:
  - The playfield entry has a homography matrix.
  - Subsequent tag detections through this playfield can map pixel
    positions to world coordinates.
- **Acceptance Criteria**:
  - [ ] After calibration, `get_playfield_info` shows `calibrated: true`.
  - [ ] The stored homography correctly maps the four corner pixels to the expected world coordinates (within 0.1 cm tolerance).
  - [ ] Unit conversion from inches to centimeters is correct.
  - [ ] Calling `calibrate_playfield` again overwrites the previous calibration.

## SUC-004: Query Playfield Information
Parent: UC-003 (Playfield Setup)

- **Actor**: LLM agent (MCP client)
- **Preconditions**:
  - A playfield exists in the registry.
- **Main Flow**:
  1. Agent calls `get_playfield_info(playfield_id="pf_cam_0")`.
  2. Server looks up the playfield entry.
  3. Server returns a JSON object containing:
     - `playfield_id`
     - `camera_id` (underlying camera)
     - `corners` (4x2 array in UL, UR, LR, LL order, pixel coords)
     - `calibrated` (boolean)
     - `width_cm` and `height_cm` (present only if calibrated)
     - `homography` (3x3 matrix, present only if calibrated)
- **Alternative Flow**:
  - 2a. Unknown playfield_id: error "unknown playfield".
- **Postconditions**:
  - No state change. Read-only operation.
- **Acceptance Criteria**:
  - [ ] Returns all specified fields for a calibrated playfield.
  - [ ] Returns corners and `calibrated: false` for an uncalibrated playfield.
  - [ ] Returns error for unknown playfield_id.

## SUC-005: Refactor Playfield and Homography for Library Use
Parent: UC-003, UC-004

- **Actor**: Developer (internal)
- **Preconditions**:
  - `playfield.py` and `homography.py` exist with CLI-oriented code.
- **Main Flow**:
  1. Extract `homography.py`'s `main()` calibration logic into a
     function `calibrate_from_frame(frame, field_spec) -> (corners, H)`
     that takes a BGR frame and a `FieldSpec`, returns the corner
     positions and homography matrix.
  2. Ensure `Playfield._detect_corners()` and `Playfield._order_poly()`
     are usable as standalone functions (or well-documented public
     methods) for the MCP tool handlers.
  3. Move `CORNER_ID_MAP` and `FieldSpec` to a shared location or
     keep in `homography.py` with clean imports.
  4. Existing CLI entry points (`homocal`, `playfield` CLI) continue
     to work by calling the refactored library functions.
- **Postconditions**:
  - Core detection and calibration logic is importable without CLI or
    GUI dependencies.
  - CLI tools are thin wrappers around the library functions.
- **Acceptance Criteria**:
  - [ ] `Playfield.update()` and `Playfield.deskew()` work when called from MCP tool handlers (no argparse, no cv.imshow).
  - [ ] A new function in `homography.py` computes calibration from a frame and FieldSpec without file I/O.
  - [ ] Existing CLI commands (`homocal`, `playfield`) pass their current test cases.
  - [ ] No circular imports between `playfield.py`, `homography.py`, and the MCP server module.
