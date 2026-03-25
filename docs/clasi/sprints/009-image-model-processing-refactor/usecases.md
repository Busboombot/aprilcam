---
status: draft
---

# Sprint 009 Use Cases

## SUC-001: Process a Static Image Without a Camera
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: Developer / Test harness
- **Preconditions**: A JPEG/PNG image file exists on disk
- **Main Flow**:
  1. Load image from disk as a NumPy array
  2. Construct an `ImageFrame` with the raw array and source metadata
  3. Run tag detection on the ImageFrame's raw array
  4. Populated ImageFrame now holds detected ArUco corners and AprilTags
  5. Run additional processing (detect_lines, detect_contours, etc.) on
     the raw array — returns structured results
- **Postconditions**: All detection and processing results are available
  without any camera having been opened
- **Acceptance Criteria**:
  - [ ] `ImageFrame` can be constructed from a file-loaded ndarray
  - [ ] Tag detection works on ImageFrame without a camera handle
  - [ ] Image processing functions accept ndarray, not source_id

## SUC-002: MCP Tool Uses ImageFrame Internally
Parent: UC-003, UC-005 (Image Processing Tools)

- **Actor**: AI agent (via MCP)
- **Preconditions**: Camera is open, MCP server running
- **Main Flow**:
  1. Agent calls `detect_lines(source_id="cam_0")`
  2. MCP server resolves source_id to camera, captures frame
  3. Server constructs an `ImageFrame` from the captured ndarray
  4. Server calls `process_detect_lines(frame.raw)` with the raw array
  5. Server returns structured line data to agent
- **Postconditions**: MCP external API unchanged; internal pipeline uses
  ImageFrame and array-based processing
- **Acceptance Criteria**:
  - [ ] MCP tools resolve source_id to ImageFrame early in the call
  - [ ] Processing functions receive ndarray from ImageFrame
  - [ ] External MCP API unchanged (same params, same response shape)

## SUC-003: Playfield Computes Tag Velocity From History
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: Detection loop / Playfield
- **Preconditions**: Playfield created, detection loop running
- **Main Flow**:
  1. Detection loop captures a frame, runs tag detection
  2. Detected tags (position, orientation) are passed to Playfield
  3. Playfield adds tags to its flow history
  4. Playfield computes velocity from the last N positions in history
  5. Agent queries `get_tags()` — response includes velocity from Playfield
- **Postconditions**: Velocity is computed by Playfield from positional
  history, not by individual tag objects or per-frame processing
- **Acceptance Criteria**:
  - [ ] Playfield owns velocity computation
  - [ ] Individual tag detection results carry position but not velocity
  - [ ] `get_tags()` returns velocity sourced from Playfield flow history

## SUC-004: Test Detection With Static Test Images
Parent: UC-003

- **Actor**: Test harness (pytest)
- **Preconditions**: Test images exist in `tests/data/`
- **Main Flow**:
  1. Test loads `playfield_cam3.jpg` as ndarray
  2. Test creates ImageFrame from the array
  3. Test runs ArUco corner detection on the array
  4. Test runs AprilTag detection on the array
  5. Test asserts expected tags are found at expected positions
  6. For velocity testing: test feeds two frames (original + moved) to
     Playfield sequentially, asserts velocity is computed
- **Postconditions**: Full detection pipeline tested without camera hardware
- **Acceptance Criteria**:
  - [ ] Tests use static images from `tests/data/`
  - [ ] No camera fixture or mock needed for detection tests
  - [ ] Velocity test uses two-frame sequence through Playfield
