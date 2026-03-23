---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 005 Use Cases

## SUC-001: Capture Raw Frame
Parent: Image Processing Tools (overview section "Image Processing Tools")

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open and has an active
  video feed.
- **Main Flow**:
  1. Agent calls `get_frame(source_id, format="base64")`.
  2. Server resolves `source_id` to a camera or playfield.
  3. Server captures a single frame from the source.
  4. Server encodes the frame as base64 PNG and returns it.
- **Postconditions**: Agent has a raw frame image with no processing
  applied. The camera/playfield state is unchanged.
- **Alternate Flow**:
  - 1a. Agent specifies `format="file"`. Server writes the frame to a
    temp file and returns the file path instead of base64.
  - 2a. `source_id` is invalid. Server returns an error with a
    descriptive message.
- **Acceptance Criteria**:
  - [ ] `get_frame` accepts both camera_id and playfield_id as source_id
  - [ ] Returns base64-encoded PNG when format is "base64" or omitted
  - [ ] Returns a file path to a valid PNG when format is "file"
  - [ ] Returns an error for an invalid or closed source_id

## SUC-002: Detect Lines in Frame
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `detect_lines(source_id)` with optional Hough
     parameters (threshold, min_length, max_gap).
  2. Server captures a frame, converts to grayscale, runs Canny edge
     detection, then Hough line transform.
  3. Server returns a list of line segments as
     `[{x1, y1, x2, y2}, ...]`.
- **Postconditions**: Agent has a list of detected line segments.
  Camera/playfield state is unchanged.
- **Alternate Flow**:
  - 3a. No lines detected. Server returns an empty list.
- **Acceptance Criteria**:
  - [ ] Returns line segments as list of {x1, y1, x2, y2} dicts
  - [ ] Optional parameters (threshold, min_length, max_gap) are
        respected when provided
  - [ ] Returns empty list when no lines are found
  - [ ] Works with both camera and playfield sources

## SUC-003: Detect Circles in Frame
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `detect_circles(source_id)` with optional parameters
     (min_radius, max_radius, param1, param2).
  2. Server captures a frame, converts to grayscale, runs Hough circle
     transform.
  3. Server returns a list of circles as
     `[{center: {x, y}, radius}, ...]`.
- **Postconditions**: Agent has a list of detected circles.
- **Alternate Flow**:
  - 3a. No circles detected. Server returns an empty list.
- **Acceptance Criteria**:
  - [ ] Returns circles as list of {center: {x, y}, radius} dicts
  - [ ] Optional parameters constrain detection (e.g., radius range)
  - [ ] Returns empty list when no circles are found
  - [ ] Works with both camera and playfield sources

## SUC-004: Detect Contours in Frame
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `detect_contours(source_id)` with optional min_area
     filter.
  2. Server captures a frame, converts to grayscale, applies threshold,
     finds contours.
  3. Server filters contours by area if min_area is specified.
  4. Server approximates each contour to a polygon and returns a list
     of polygons, each as a list of {x, y} points, along with area
     and bounding box.
- **Postconditions**: Agent has a list of detected contour polygons.
- **Alternate Flow**:
  - 4a. No contours pass the filter. Server returns an empty list.
- **Acceptance Criteria**:
  - [ ] Returns contours as list of {points, area, bbox} dicts
  - [ ] min_area parameter filters out small contours
  - [ ] Returns empty list when no contours pass the filter
  - [ ] Works with both camera and playfield sources

## SUC-005: Detect Motion Between Frames
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `detect_motion(source_id)`.
  2. Server captures the current frame.
  3. Server computes the absolute difference between the current frame
     and the previous frame stored for this source.
  4. Server thresholds the difference to produce a motion mask.
  5. Server finds contours in the motion mask and returns a list of
     motion regions as bounding boxes with area.
  6. Server stores the current frame as the new "previous frame" for
     this source.
- **Postconditions**: Agent has a list of motion regions. The server
  has updated its stored previous frame for the source.
- **Alternate Flow**:
  - 3a. No previous frame exists (first call for this source). Server
    stores the current frame, returns an empty motion list with a
    message indicating this is the baseline frame.
- **Acceptance Criteria**:
  - [ ] Returns motion regions as list of {bbox, area} dicts
  - [ ] First call for a source returns empty list (baseline)
  - [ ] Subsequent calls detect actual motion between frames
  - [ ] Per-source state is maintained independently
  - [ ] Works with both camera and playfield sources

## SUC-006: Detect and Decode QR Codes
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `detect_qr_codes(source_id)`.
  2. Server captures a frame and runs OpenCV QRCodeDetector.
  3. Server returns a list of QR codes as
     `[{data, corners: [{x, y}, ...]}, ...]`.
- **Postconditions**: Agent has decoded QR code data and their
  locations in the frame.
- **Alternate Flow**:
  - 3a. No QR codes detected. Server returns an empty list.
- **Acceptance Criteria**:
  - [ ] Returns QR codes as list of {data, corners} dicts
  - [ ] data field contains the decoded string content
  - [ ] corners field contains the four corner points of each QR code
  - [ ] Returns empty list when no QR codes are found
  - [ ] Works with both camera and playfield sources

## SUC-007: Crop Region from Frame
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `crop_region(source_id, x, y, w, h, format="base64")`.
  2. Server captures a frame.
  3. Server validates that the crop rectangle is within frame bounds
     (clipping to frame edges if partially out of bounds).
  4. Server extracts the sub-image and returns it in the requested
     format.
- **Postconditions**: Agent has the cropped sub-image.
- **Alternate Flow**:
  - 3a. Crop rectangle is entirely outside frame bounds. Server returns
    an error.
  - 1a. Agent specifies `format="file"`. Server writes the cropped
    image to a temp file and returns the path.
- **Acceptance Criteria**:
  - [ ] Returns cropped image in base64 or file format
  - [ ] Crop coordinates (x, y, w, h) correctly define the region
  - [ ] Partially out-of-bounds regions are clipped to frame edges
  - [ ] Entirely out-of-bounds regions produce an error
  - [ ] Works with both camera and playfield sources

## SUC-008: Apply Image Transform
Parent: Image Processing Tools

- **Actor**: AI agent (via MCP)
- **Preconditions**: A camera or playfield is open with an active feed.
- **Main Flow**:
  1. Agent calls `apply_transform(source_id, operation="canny",
     params={...}, format="base64")`.
  2. Server captures a frame.
  3. Server applies the requested operation with the given parameters.
  4. Server returns the transformed image in the requested format.
- **Postconditions**: Agent has the transformed image.
- **Supported Operations**:
  - `rotate` -- params: {angle} (degrees, counterclockwise)
  - `scale` -- params: {factor} (e.g., 0.5 for half size)
  - `threshold` -- params: {value} (0-255, binary threshold)
  - `canny` -- params: {low, high} (Canny edge thresholds)
  - `blur` -- params: {kernel_size} (Gaussian blur kernel, odd integer)
- **Alternate Flow**:
  - 1a. Unknown operation name. Server returns an error listing the
    supported operations.
  - 1b. Missing or invalid params for the operation. Server returns
    an error describing the required params.
- **Acceptance Criteria**:
  - [ ] All five operations (rotate, scale, threshold, canny, blur)
        are supported
  - [ ] Each operation respects its params
  - [ ] Unknown operations produce a clear error message
  - [ ] Returns transformed image in base64 or file format
  - [ ] Works with both camera and playfield sources
