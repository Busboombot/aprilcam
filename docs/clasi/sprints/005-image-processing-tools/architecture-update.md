---
sprint: "005"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 005: Image Processing Tools

## What Changed

### New Module: `src/aprilcam/image_processing.py`

A new module containing pure-CV image processing functions, one per
tool. Each function takes a BGR numpy array (and tool-specific
parameters) and returns structured results. Functions:

- `process_detect_lines(frame, threshold?, min_length?, max_gap?)`
  -- Grayscale conversion, Canny edges, HoughLinesP. Returns list
  of `(x1, y1, x2, y2)` tuples.

- `process_detect_circles(frame, min_radius?, max_radius?, param1?, param2?)`
  -- Grayscale conversion, HoughCircles. Returns list of
  `{center: (x, y), radius: r}` dicts.

- `process_detect_contours(frame, min_area?)`
  -- Grayscale, adaptive threshold, findContours, approxPolyDP.
  Returns list of `{points, area, bbox}` dicts.

- `process_detect_motion(frame, prev_frame)`
  -- Absolute difference, threshold, findContours. Returns list of
  `{bbox, area}` dicts. Caller manages previous-frame storage.

- `process_detect_qr_codes(frame)`
  -- QRCodeDetector.detectAndDecodeMulti. Returns list of
  `{data, corners}` dicts.

- `process_crop_region(frame, x, y, w, h)`
  -- Array slice with bounds clipping. Returns the sub-image.

- `process_apply_transform(frame, operation, params)`
  -- Dispatcher for rotate/scale/threshold/canny/blur. Returns the
  transformed image.

### New Utility: Source Resolution

A `resolve_source(source_id, server_state)` function (located in the
MCP server module or a shared utility) that:

1. Checks the camera registry for `source_id`. If found, captures a
   frame from that camera.
2. Checks the playfield registry for `source_id`. If found, captures
   a frame from the playfield's underlying camera and applies the
   playfield's deskew/homography warp.
3. Raises a descriptive error if `source_id` is not found in either
   registry.

This centralizes frame acquisition logic that would otherwise be
duplicated across all eight tools.

### New Utility: Image Output Formatting

A `format_image_output(frame, format, prefix?)` function that:

- If `format == "base64"`: encodes the frame as PNG via
  `cv2.imencode`, then base64-encodes the bytes. Returns a dict
  with `{format: "base64", data: "..."}`.
- If `format == "file"`: writes the frame to a temp file using
  `tempfile.NamedTemporaryFile(suffix=".png", delete=False)`.
  Returns `{format: "file", path: "/tmp/..."}`.

### Server State Addition: Motion Previous Frames

The MCP server state gains a `_motion_prev_frames: Dict[str, np.ndarray]`
dictionary. Keyed by `source_id`, it stores the grayscale version of
the last frame used in a `detect_motion` call. This is managed by the
`detect_motion` tool handler, not the processing function itself.

### Eight New MCP Tool Registrations

Each tool is registered with the MCP server's tool registry following
the same pattern as existing tools (e.g., `list_cameras`,
`open_camera`, `capture_frame`). Each handler:

1. Validates input parameters.
2. Calls `resolve_source` to get a frame.
3. Calls the corresponding `process_*` function.
4. Formats and returns the response (JSON for detection results,
   image output for frame-returning tools).

## Why

The project overview specifies "Image Processing Tools" as a core
requirement. AI agents need to analyze visual features beyond fiducial
markers to reason about playfield state -- for example, detecting game
board lines, identifying circular tokens, reading QR codes on objects,
or tracking motion. Sprint 5 delivers this capability as eight focused
MCP tools, each performing a single well-defined operation.

This follows the design principle of the existing server: each tool
does one thing, accepts a source_id, and returns structured data. The
agent composes these tools as needed for its task.

## Impact on Existing Components

### MCP Server (`aprilcam mcp`)

- **Tool registry**: Eight new tools are added. No existing tools are
  modified.
- **Server state**: Gains `_motion_prev_frames` dict. No existing
  state is changed.
- **Dependencies**: No new external dependencies. All processing uses
  OpenCV functions already available via `opencv-contrib-python`.

### Camera and Playfield Modules

- **No changes** to `camutil.py`, `playfield.py`, `aprilcam.py`, or
  `models.py`. The new tools read frames through the existing
  capture interfaces.
- The source resolver depends on the camera and playfield registries
  but does not modify them.

### CLI

- **No changes**. The image processing tools are MCP-only; they have
  no CLI subcommands.

### Image Return Format

- The `format_image_output` utility follows the same base64/file
  convention established in Sprint 2 for `capture_frame`. Consistent
  across all image-returning tools.

## Migration Concerns

None. This sprint adds new modules and tools without modifying any
existing interfaces. No data migration is required. The new tools are
purely additive and do not affect backward compatibility.
