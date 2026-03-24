---
timestamp: '2026-03-24T04:20:55'
parent: team-lead
child: technical-lead
scope: /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/005-image-processing-tools
sprint: 005-image-processing-tools
context_documents:
- docs/clasi/sprints/005-image-processing-tools/sprint.md
- docs/clasi/sprints/005-image-processing-tools/architecture-update.md
- docs/clasi/sprints/005-image-processing-tools/usecases.md
result: "success \u2014 6 tickets created"
files_modified:
- docs/clasi/sprints/005-image-processing-tools/tickets/001-source-resolver-and-image-output-utilities.md
- docs/clasi/sprints/005-image-processing-tools/tickets/002-get-frame-and-crop-region-mcp-tools.md
- docs/clasi/sprints/005-image-processing-tools/tickets/003-detect-lines-and-detect-circles-mcp-tools.md
- docs/clasi/sprints/005-image-processing-tools/tickets/004-detect-contours-and-detect-motion-mcp-tools.md
- docs/clasi/sprints/005-image-processing-tools/tickets/005-detect-qr-codes-and-apply-transform-mcp-tools.md
- docs/clasi/sprints/005-image-processing-tools/tickets/006-integration-tests.md
---

# Dispatch: team-lead → technical-lead

Create tickets for Sprint 005: Image Processing Tools. Use `create_ticket` MCP tool with sprint_id="005".

## Ticket Breakdown

### Ticket 001: Source resolver and image output utilities
- Add `resolve_source(source_id)` function to mcp_server.py — resolves camera_id or playfield_id to a BGR frame
- Add `format_image_output(frame, format, quality)` function — encodes as base64 ImageContent or temp file TextContent
- These are shared utilities used by all 8 tools
- Use cases: all (foundation)
- Tests: unit tests for both utilities

### Ticket 002: get_frame and crop_region MCP tools
- `get_frame(source_id, format?)` — raw frame capture
- `crop_region(source_id, x, y, w, h, format?)` — crop and return sub-image
- Depends on: 001
- Use cases: SUC-001, SUC-007

### Ticket 003: detect_lines and detect_circles MCP tools  
- `detect_lines(source_id, threshold?, min_length?, max_gap?)` — Hough lines
- `detect_circles(source_id, min_radius?, max_radius?, param1?, param2?)` — Hough circles
- Add processing functions to `src/aprilcam/image_processing.py` (new file)
- Depends on: 001
- Use cases: SUC-002, SUC-003

### Ticket 004: detect_contours and detect_motion MCP tools
- `detect_contours(source_id, min_area?)` — contour detection
- `detect_motion(source_id)` — frame differencing with per-source state
- Add `_motion_prev_frames` dict to server state
- Depends on: 001
- Use cases: SUC-004, SUC-005

### Ticket 005: detect_qr_codes and apply_transform MCP tools
- `detect_qr_codes(source_id)` — QR detection and decoding
- `apply_transform(source_id, operation, params?, format?)` — rotate/scale/threshold/canny/blur
- Depends on: 001
- Use cases: SUC-006, SUC-008

### Ticket 006: Integration tests
- End-to-end MCP tool tests for all 8 tools using mock camera
- Depends on: 002, 003, 004, 005
- Use cases: all

For each ticket write substantive descriptions, acceptance criteria as checkboxes, and testing sections.

## Context Documents

- `docs/clasi/sprints/005-image-processing-tools/sprint.md`
- `docs/clasi/sprints/005-image-processing-tools/architecture-update.md`
- `docs/clasi/sprints/005-image-processing-tools/usecases.md`
