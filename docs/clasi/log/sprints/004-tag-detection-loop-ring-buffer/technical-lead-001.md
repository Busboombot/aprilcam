---
timestamp: '2026-03-24T02:08:06'
parent: team-lead
child: technical-lead
scope: /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/004-tag-detection-loop-ring-buffer
sprint: 004-tag-detection-loop-ring-buffer
context_documents:
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/sprint.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/architecture-update.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/usecases.md
result: "success \u2014 7 tickets created with dependency ordering"
files_modified:
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/001-tagrecord-and-framerecord-dataclasses.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/002-ringbuffer-class.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/003-refactor-aprilcam-extract-process-frame.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/004-detectionloop-class.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/005-mcp-tools-start-detection-and-stop-detection.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/006-mcp-tools-get-tags-and-get-tag-history.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/tickets/007-integration-tests.md
---

# Dispatch: team-lead → technical-lead

You are the technical-lead agent. Create implementation tickets for Sprint 004: Tag Detection Loop &amp; Ring Buffer.

Use the `create_ticket` MCP tool for each ticket with sprint_id="004".

## Sprint Architecture Summary

The sprint adds:
1. `TagRecord` and `FrameRecord` dataclasses in `src/aprilcam/detection.py`
2. `RingBuffer` class (thread-safe deque, maxlen=300) in `src/aprilcam/detection.py`
3. Refactor `AprilCam.run()` to extract `process_frame()` method
4. `DetectionLoop` class (background thread) in `src/aprilcam/detection.py`
5. Four MCP tools: `start_detection`, `stop_detection`, `get_tags`, `get_tag_history`
6. Unit and integration tests

## Use Cases

- SUC-001: Start a Detection Loop
- SUC-002: Stop a Detection Loop
- SUC-003: Query Current Tag State
- SUC-004: Query Tag History
- SUC-005: Refactored CLI Detection

## Ticket Breakdown

Create these tickets in dependency order:

### Ticket 001: TagRecord and FrameRecord dataclasses
- Add `TagRecord` and `FrameRecord` to `src/aprilcam/detection.py`
- TagRecord fields: id, center_px, corners_px, orientation_yaw, world_xy, in_playfield, vel_px, speed_px, vel_world, speed_world, heading_rad, timestamp, frame_index
- FrameRecord fields: timestamp, frame_index, tags (list of TagRecord)
- Both have `to_dict()` for JSON serialization
- Use cases: SUC-003, SUC-004
- Tests: construction, serialization, round-trip

### Ticket 002: RingBuffer class
- Thread-safe circular buffer in `src/aprilcam/detection.py`
- Backed by `collections.deque(maxlen=300)`, protected by `threading.Lock`
- Methods: append, get_latest, get_last_n, clear, __len__
- Depends on: 001
- Use cases: SUC-003, SUC-004
- Tests: append, overflow, get_latest, get_last_n edge cases, thread safety

### Ticket 003: Refactor AprilCam — extract process_frame()
- Extract detection/tracking core from `run()` into `process_frame(frame_bgr, timestamp) -> List[TagRecord]`
- Move local state (prev_gray, tracks, tag_models, etc.) to instance attributes
- Add `reset_state()` method
- `run()` calls `process_frame()` internally — no behavior change
- Depends on: 001
- Use cases: SUC-005
- Tests: verify process_frame returns TagRecords from test images, existing CLI behavior preserved

### Ticket 004: DetectionLoop class
- Background thread that grabs frames, calls `process_frame()`, writes to RingBuffer
- Constructor: source (VideoCapture or playfield), aprilcam, ring_buffer
- start() / stop() lifecycle, threading.Event for stop signal
- is_running property, per-frame exception catching
- Depends on: 002, 003
- Use cases: SUC-001, SUC-002
- Tests: start/stop lifecycle, frame processing with mock capture, thread safety

### Ticket 005: MCP tools — start_detection and stop_detection
- `start_detection(source_id, family?, proc_width?, detect_interval?, use_clahe?, use_sharpen?)` — creates DetectionLoop, returns detection_id
- `stop_detection(detection_id)` — stops loop, cleans up
- Detection loop registry in MCP server
- One loop per source enforcement
- Depends on: 004
- Use cases: SUC-001, SUC-002
- Tests: start/stop round-trip, error cases (invalid source, duplicate start, invalid stop)

### Ticket 006: MCP tools — get_tags and get_tag_history
- `get_tags(source_id)` — returns latest FrameRecord as JSON
- `get_tag_history(source_id, num_frames?)` — returns last N FrameRecords
- Depends on: 005
- Use cases: SUC-003, SUC-004
- Tests: query after start, empty tags, history ordering, no-loop error

### Ticket 007: Integration tests
- Full round-trip: start -> get_tags -> get_tag_history -> stop using mock camera with test images
- Verify tag positions, velocity computation, all JSON fields present
- Depends on: 006
- Use cases: SUC-001 through SUC-005
- Tests: end-to-end MCP tool flow

For each ticket, fill in the description, acceptance criteria (as checkboxes), testing section, use-cases list, and depends-on list. Write substantive content — not placeholders.

## Context Documents

- `docs/clasi/sprints/004-tag-detection-loop-ring-buffer/sprint.md`
- `docs/clasi/sprints/004-tag-detection-loop-ring-buffer/architecture-update.md`
- `docs/clasi/sprints/004-tag-detection-loop-ring-buffer/usecases.md`
