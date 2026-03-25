---
status: reviewed
---

# Sprint 009 Use Cases

## SUC-001: Create and Process a Frame From a Static Image
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: AI agent / Test harness
- **Preconditions**: A JPEG/PNG image file exists on disk
- **Main Flow**:
  1. Call `create_frame_from_image(path, operations=["detect_tags"])`
  2. Receive `frame_id` and detection results in one response
  3. Call `get_frame_image(frame_id, "original")` to inspect the raw image
  4. Call `get_frame_image(frame_id, "processed")` to see pipeline output
  5. Call `save_frame(frame_id, output_dir)` to persist all three slots
- **Postconditions**: Frame exists in registry with detection results.
  No camera was opened.
- **Acceptance Criteria**:
  - [ ] `create_frame_from_image` loads image and returns frame_id
  - [ ] Optional operations run during creation
  - [ ] `get_frame_image` returns image at each stage
  - [ ] `save_frame` writes directory with original.jpg, deskewed.jpg,
        processed.jpg, metadata.json

## SUC-002: Batch Operations on a Camera Frame
Parent: UC-003, UC-005 (Image Processing Tools)

- **Actor**: AI agent (via MCP)
- **Preconditions**: Camera is open
- **Main Flow**:
  1. Call `create_frame(source_id="cam_0")`
  2. Call `process_frame(frame_id, ["deskew", "detect_tags", "detect_lines"])`
  3. Receive all results in one response
  4. Call `get_frame_image(frame_id, "deskewed")` to inspect deskewed image
  5. Call `get_frame_image(frame_id, "processed")` to inspect final result
- **Postconditions**: Frame holds deskewed image in slot 2, detection and
  line results in results dict
- **Acceptance Criteria**:
  - [ ] Operations execute in order
  - [ ] Deskew updates slot 2, detection reads from slot 3
  - [ ] All results returned in one response
  - [ ] Frame inspectable at each stage

## SUC-003: Playfield Computes Tag Velocity From History
Parent: UC-003

- **Actor**: Detection loop / Playfield
- **Preconditions**: Playfield created, detection running
- **Main Flow**:
  1. Detection loop captures frame, detects tags
  2. Tags passed to Playfield via `add_tag()`
  3. Playfield computes velocity using EMA + dead-band
  4. Agent queries `get_tags()` — velocity from Playfield
- **Postconditions**: Velocity computed by Playfield, not per-frame
- **Acceptance Criteria**:
  - [ ] Playfield owns velocity computation (EMA + dead-band)
  - [ ] Individual tag detections carry position but not velocity
  - [ ] `get_tags()` returns velocity sourced from Playfield

## SUC-004: Stream Tags With Fixed Pipeline
Parent: UC-003

- **Actor**: AI agent
- **Preconditions**: Camera open, playfield created
- **Main Flow**:
  1. Call `stream_tags(source_id, operations=["deskew", "detect_tags"])`
  2. Loop runs continuously: capture -> deskew -> detect -> ring buffer
  3. Agent calls `get_tags()` / `get_tag_history()` to read results
  4. Frames cycle through frame ring buffer (accessible by frame_id)
  5. Agent calls `stop_stream()` to end
- **Postconditions**: Tag history in detection ring buffer, recent frames
  in frame ring buffer
- **Acceptance Criteria**:
  - [ ] Pipeline fixed at start, runs every frame
  - [ ] Results in detection ring buffer (TagRecords)
  - [ ] Frames in frame ring buffer (300 entries)
  - [ ] get_tags / get_tag_history work as before

## SUC-005: Access Earlier Frames From Ring Buffer
Parent: UC-003

- **Actor**: AI agent
- **Preconditions**: Frames have been created (manually or via streaming)
- **Main Flow**:
  1. Call `list_frames()` to see available frames
  2. Pick an earlier frame_id
  3. Call `get_frame_image(frame_id, "original")` to inspect it
  4. Call `process_frame(frame_id, ["detect_contours"])` to run new
     operations on an old frame
- **Postconditions**: Earlier frame retrieved and further processed
- **Acceptance Criteria**:
  - [ ] Ring buffer holds up to 300 frames
  - [ ] Old frames auto-evict when buffer is full
  - [ ] Earlier frames accessible by frame_id
  - [ ] Can run new operations on earlier frames
