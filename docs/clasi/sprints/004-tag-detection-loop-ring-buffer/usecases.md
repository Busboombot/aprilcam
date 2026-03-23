---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 004 Use Cases

## SUC-001: Start a Detection Loop
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - The MCP server is running.
  - A camera or playfield has been opened (source_id is valid).
  - No detection loop is already running on this source_id.
- **Main Flow**:
  1. Agent calls `start_detection(source_id)` with optional parameters:
     `family` (default "all"), `proc_width` (default 960),
     `detect_interval` (default 1), `use_clahe` (default false),
     `use_sharpen` (default false).
  2. The server creates a `DetectionLoop` bound to the source,
     configures an `AprilCam` instance in headless mode with the
     specified parameters.
  3. The server starts the detection loop in a background daemon thread.
  4. The server returns `{ detection_id, source_id, status: "running" }`.
- **Postconditions**:
  - A background thread is actively capturing frames and detecting tags.
  - The ring buffer begins accumulating `FrameRecord` entries.
  - The detection_id is registered in the server's active-loop registry.
- **Acceptance Criteria**:
  - [ ] `start_detection` returns a detection_id and status "running".
  - [ ] The detection loop begins processing frames within 1 second.
  - [ ] Calling `start_detection` on a source that already has an active
        loop returns an error (not a second loop).
  - [ ] Optional parameters (family, proc_width, etc.) are respected
        by the underlying `AprilCam` configuration.

## SUC-002: Stop a Detection Loop
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - A detection loop is running with the given detection_id.
- **Main Flow**:
  1. Agent calls `stop_detection(detection_id)`.
  2. The server signals the detection loop thread to stop.
  3. The thread finishes its current frame, exits cleanly, and releases
     any resources (but does not close the underlying camera -- the
     camera lifecycle is managed separately).
  4. The server removes the detection_id from the active-loop registry.
  5. The server returns `{ detection_id, status: "stopped" }`.
- **Postconditions**:
  - The background thread has terminated.
  - The ring buffer contents are discarded.
  - The source_id is free for a new detection loop.
- **Acceptance Criteria**:
  - [ ] `stop_detection` returns within 2 seconds (does not hang).
  - [ ] After stopping, `get_tags` on the same source returns an error
        or empty result indicating no active loop.
  - [ ] Calling `stop_detection` with an invalid detection_id returns
        an appropriate error message.
  - [ ] The underlying camera remains open and usable after stopping.

## SUC-003: Query Current Tag State
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - A detection loop is running on the specified source_id.
- **Main Flow**:
  1. Agent calls `get_tags(source_id)`.
  2. The server reads the latest `FrameRecord` from the ring buffer
     (thread-safe read under lock).
  3. The server serializes the frame record to JSON containing:
     - `frame_index`: integer frame counter
     - `timestamp`: monotonic timestamp (seconds)
     - `tags`: array of tag records, each with:
       - `id`: tag ID
       - `center_px`: [x, y] pixel coordinates
       - `corners_px`: [[x,y], [x,y], [x,y], [x,y]] pixel corners
       - `orientation_yaw`: radians
       - `world_xy`: [x, y] in world units, or null
       - `in_playfield`: boolean
       - `vel_px`: [vx, vy] in px/s
       - `speed_px`: scalar px/s
       - `vel_world`: [vx, vy] in world units/s, or null
       - `speed_world`: scalar world units/s, or null
       - `heading_rad`: direction of motion in radians, or null
  4. The server returns the JSON response.
- **Postconditions**:
  - The agent has a snapshot of all currently visible tags.
- **Acceptance Criteria**:
  - [ ] Response includes all fields listed above for every detected tag.
  - [ ] `world_xy`, `vel_world`, `speed_world` are null when no
        homography/calibration is available.
  - [ ] If no tags are detected in the latest frame, `tags` is an
        empty array (not an error).
  - [ ] Response latency is under 50ms (buffer read, not a new capture).
  - [ ] Calling `get_tags` on a source with no active loop returns an
        error indicating no detection is running.

## SUC-004: Query Tag History
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: AI agent (via MCP)
- **Preconditions**:
  - A detection loop is running on the specified source_id.
  - The ring buffer has at least one frame recorded.
- **Main Flow**:
  1. Agent calls `get_tag_history(source_id, num_frames=30)`.
  2. The server reads the last `num_frames` entries from the ring buffer
     (clamped to the number of available frames).
  3. The server serializes the frame records to JSON: an array of
     `FrameRecord` objects (same structure as `get_tags` but multiple
     frames).
  4. The server returns the JSON response.
- **Postconditions**:
  - The agent has a time-ordered sequence of frame snapshots for
    trajectory analysis.
- **Acceptance Criteria**:
  - [ ] Returns exactly `min(num_frames, available_frames)` records.
  - [ ] Records are ordered oldest-first (ascending timestamp).
  - [ ] `num_frames` defaults to 30 when omitted.
  - [ ] Requesting more than 300 frames returns at most 300 (buffer cap).
  - [ ] Each frame record includes all tags visible in that frame, with
        full field set (same as `get_tags`).
  - [ ] Calling with `num_frames=0` returns an empty array.

## SUC-005: Refactored CLI Detection
Parent: UC-003 (Tag Detection & Tracking)

- **Actor**: Developer (via CLI)
- **Preconditions**:
  - The `aprilcam` CLI is installed.
  - A camera is connected.
- **Main Flow**:
  1. Developer runs `aprilcam run --camera 0` (existing CLI command).
  2. The refactored `AprilCam.run()` method internally calls
     `process_frame()` for each iteration of the loop.
  3. Detection, tracking, playfield updates, overlay rendering, and
     keyboard input handling work exactly as before.
  4. Developer presses 'q' or Esc to exit.
- **Postconditions**:
  - The CLI experience is unchanged from the user's perspective.
- **Acceptance Criteria**:
  - [ ] `aprilcam run` produces the same visual output and overlays as
        before the refactor.
  - [ ] Tag detection, LK tracking, and playfield filtering all work.
  - [ ] Speed/velocity display in the overlay is correct.
  - [ ] No regressions in pause ('space'), quit ('q'/'Esc') behavior.
