---
sprint: "004"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 004: Tag Detection Loop & Ring Buffer

## What Changed

### New Module: `src/aprilcam/detection.py`

This module contains the core additions for this sprint:

- **`TagRecord` dataclass** -- Immutable snapshot of a single tag's
  state at a point in time. Fields: `id`, `center_px`, `corners_px`,
  `orientation_yaw`, `world_xy`, `in_playfield`, `vel_px`, `speed_px`,
  `vel_world`, `speed_world`, `heading_rad`, `timestamp`, `frame_index`.
  Includes a `to_dict()` method for JSON serialization.

- **`FrameRecord` dataclass** -- Groups a `timestamp`, `frame_index`,
  and a list of `TagRecord` objects representing all tags visible in a
  single frame. Includes a `to_dict()` method.

- **`RingBuffer` class** -- Thread-safe circular buffer backed by
  `collections.deque(maxlen=300)`. Protected by `threading.Lock`.
  Methods:
  - `append(frame_record)` -- add a frame (O(1), drops oldest if full).
  - `get_latest() -> Optional[FrameRecord]` -- most recent frame.
  - `get_last_n(n) -> List[FrameRecord]` -- last N frames, oldest first.
  - `clear()` -- discard all records.

- **`DetectionLoop` class** -- Manages the background detection thread.
  Key attributes and methods:
  - `__init__(source, aprilcam, ring_buffer)` -- takes a video source
    (camera or playfield), a headless `AprilCam` instance, and a
    `RingBuffer`.
  - `start()` -- spawns a daemon thread that loops: grab frame, call
    `aprilcam.process_frame(frame)`, build `FrameRecord`, append to
    ring buffer. Uses `threading.Event` for stop signaling.
  - `stop()` -- sets the stop event, joins the thread (timeout 2s).
  - `is_running` -- property indicating whether the thread is alive.
  - The loop catches exceptions per-frame to avoid crashing on transient
    camera glitches.

### Modified Module: `src/aprilcam/aprilcam.py`

- **New method `AprilCam.process_frame(frame_bgr, timestamp) -> List[TagRecord]`**:
  Extracted from the body of `run()`. Performs one iteration of
  detection (or LK tracking on non-detect frames), playfield filtering,
  tag model updates, and velocity computation. Returns a list of
  `TagRecord` objects for the frame. Manages internal state (prev_gray,
  tracks, tag_models, frame_idx) across calls.

- **Refactored `AprilCam.run()`**: The existing loop now calls
  `process_frame()` on each frame, then handles display, overlays, and
  keyboard input using the returned tag records. External behavior is
  unchanged.

- **New internal state attributes**: `_prev_gray`, `_tracks`,
  `_tag_models`, `_vel_ema`, `_last_seen`, `_frame_idx` -- moved from
  local variables in `run()` to instance attributes so that
  `process_frame()` can maintain state across calls. Initialized in a
  new `reset_state()` method called at the start of `run()` and by
  `DetectionLoop.start()`.

### Modified Module: `src/aprilcam/models.py`

- **`AprilTagFlow` enhancement**: Add `vel_world` and `speed_world`
  properties that compute world-coordinate velocity when `world_xy` is
  available on consecutive observations. Add `heading_rad` property
  (direction of motion from `vel_px`).

### Modified: MCP Server Tool Registration

Four new tools registered in the MCP server:

| Tool | Input | Output |
|------|-------|--------|
| `start_detection` | `source_id`, `family?`, `proc_width?`, `detect_interval?`, `use_clahe?`, `use_sharpen?` | `{ detection_id, source_id, status }` |
| `stop_detection` | `detection_id` | `{ detection_id, status }` |
| `get_tags` | `source_id` | `FrameRecord` as JSON |
| `get_tag_history` | `source_id`, `num_frames?` (default 30) | Array of `FrameRecord` as JSON |

The MCP server maintains an internal registry:
- `_detection_loops: Dict[str, DetectionLoop]` -- keyed by detection_id.
- `_source_to_detection: Dict[str, str]` -- maps source_id to
  detection_id, enforcing one loop per source.

## Why

The existing `AprilCam.run()` is an interactive, blocking loop designed
for human use with a display window. The MCP server needs to run
detection headlessly in a background thread so that AI agents can query
tag state on demand without blocking.

The ring buffer provides temporal context that agents need for motion
analysis -- the existing `AprilTagFlow` deque of 5 per-tag observations
is insufficient for trajectory analysis, and there is no concept of a
per-frame snapshot capturing all tags simultaneously.

Extracting `process_frame()` from `run()` creates a clean boundary
between the detection/tracking core and the display/input layer,
enabling both the CLI and MCP server to share the same detection logic.

## Impact on Existing Components

### `AprilCam` class

The `run()` method is refactored but its external behavior is preserved.
Internal state that was previously held in local variables moves to
instance attributes. This is a **non-breaking** change for CLI users.

The new `process_frame()` method is the primary interface for the
`DetectionLoop`. It does not affect any existing callers.

### `models.py`

New dataclasses (`TagRecord`, `FrameRecord`) are added. The existing
`AprilTag` and `AprilTagFlow` classes gain new properties but no
existing properties are changed or removed. **Non-breaking**.

### MCP Server

Four new tools are added to the tool registry. No existing tools are
modified. The server gains a detection-loop registry as internal state.
**Additive only**.

### `Playfield` class

No changes. `DetectionLoop` reads from `Playfield` via `AprilCam` in
the same way that `run()` does today.

### Dependencies

No new external dependencies. Uses only `threading`, `collections`,
`time`, and `dataclasses` from the standard library.

## Migration Concerns

None. This sprint adds new modules and methods without changing any
existing public interfaces. No data migration is needed. The ring
buffer is ephemeral (in-memory only, not persisted).
