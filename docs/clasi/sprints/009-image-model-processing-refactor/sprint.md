---
id: "009"
title: "Image Model & Processing Refactor"
status: planning
branch: sprint/009-image-model-processing-refactor
use-cases: [SUC-001, SUC-002, SUC-003, SUC-004, SUC-005]
---

# Sprint 009: Image Model & Processing Refactor

## Goals

Introduce a handle-based `FrameEntry` model where image frames are
server-side resources. Agents create a frame (from camera or file), get
a handle back, then request batch operations on that handle. Frames are
stored in a ring buffer and can be inspected at each processing stage.
Refactor velocity computation into Playfield with EMA + dead-band.

## Problem

The current architecture tightly couples camera capture with image processing.
MCP tools take a `source_id`, internally capture a frame, process it, and
return results in one atomic step. This prevents:

- Testing without a live camera
- Step-by-step inspection of the processing pipeline
- Batch processing (multiple operations in one call)
- Reusing a captured frame across multiple operations
- Going back to earlier frames for comparison

Velocity computation is duplicated between `AprilTagFlow` and `AprilCam`.

## Solution

### Handle-Based Frame Model

1. **FrameEntry** — server-side resource with three image slots:
   - `original`: raw captured image, never modified
   - `deskewed`: deskewed version (or reference to original if not deskewed)
   - `processed`: pipeline output (starts as reference to deskewed)

2. **FrameRegistry** — ring buffer of 300 frames (~10s at 30fps). Frames
   auto-evict when full. Deterministic IDs (`frm_000`, `frm_001`, ...).

3. **New MCP tools**:
   - `create_frame(source_id, operations?)` — capture + optional pipeline
   - `create_frame_from_image(image_path, operations?)` — load from disk
   - `process_frame(frame_id, operations)` — batch operations on a frame
   - `get_frame_image(frame_id, stage)` — inspect original/deskewed/processed
   - `save_frame(frame_id, output_dir)` — write frame directory to disk
   - `release_frame(frame_id)` — explicit cleanup
   - `list_frames()` — show ring buffer contents

4. **Batch operations**: `["deskew", "detect_tags", "detect_lines", ...]`
   run in order, results returned in one response.

5. **Streaming**: `stream_tags(source_id, operations)` — fast continuous mode
   with fixed pipeline. Frames cycle through frame ring buffer, TagRecords
   through detection ring buffer.

### Velocity Refactor

- Playfield owns velocity (EMA + dead-band from AprilCam)
- AprilTagFlow remains, velocity set by Playfield externally
- AprilTag gets `family: str` field

### Backward Compatibility

- Existing per-operation tools remain as convenience wrappers
- No external API changes for current tools

## Success Criteria

- Frame lifecycle works: create from camera and from file, process, inspect
- Batch operations return all results in one call
- `save_frame` writes directory with all three images + metadata
- Frame ring buffer limits memory, allows accessing earlier frames
- Static image testing works end-to-end via MCP tools
- Playfield computes velocity with EMA + dead-band
- All existing tests pass
- Existing MCP tool API unchanged

## Scope

### In Scope

- FrameEntry dataclass with three image slots
- FrameRegistry with ring buffer (300 frames)
- New MCP tools: create_frame, create_frame_from_image, process_frame,
  get_frame_image, save_frame, release_frame, list_frames
- Batch operation pipeline (deskew, detect_tags, detect_aruco, detect_lines,
  detect_circles, detect_contours, detect_qr)
- stream_tags with fixed operation pipeline
- Velocity computation moved to Playfield (EMA + dead-band)
- AprilTag family field
- AprilTagFlow velocity set externally by Playfield
- Existing tools refactored as thin wrappers
- Tests using static test images

### Out of Scope

- New image processing algorithms
- Camera hardware changes
- Recording new test data (stakeholder will assist)
- Live view changes
- Streamable HTTP transport

## Test Strategy

- **Unit tests**: FrameEntry slot promotion logic, FrameRegistry ring buffer,
  batch operation pipeline, velocity EMA computation
- **Integration tests**: MCP tool flow with static images — create from file,
  process, inspect, save
- **Backward compat tests**: Existing per-operation tools still work
- **Regression**: All existing tests pass

## Architecture Notes

See architecture-update.md for full details. Key decisions:
- FrameEntry is mutable (progressive enrichment)
- Three image slots with zero-copy reference promotion
- Ring buffer for frame storage (300 entries)
- Operations are flags: `["deskew", "detect_tags", ...]`
- Playfield uses EMA + dead-band velocity (from AprilCam)
- AprilTagFlow stays as data structure, velocity set externally
- TagRecord construction stays in DetectionLoop

## GitHub Issues

None.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [x] Sprint planning documents are complete
- [x] Architecture review passed
- [x] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
