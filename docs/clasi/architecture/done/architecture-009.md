---
sprint: "009"
status: reviewed
---

# Architecture Update -- Sprint 009: Image Model & Processing Refactor

## What Changed

### New: `FrameEntry` dataclass (in `mcp_server.py`)

Server-side resource representing a captured image with three slots:

```python
@dataclass
class FrameEntry:
    frame_id: str
    source: str               # "cam_0", "file:/path/to/img.jpg"
    timestamp: float

    # --- Three image slots ---
    original: np.ndarray      # Slot 1: raw captured image, never modified
    deskewed: np.ndarray      # Slot 2: deskewed (or reference to original)
    processed: np.ndarray     # Slot 3: pipeline output (starts as ref to slot 2)

    # --- Metadata ---
    is_deskewed: bool = False
    operations_applied: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)

    # Detection results cached on the frame
    aruco_corners: Optional[Dict] = None
    apriltags: Optional[List] = None
```

**Slot promotion logic:**
1. On create: `original = captured`. `deskewed = original` (same ref).
   `processed = deskewed` (same ref). Zero copies.
2. On deskew: `deskewed = warp(original)` (new array).
   `processed = deskewed` (ref updated). `is_deskewed = True`.
3. On pipeline ops: read from `processed`. Detection ops store results
   without modifying the array. Transform ops write new array to `processed`.
4. `get_frame_image` stages: `"original"` -> slot 1, `"deskewed"` -> slot 2,
   `"processed"` -> slot 3.

### New: `FrameRegistry` (in `mcp_server.py`)

Ring-buffer-backed registry for FrameEntry objects. Mirrors CameraRegistry
pattern with auto-eviction:

- Capacity: 300 entries (configurable)
- Deterministic IDs: `frm_000`, `frm_001`, ...
- Auto-evicts oldest frame when full
- `release_frame` for explicit early cleanup
- Thread-safe (locking like existing RingBuffer)

### New MCP Tools: Frame Lifecycle

| Tool | Purpose |
|------|---------|
| `create_frame(source_id, operations?)` | Capture from camera/playfield, optional pipeline |
| `create_frame_from_image(image_path, operations?)` | Load from disk, optional pipeline |
| `process_frame(frame_id, operations)` | Batch operations on existing frame |
| `get_frame_image(frame_id, stage)` | Inspect original/deskewed/processed |
| `save_frame(frame_id, output_dir)` | Write frame directory to disk |
| `release_frame(frame_id)` | Explicit cleanup |
| `list_frames()` | Show ring buffer contents |

### New MCP Tool: `stream_tags`

Replaces `start_detection` internally. Accepts `operations` list (fixed at
start). Continuous loop: capture -> run pipeline -> write TagRecords to
detection RingBuffer, frames to frame RingBuffer. `get_tags` / `get_tag_history`
read from detection RingBuffer as before.

### Batch Operations

`process_frame` accepts an ordered list of operation flags:

| Operation | Reads from | Writes to |
|-----------|-----------|-----------|
| `"deskew"` | `original` | `deskewed` (new array), `processed` (ref update) |
| `"detect_tags"` | `processed` | `results["detect_tags"]`, `apriltags` |
| `"detect_aruco"` | `processed` | `results["detect_aruco"]`, `aruco_corners` |
| `"detect_lines"` | `processed` | `results["detect_lines"]` |
| `"detect_circles"` | `processed` | `results["detect_circles"]` |
| `"detect_contours"` | `processed` | `results["detect_contours"]` |
| `"detect_qr"` | `processed` | `results["detect_qr"]` |

Operations execute in order. All results returned in one response.

### Modified: `AprilTag` dataclass (in `models.py`)

- Adds: `family: str` field (e.g. "36h11"). Populated by modifying
  `AprilCam.detect_apriltags()` to thread the family name through detections.
- No velocity fields exist on AprilTag (unchanged).

### Modified: `AprilTagFlow` (in `models.py`)

- Remains a separate class with history deque
- Velocity self-computation removed
- `vel_px` and `speed_px` set externally by Playfield

### Modified: `Playfield` (in `playfield.py`)

- Adds EMA + dead-band velocity computation (adopted from `AprilCam.process_frame()`)
- Parameters configurable via `__init__()`: EMA alpha, dead-band threshold
- `add_tag()` computes velocity and stores on the flow
- ArUco corner detection remains one-time setup

### Modified: `AprilCam` (in `aprilcam.py`)

- Velocity computation (`_vel_ema`, `_last_seen`) removed — moved to Playfield
- `process_frame()` returns detection results suitable for FrameEntry

### Modified: `detection.py`

- `DetectionLoop` uses batch-operation pipeline internally
- `TagRecord` construction remains in DetectionLoop
- `RingBuffer` and `FrameRecord` unchanged

### Modified: `mcp_server.py`

- Existing per-operation tools become thin wrappers: create transient frame,
  run operation, return results. Backward compatible.
- `resolve_source()` still used internally for backward-compat tools

## Why

1. **Testability** — `create_frame_from_image` enables full pipeline testing
   with static images, no camera needed
2. **Inspectability** — `get_frame_image` at each stage lets agents understand
   what the pipeline does
3. **Efficiency** — batch operations eliminate multiple round-trips for common
   multi-operation workflows
4. **Memory safety** — ring buffer bounds frame storage, auto-evicts old frames
5. **Velocity ownership** — single source of truth in Playfield with proven
   EMA + dead-band algorithm

## Impact on Existing Components

- **MCP external API**: Existing tools unchanged. New tools are additive.
- **Detection loop**: Internal change — uses batch pipeline, writes to both
  frame and detection ring buffers.
- **Tests**: Existing tests updated. New tests for frame lifecycle, batch
  operations, static image flow.
- **CLI tools**: No changes.

## Migration Concerns

- All changes are internal refactoring + additive MCP tools
- No data migration needed
- Backward compatibility: existing tool responses unchanged
- Velocity behavioral parity: Playfield EMA must match current AprilCam output

## Decisions

1. **FrameEntry is mutable** — progressive enrichment through pipeline.
2. **Three image slots** — original (immutable), deskewed (copy or ref),
   processed (pipeline output). Zero-copy when no deskew applied.
3. **Frame ring buffer (300 entries)** — auto-eviction, bounded memory.
4. **Create tools accept optional operations** — one call for capture + process.
5. **Keep both old and new tools** — no breaking changes.
6. **Stream pipeline fixed at start** — stop and restart to change.
7. **AprilTagFlow stays** — Playfield sets velocity externally.
8. **Playfield uses EMA + dead-band velocity** — from AprilCam.
9. **TagRecord construction stays in DetectionLoop**.
