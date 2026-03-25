---
id: "001"
title: "FrameEntry dataclass and FrameRegistry with ring buffer"
status: todo
use-cases: [SUC-005]
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# FrameEntry dataclass and FrameRegistry with ring buffer

## Description

Create the foundational `FrameEntry` dataclass and `FrameRegistry` ring buffer
that all other frame-related tickets build on. This is the core data model for
the handle-based frame architecture.

`FrameEntry` is a mutable dataclass representing a server-side image resource
with three image slots that support progressive enrichment through the processing
pipeline. `FrameRegistry` is a thread-safe ring buffer that stores FrameEntry
objects with deterministic IDs and auto-eviction.

### FrameEntry dataclass

Define in `src/aprilcam/mcp_server.py` (or a new `src/aprilcam/frame.py` module
if mcp_server.py is getting large):

```python
@dataclass
class FrameEntry:
    frame_id: str
    source: str               # "cam_0", "file:/path/to/img.jpg"
    timestamp: float

    # Three image slots
    original: np.ndarray      # Slot 1: raw captured image, never modified
    deskewed: np.ndarray      # Slot 2: deskewed (or reference to original)
    processed: np.ndarray     # Slot 3: pipeline output (starts as ref to slot 2)

    # Metadata
    is_deskewed: bool = False
    operations_applied: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)

    # Detection results cached on the frame
    aruco_corners: Optional[Dict] = None
    apriltags: Optional[List] = None
```

### Slot promotion logic

1. On create: `original = captured`. `deskewed = original` (same reference).
   `processed = deskewed` (same reference). Zero copies.
2. On deskew: `deskewed = warp(original)` (new array). `processed = deskewed`
   (reference updated). `is_deskewed = True`.
3. On pipeline ops: detection ops read from `processed` and store results
   without modifying the array. Transform ops write a new array to `processed`.

### FrameRegistry with ring buffer

- Capacity: 300 entries (configurable via constructor param)
- Deterministic IDs: `frm_000`, `frm_001`, ... wrapping at capacity
- Auto-evicts oldest frame when buffer is full
- Thread-safe with threading.Lock (same pattern as existing RingBuffer)
- Methods: `add(source, image) -> FrameEntry`, `get(frame_id) -> FrameEntry`,
  `release(frame_id)`, `list() -> List[dict]`, `__len__`

## Acceptance Criteria

- [ ] `FrameEntry` dataclass defined with all three image slots and metadata fields
- [ ] Slot promotion: on creation, `deskewed` and `processed` are the same object
      reference as `original` (verified with `is` operator)
- [ ] Deskew creates a new array for `deskewed`, updates `processed` reference
- [ ] `FrameRegistry` stores up to 300 entries (configurable)
- [ ] Deterministic IDs: first frame is `frm_000`, second is `frm_001`, etc.
- [ ] Auto-eviction: adding frame 301 evicts frame `frm_000`
- [ ] `release(frame_id)` removes a frame before auto-eviction
- [ ] `get(frame_id)` raises `KeyError` for unknown/evicted frames
- [ ] `list()` returns summary dicts (frame_id, source, timestamp, operations_applied)
- [ ] Thread-safe: concurrent add/get/release do not corrupt state

## Implementation Notes

### Key files
- `src/aprilcam/mcp_server.py` -- add FrameEntry and FrameRegistry classes
  (or create `src/aprilcam/frame.py` if preferred for modularity)
- `src/aprilcam/detection.py` -- reference for existing `RingBuffer` pattern

### Design decisions
- FrameEntry is mutable (progressive enrichment through pipeline)
- Zero-copy reference promotion: no `np.copy()` on creation, only on deskew
- Ring buffer wraps IDs: after `frm_299`, next is `frm_000` (old entry evicted)
- The `results` dict keys match operation names: `results["detect_tags"]`, etc.

## Testing

- **Existing tests to run**: `uv run pytest` (full suite, ensure no regressions)
- **New tests to write**:
  - `test_frame_entry_slot_promotion` -- verify zero-copy refs on creation
  - `test_frame_entry_deskew_creates_new_array` -- verify deskew breaks ref
  - `test_frame_registry_deterministic_ids` -- verify frm_000, frm_001 sequence
  - `test_frame_registry_auto_eviction` -- fill buffer, verify oldest evicted
  - `test_frame_registry_release` -- explicit release, verify KeyError after
  - `test_frame_registry_list` -- verify summary output
  - `test_frame_registry_thread_safety` -- concurrent operations
- **Verification command**: `uv run pytest`
