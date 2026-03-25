---
ticket: "001"
sprint: "009"
---

# Ticket Plan: FrameEntry dataclass and FrameRegistry with ring buffer

## Approach

Create a new module `src/aprilcam/frame.py` (mcp_server.py is 1894 lines,
too large to add more classes). This module contains `FrameEntry` and
`FrameRegistry`.

### FrameEntry

Mutable dataclass with three image slots. On construction, `deskewed` and
`processed` are the same object reference as `original` (no copy). A
`deskew()` method creates a new array for `deskewed` and updates
`processed` to point to it.

### FrameRegistry

Dict-backed registry with ring buffer behavior. Uses an `OrderedDict` or
plain dict + deque for eviction ordering. Deterministic IDs via a counter
(`frm_000`, `frm_001`, ...). Thread-safe with `threading.Lock`.

Key difference from existing `RingBuffer`: FrameRegistry is keyed by
`frame_id` (supports random access by ID), while the existing RingBuffer
is a sequential deque. Use a dict for O(1) lookup + a deque for eviction
order.

## Files to Create

- `src/aprilcam/frame.py` — FrameEntry dataclass, FrameRegistry class

## Files to Modify

None (this ticket is foundation only — no integration yet).

## Testing Plan

New test file: `tests/unit/test_frame.py`

Tests:
- `test_frame_entry_creation_slot_references` — verify `original is deskewed is processed`
- `test_frame_entry_deskew` — after deskew, `deskewed` is new array, `processed` updated
- `test_frame_registry_add_and_get` — add frame, retrieve by ID
- `test_frame_registry_deterministic_ids` — verify frm_000, frm_001 sequence
- `test_frame_registry_auto_eviction` — fill to capacity, verify oldest evicted
- `test_frame_registry_release` — explicit release, KeyError after
- `test_frame_registry_list` — verify summary dicts
- `test_frame_registry_thread_safety` — concurrent add/get from multiple threads

Verification: `uv run pytest tests/unit/test_frame.py -v && uv run pytest`

## Documentation Updates

None (internal module, no user-facing docs needed yet).
