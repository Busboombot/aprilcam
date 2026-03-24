---
id: '002'
title: RingBuffer class
status: done
use-cases:
- SUC-003
- SUC-004
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# RingBuffer class

## Description

Add a thread-safe `RingBuffer` class to `src/aprilcam/detection.py`
that stores `FrameRecord` objects in a fixed-size circular buffer. This
is the shared data structure between the `DetectionLoop` (writer) and
the MCP query tools (readers).

The buffer is backed by `collections.deque(maxlen=300)` and protected
by a `threading.Lock`. The lock is held only during reads and writes
(microsecond-scale operations on the deque) to minimize contention
between the detection thread and the MCP server thread.

### API

```python
class RingBuffer:
    def __init__(self, maxlen: int = 300) -> None: ...
    def append(self, record: FrameRecord) -> None: ...
    def get_latest(self) -> FrameRecord | None: ...
    def get_last_n(self, n: int) -> list[FrameRecord]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
```

- `append(record)` -- add a FrameRecord; oldest is evicted if at capacity
- `get_latest()` -- return the most recent FrameRecord, or None if empty
- `get_last_n(n)` -- return the last N records as a list (oldest first);
  if n > len, return all available; if n <= 0, return empty list
- `clear()` -- remove all records
- `__len__()` -- return current count of records

All methods acquire the lock before accessing the deque.

## Acceptance Criteria

- [ ] `RingBuffer` class exists in `src/aprilcam/detection.py`
- [ ] Constructor accepts `maxlen` parameter (default 300)
- [ ] `append()` adds a `FrameRecord` to the buffer
- [ ] Buffer evicts oldest record when at capacity (deque maxlen behavior)
- [ ] `get_latest()` returns the most recent `FrameRecord` or `None` if empty
- [ ] `get_last_n(n)` returns up to N records in chronological order (oldest first)
- [ ] `get_last_n(0)` returns an empty list
- [ ] `get_last_n(n)` where n > buffer size returns all available records
- [ ] `clear()` empties the buffer
- [ ] `__len__()` returns the current number of records
- [ ] All methods are thread-safe (protected by `threading.Lock`)
- [ ] All existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite
- **New tests to write** (in `tests/test_detection.py`):
  - `test_ringbuffer_empty` -- new buffer has len 0, get_latest returns None, get_last_n returns []
  - `test_ringbuffer_append_and_get_latest` -- append one record, get_latest returns it
  - `test_ringbuffer_get_last_n` -- append 5 records, get_last_n(3) returns last 3 in order
  - `test_ringbuffer_overflow` -- append 310 records to a maxlen=300 buffer, verify len is 300, oldest records are gone
  - `test_ringbuffer_get_last_n_exceeds_size` -- buffer has 5 records, get_last_n(100) returns all 5
  - `test_ringbuffer_get_last_n_zero` -- get_last_n(0) returns empty list
  - `test_ringbuffer_clear` -- append records, clear, verify empty
  - `test_ringbuffer_thread_safety` -- spawn multiple reader threads calling get_latest/get_last_n while a writer thread appends records; verify no exceptions or corruption
- **Verification command**: `uv run pytest tests/test_detection.py -v`
