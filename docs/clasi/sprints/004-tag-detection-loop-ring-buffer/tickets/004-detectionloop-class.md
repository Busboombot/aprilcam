---
id: "004"
title: "DetectionLoop class"
status: todo
use-cases:
  - SUC-001
  - SUC-002
depends-on:
  - "002"
  - "003"
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# DetectionLoop class

## Description

Add a `DetectionLoop` class to `src/aprilcam/detection.py` that runs
the frame-grab / detect / track cycle in a background daemon thread.
It reads frames from a video source, passes them through
`AprilCam.process_frame()`, and writes the resulting `FrameRecord`
objects into a `RingBuffer`.

This class is the core engine that the MCP tools (`start_detection`,
`stop_detection`) will manage. It decouples detection from the
interactive `run()` loop so AI agents can start detection without
blocking the MCP server.

### API

```python
class DetectionLoop:
    def __init__(
        self,
        source: cv2.VideoCapture,
        aprilcam: AprilCam,
        ring_buffer: RingBuffer,
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self, timeout: float = 5.0) -> None: ...

    @property
    def is_running(self) -> bool: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def error(self) -> Exception | None: ...
```

### Design Details

- **Constructor** receives a pre-opened `VideoCapture` (or playfield
  capture proxy), a headless `AprilCam` instance, and a `RingBuffer`.
  The `DetectionLoop` does not own or release the capture -- that is
  the caller's responsibility.

- **`start()`** spawns a daemon thread that loops: grab frame, call
  `aprilcam.process_frame(frame, time.monotonic())`, wrap the result
  in a `FrameRecord`, and `ring_buffer.append()` it. The loop checks
  a `threading.Event` (`_stop_event`) each iteration.

- **`stop(timeout)`** sets the stop event and joins the thread with
  the given timeout. If the thread does not stop within the timeout,
  log a warning but do not raise.

- **Per-frame exception handling** -- if `process_frame()` or
  `cap.read()` raises, catch the exception, store it in `self._error`,
  and continue to the next frame. After a configurable number of
  consecutive failures (default 10), stop the loop automatically.

- **`is_running`** property returns True if the thread is alive and
  the stop event is not set.

- **`frame_count`** property returns the number of frames processed.

- **`error`** property returns the last exception, or None.

- The thread is a daemon thread so it does not prevent process exit.

## Acceptance Criteria

- [ ] `DetectionLoop` class exists in `src/aprilcam/detection.py`
- [ ] Constructor accepts `source`, `aprilcam`, and `ring_buffer` parameters
- [ ] `start()` spawns a daemon thread that grabs frames and calls `process_frame()`
- [ ] Each frame's `TagRecord` list is wrapped in a `FrameRecord` and appended to the ring buffer
- [ ] `stop()` signals the thread to stop and joins with a timeout
- [ ] `is_running` returns True while the loop is active, False after stop
- [ ] `frame_count` tracks the number of frames processed
- [ ] Per-frame exceptions are caught and stored; loop continues
- [ ] After 10 consecutive frame failures, loop stops automatically
- [ ] `error` property exposes the last exception
- [ ] Thread is a daemon thread
- [ ] Calling `start()` twice raises `RuntimeError`
- [ ] Calling `stop()` on an already-stopped loop is a no-op
- [ ] All existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite
- **New tests to write** (in `tests/test_detection.py`):
  - `test_detectionloop_start_stop` -- create a mock VideoCapture that returns test images, start the loop, wait briefly, stop, verify `is_running` transitions and `frame_count > 0`
  - `test_detectionloop_writes_to_ringbuffer` -- start loop with mock capture, wait for a few frames, verify `ring_buffer.get_latest()` returns a valid FrameRecord
  - `test_detectionloop_double_start_raises` -- call `start()` twice, verify `RuntimeError`
  - `test_detectionloop_stop_idempotent` -- call `stop()` twice, no exception
  - `test_detectionloop_handles_frame_errors` -- mock capture that raises on `read()`, verify loop continues and `error` is set
  - `test_detectionloop_consecutive_failure_stops` -- mock capture that always fails, verify loop auto-stops after threshold
  - `test_detectionloop_concurrent_reads` -- start loop, spawn reader threads calling `ring_buffer.get_latest()` and `get_last_n()` concurrently, verify no exceptions
- **Verification command**: `uv run pytest tests/test_detection.py -v`
