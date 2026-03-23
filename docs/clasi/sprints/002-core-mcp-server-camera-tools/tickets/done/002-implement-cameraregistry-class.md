---
id: "002"
title: "Implement CameraRegistry class"
status: todo
use-cases:
  - SUC-002
  - SUC-004
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Implement CameraRegistry class

## Description

Create the `CameraRegistry` class in a new file `src/aprilcam/mcp_server.py`.
This class manages a dictionary mapping UUID string handles to open capture
objects (`cv2.VideoCapture` or `ScreenCaptureMSS`). It provides the
handle-based resource management that all MCP camera tools depend on.

This ticket creates the file and the registry class only. The MCP server
setup and tool handlers are added in ticket 003.

## Acceptance Criteria

- [ ] `CameraRegistry` class exists in `src/aprilcam/mcp_server.py`
- [ ] `open(capture)` accepts a capture object, assigns a UUID handle, stores it, and returns the handle string
- [ ] `get(camera_id)` returns the capture object for a valid handle
- [ ] `get(camera_id)` raises `KeyError` with a descriptive message for an invalid handle
- [ ] `close(camera_id)` calls `release()` on the capture object and removes it from the registry
- [ ] `close(camera_id)` raises `KeyError` for an invalid or already-closed handle
- [ ] `close_all()` releases all open captures and clears the registry
- [ ] `list_open()` returns a list of active camera_id strings
- [ ] `__del__` calls `close_all()` for cleanup on garbage collection

## Implementation Notes

File to create: `src/aprilcam/mcp_server.py`

```python
import uuid
from typing import Any


class CameraRegistry:
    """Manages open camera handles for the MCP server."""

    def __init__(self):
        self._cameras: dict[str, Any] = {}

    def open(self, capture: Any) -> str:
        camera_id = str(uuid.uuid4())
        self._cameras[camera_id] = capture
        return camera_id

    def get(self, camera_id: str) -> Any:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise KeyError(f"Invalid camera handle: {camera_id}")

    def close(self, camera_id: str) -> None:
        cap = self.get(camera_id)  # raises KeyError if invalid
        cap.release()
        del self._cameras[camera_id]

    def close_all(self) -> None:
        for cap in self._cameras.values():
            try:
                cap.release()
            except Exception:
                pass
        self._cameras.clear()

    def list_open(self) -> list[str]:
        return list(self._cameras.keys())

    def __del__(self):
        self.close_all()
```

## Testing

- **Existing tests to run**: `uv run pytest` -- no regressions.
- **New tests to write**: Unit tests for `CameraRegistry` using mock
  capture objects are written in ticket 004.
- **Verification command**: `uv run python -c "from aprilcam.mcp_server import CameraRegistry; print('OK')"`
