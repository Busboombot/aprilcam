---
id: "004"
title: "Camera contention error reporting"
status: done
use-cases: [SUC-011-003]
depends-on: []
github-issue: ""
todo: ""
---

# Camera contention error reporting

## Description

When a camera fails to open, diagnose why and report the blocking
process if the camera is in use.

### Changes

1. **New `aprilcam/errors.py`**: Exception classes:
   - `CameraError(Exception)` — base
   - `CameraNotFoundError(CameraError)` — index doesn't exist
   - `CameraInUseError(CameraError)` — camera busy, includes PID/name
   - `CameraPermissionError(CameraError)` — permission denied

2. **`camutil.py`**: Add `diagnose_camera_failure(index)` function:
   - On macOS: run `lsof` and look for processes with the camera
     device open; also check `system_profiler SPCameraDataType` to
     confirm the camera exists
   - On Linux: check `/dev/video{index}` exists, use `fuser` to find
     blocking processes
   - Return diagnostic info: exists (bool), blocking_processes
     (list of {pid, name})

3. **`aprilcam.py`**: Replace `print("Failed to open camera.")`
   with proper exception raising. Call `diagnose_camera_failure()`
   and raise the appropriate exception subclass.

4. **`mcp_server.py`**: Catch the new exceptions in `open_camera`
   handler and return structured error messages.

## Acceptance Criteria

- [ ] `CameraInUseError` raised with PID and process name when
      camera is busy
- [ ] `CameraNotFoundError` raised when camera index doesn't exist
- [ ] Error message includes `kill <PID>` suggestion
- [ ] MCP `open_camera` returns structured error with contention info
- [ ] Works on macOS (lsof); graceful fallback on unsupported platforms

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test `diagnose_camera_failure()` with mocked
  subprocess output; test exception hierarchy
- **Verification command**: `uv run pytest`
