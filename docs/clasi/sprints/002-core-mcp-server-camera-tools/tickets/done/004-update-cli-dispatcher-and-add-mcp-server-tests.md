---
id: "004"
title: "Update CLI dispatcher and add MCP server tests"
status: todo
use-cases:
  - SUC-001
  - SUC-002
  - SUC-003
  - SUC-004
  - SUC-005
depends-on:
  - "003"
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Update CLI dispatcher and add MCP server tests

## Description

Two tasks that depend on the MCP server module (ticket 003) being complete:

1. **Update `src/aprilcam/cli/__init__.py`**: Replace the stub
   `"MCP server not yet implemented"` message with an actual call to
   `aprilcam.mcp_server.main()`, so that `aprilcam mcp` launches the
   MCP server (same as the `aprilcam-mcp` entry point).

2. **Create `tests/test_mcp_server.py`**: Unit tests for `CameraRegistry`
   and the four MCP tool handler functions, using mock capture objects to
   avoid needing real cameras in CI.

## Acceptance Criteria

### CLI Dispatcher Update

- [ ] `aprilcam mcp` calls `aprilcam.mcp_server.main()` instead of printing stub message
- [ ] The `SUBCOMMANDS["mcp"]["module"]` value is updated to `"aprilcam.mcp_server"`
- [ ] The special-case `if args.command == "mcp"` block is removed
- [ ] Existing subcommands are unaffected

### CameraRegistry Tests

- [ ] Test `open()` returns a UUID string and stores the capture
- [ ] Test `get()` returns the stored capture for a valid handle
- [ ] Test `get()` raises `KeyError` for invalid handle
- [ ] Test `close()` calls `release()` and removes the handle
- [ ] Test `close()` raises `KeyError` for invalid/already-closed handle
- [ ] Test `close_all()` releases all captures and clears registry
- [ ] Test `list_open()` returns correct active handles
- [ ] Test multiple cameras can be open simultaneously

### Tool Handler Tests

- [ ] Test `list_cameras` returns structured JSON array (mock `camutil.list_cameras`)
- [ ] Test `list_cameras` returns empty array when no cameras found
- [ ] Test `open_camera` with index creates `VideoCapture` and returns handle
- [ ] Test `open_camera` with pattern resolves index via `select_camera_by_pattern`
- [ ] Test `open_camera` with `source="screen"` creates `ScreenCaptureMSS`
- [ ] Test `open_camera` returns error when camera fails to open
- [ ] Test `capture_frame` returns base64 `ImageContent` by default
- [ ] Test `capture_frame` with `format="file"` writes temp file and returns path
- [ ] Test `capture_frame` returns error for invalid `camera_id`
- [ ] Test `capture_frame` returns error when `read()` fails
- [ ] Test `close_camera` returns `{"status": "closed"}` and releases capture
- [ ] Test `close_camera` returns error for invalid handle

## Implementation Notes

### CLI Update

File to modify: `src/aprilcam/cli/__init__.py`

1. Update the SUBCOMMANDS dict entry for `"mcp"`:

```python
"mcp": {
    "help": "Start the MCP server",
    "module": "aprilcam.mcp_server",
},
```

2. Remove the special-case block (lines 64-66):

```python
# Remove this:
if args.command == "mcp":
    print("MCP server not yet implemented")
    sys.exit(0)
```

The existing lazy-import dispatcher (lines 68-73) will handle importing
and calling `main()` on the module automatically.

### Test File

File to create: `tests/test_mcp_server.py`

Use `unittest.mock.MagicMock` for capture objects. Mock `isOpened()` to
return `True`, `read()` to return `(True, numpy_array)`, `release()` as
a no-op.

For tool handler tests, mock `camutil.list_cameras` and
`cv2.VideoCapture` to avoid real hardware. Call the tool handler
functions directly (they are async, use `pytest-asyncio` or
`asyncio.run()`).

Structure:

```python
class TestCameraRegistry:
    # Pure unit tests, no mocking needed beyond fake capture objects

class TestListCamerasTool:
    # Mock camutil.list_cameras

class TestOpenCameraTool:
    # Mock cv2.VideoCapture, ScreenCaptureMSS

class TestCaptureFrameTool:
    # Mock registry.get() and cap.read()

class TestCloseCameraTool:
    # Mock registry.close()
```

## Testing

- **Existing tests to run**: `uv run pytest` -- all existing tests must still pass.
- **New tests to write**: `tests/test_mcp_server.py` (described above).
- **Verification command**: `uv run pytest tests/test_mcp_server.py -v`
