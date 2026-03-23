---
id: "003"
title: "Implement MCP server with camera tool handlers"
status: todo
use-cases:
  - SUC-001
  - SUC-002
  - SUC-003
  - SUC-004
  - SUC-005
depends-on:
  - "001"
  - "002"
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Implement MCP server with camera tool handlers

## Description

Add the MCP server setup and four tool handler functions to
`src/aprilcam/mcp_server.py` (which already contains `CameraRegistry`
from ticket 002). This is the core deliverable of the sprint: a working
MCP server that exposes camera management tools over stdio JSON-RPC.

## Acceptance Criteria

- [ ] MCP `Server` instance named `"aprilcam"` is created at module level
- [ ] `list_cameras` tool is registered and returns JSON array of `{index, name, backend}` objects (SUC-001)
- [ ] `list_cameras` returns an empty array (not an error) when no cameras are found (SUC-001)
- [ ] `open_camera` tool accepts optional `index`, `pattern`, `source`, and `backend` parameters (SUC-002)
- [ ] `open_camera` opens `cv2.VideoCapture` for index/pattern, `ScreenCaptureMSS` for `source="screen"` (SUC-002)
- [ ] `open_camera` returns `{"camera_id": "<uuid>"}` on success (SUC-002)
- [ ] `open_camera` returns a clear error when the camera cannot be opened (SUC-002)
- [ ] `capture_frame` tool accepts `camera_id`, optional `format` ("base64"|"file"), and optional `quality` (SUC-003)
- [ ] `capture_frame` default format returns base64 JPEG as MCP `ImageContent` (SUC-003)
- [ ] `capture_frame` with `format="file"` writes a temp file and returns the path as `TextContent` (SUC-003)
- [ ] `capture_frame` JPEG quality defaults to 85, is configurable via `quality` param (SUC-003)
- [ ] `capture_frame` returns error for invalid `camera_id` or failed `read()` (SUC-003)
- [ ] `close_camera` tool releases the capture and removes the handle (SUC-004)
- [ ] `close_camera` returns `{"status": "closed"}` on success (SUC-004)
- [ ] `close_camera` returns error for invalid/already-closed handle (SUC-004)
- [ ] `main()` function runs the server using `mcp.server.stdio.stdio_server()` (SUC-005)
- [ ] Server responds to MCP `initialize` handshake (SUC-005)
- [ ] `tools/list` returns all four tools with schemas (SUC-005)
- [ ] All open cameras are released on server shutdown via `CameraRegistry.close_all()` (SUC-005)

## Implementation Notes

File to modify: `src/aprilcam/mcp_server.py`

### Server setup

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, ImageContent

server = Server("aprilcam")
registry = CameraRegistry()
```

### Tool: `list_cameras`

```python
@server.tool()
async def list_cameras() -> list[TextContent]:
    from aprilcam.camutil import list_cameras as _list_cameras
    cameras = _list_cameras(quiet=True)
    result = [{"index": c.index, "name": c.name, "backend": c.backend}
              for c in cameras]
    return [TextContent(type="text", text=json.dumps(result))]
```

### Tool: `open_camera`

- If `source == "screen"`: create `ScreenCaptureMSS(monitor=monitor or 1)`
- If `pattern` is provided: call `select_camera_by_pattern()` to resolve
  index, using `list_cameras(quiet=True)` to get the camera list
- If `index` is provided: use directly
- If `backend` is provided: resolve backend string to cv2 constant, pass
  to `cv2.VideoCapture(index, backend_id)`
- Verify `isOpened()`, register in `CameraRegistry`, return handle
- On failure: return error text

### Tool: `capture_frame`

- `registry.get(camera_id)` -- KeyError becomes error response
- `cap.read()` -- check ret is True
- `cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])`
- If format == "base64" (default): return `ImageContent` with
  base64-encoded JPEG data, `mimeType="image/jpeg"`
- If format == "file": write to `tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)`,
  return path as `TextContent`

### Tool: `close_camera`

- `registry.close(camera_id)` -- KeyError becomes error response
- Return `TextContent` with `{"status": "closed"}`

### `main()` function

```python
async def _run():
    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options()
            )
        finally:
            registry.close_all()

def main():
    import asyncio
    asyncio.run(_run())
```

## Testing

- **Existing tests to run**: `uv run pytest`
- **New tests to write**: See ticket 004.
- **Verification command**: `echo '{}' | timeout 2 uv run aprilcam-mcp` (should start without crash)
