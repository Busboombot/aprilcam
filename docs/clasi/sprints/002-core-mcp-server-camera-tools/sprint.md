---
id: "002"
title: "Core MCP Server & Camera Tools"
status: planning
branch: sprint/002-core-mcp-server-camera-tools
use-cases:
  - SUC-001
  - SUC-002
  - SUC-003
  - SUC-004
  - SUC-005
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 002: Core MCP Server & Camera Tools

## Goals

Implement an MCP (Model Context Protocol) server for AprilCam that exposes
camera management as structured tools callable by AI agents. This converts
AprilCam from a human-only CLI application into a programmatic service that
agents can drive via stdio transport.

Specifically:

1. Add the Python MCP SDK (`mcp`) as a dependency and create a server module.
2. Wire a new `aprilcam mcp` CLI subcommand that starts the MCP server on
   stdio.
3. Implement four camera management tools: `list_cameras`, `open_camera`,
   `close_camera`, and `capture_frame`.
4. Maintain a server-side registry of open camera handles so agents can
   manage multiple cameras across tool calls.
5. Support both hardware cameras (via OpenCV `VideoCapture`) and screen
   capture sources (via `ScreenCaptureMSS`) through the same handle
   abstraction.

## Problem

AprilCam's camera functionality is locked inside CLI commands and interactive
loops that require a human operator. An AI agent (e.g., Claude) cannot
enumerate cameras, open a feed, or grab a frame without shelling out and
parsing unstructured text. There is no programmatic API boundary between
the camera subsystem and consumers.

## Solution

Expose the camera subsystem through an MCP server using the Python MCP SDK's
stdio transport. The server will:

- Wrap `camutil.list_cameras()` as a tool returning structured JSON.
- Manage a `CameraRegistry` that maps opaque string handles (UUIDs) to open
  `VideoCapture` or `ScreenCaptureMSS` instances.
- Provide `open_camera` accepting an index, name pattern, or the literal
  `"screen"` to open a screen capture source.
- Provide `capture_frame` that reads a frame from a handle and returns it as
  either a base64-encoded JPEG or a path to a temp file.
- Provide `close_camera` to release resources.

The server is started by running `aprilcam mcp`, which delegates to
`src/aprilcam/mcp_server.py`. The CLI itself remains a thin entry point.

## Success Criteria

- `aprilcam mcp` starts an MCP server that responds to the `initialize`
  handshake over stdio.
- `list_cameras` tool returns a JSON array of `{index, name, backend}`.
- `open_camera` with a valid index returns a `camera_id` string. Opening
  with `"screen"` returns a handle to a `ScreenCaptureMSS` instance.
- `capture_frame` with a valid `camera_id` returns a base64 JPEG (default)
  or writes to a temp file and returns the path.
- `close_camera` releases the underlying capture object and invalidates the
  handle.
- Calling a tool with an invalid or stale `camera_id` returns a clear error.
- All four tools are discoverable via the MCP `tools/list` method.
- Unit tests cover the `CameraRegistry` and tool handler logic using mock
  capture objects.

## Scope

### In Scope

- New module `src/aprilcam/mcp_server.py` containing the MCP server, tool
  definitions, and `CameraRegistry` class.
- New CLI subcommand `aprilcam mcp` (requires converting the current
  single-command CLI into a subcommand-based CLI, or adding a dedicated
  entry point).
- `CameraRegistry` class: open/close/get handles, automatic cleanup.
- Four MCP tools: `list_cameras`, `open_camera`, `close_camera`,
  `capture_frame`.
- Base64 and temp-file return formats for `capture_frame`.
- Screen capture source support via `ScreenCaptureMSS`.
- Adding `mcp` (Python MCP SDK) to `pyproject.toml` dependencies.
- Unit tests for `CameraRegistry` and tool handlers with mocked cameras.

### Out of Scope

- AprilTag detection tools (Sprint 004).
- Homography/playfield tools (Sprint 003).
- Image processing tools (Sprint 005).
- Video streaming or continuous capture.
- HTTP/SSE transport (stdio only for now).
- Authentication or access control.
- Configuration of camera resolution or properties via MCP tools.

## Test Strategy

**Unit tests** (`tests/test_mcp_server.py`):

- `CameraRegistry`: test open, close, get, double-close, max-handles,
  cleanup-on-delete. Use a mock object implementing `isOpened()`, `read()`,
  and `release()`.
- Tool handlers: test each tool function with mocked registry and capture
  objects. Verify correct JSON schema of responses. Verify error responses
  for invalid handles.

**Integration tests** (manual or scripted):

- Start `aprilcam mcp` in a subprocess, send JSON-RPC `initialize` +
  `tools/list` over stdin, verify the four tools appear in the response.
- If a camera is available, exercise the full open/capture/close cycle.

**No GUI or display tests** are needed for this sprint.

## Architecture Notes

- The MCP server uses the `mcp` Python SDK (`mcp.server.Server`) with
  `stdio_server` transport. This is the same pattern used by CLASI and
  other MCP servers in this ecosystem.
- `CameraRegistry` is a plain Python class (not async) since OpenCV
  `VideoCapture` operations are synchronous. The MCP tool handlers will
  call registry methods directly from async tool functions (the actual
  I/O is fast enough that blocking briefly is acceptable; no thread pool
  needed for v1).
- Camera handles are UUID4 strings. The registry maps handle -> capture
  object. This avoids exposing raw integer indices across tool calls and
  makes handle reuse impossible.
- `capture_frame` encodes to JPEG by default (configurable quality).
  Base64 output uses the MCP `ImageContent` type for inline images.
  File output writes to `tempfile.mkstemp()` and returns the path as
  `TextContent`.
- Screen capture handles are opened via `ScreenCaptureMSS(monitor=1)`.
  They implement the same `isOpened()`/`read()`/`release()` interface
  as `VideoCapture`, so the registry treats them uniformly.
- The CLI entry point for `aprilcam mcp` will be a new script entry or
  a subcommand. The simplest approach is a new entry in
  `[project.scripts]` pointing to `aprilcam.mcp_server:main`.

## GitHub Issues

None linked yet.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
