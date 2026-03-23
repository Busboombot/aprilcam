---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 002 Use Cases

## SUC-001: Agent Discovers Available Cameras

Parent: UC-001 (Agent-Driven Camera Management)

- **Actor**: AI agent connected to the MCP server
- **Preconditions**: The MCP server is running (`aprilcam mcp` process is
  alive and stdio transport is connected). At least one camera or screen
  capture source is available on the host.
- **Main Flow**:
  1. Agent sends an MCP `tools/call` request for `list_cameras`.
  2. Server calls `camutil.list_cameras()` with default parameters.
  3. Server returns a JSON array of objects, each containing `index` (int),
     `name` (str), and `backend` (str or null).
- **Postconditions**: Agent has a structured list of available cameras and
  can choose one to open by index or name pattern.
- **Acceptance Criteria**:
  - [ ] `list_cameras` tool is listed in `tools/list` response
  - [ ] Response is a JSON array of `{index, name, backend}` objects
  - [ ] Works when no cameras are found (returns empty array, no error)
  - [ ] Does not open or hold any camera resources

## SUC-002: Agent Opens a Camera by Index

Parent: UC-001 (Agent-Driven Camera Management)

- **Actor**: AI agent
- **Preconditions**: MCP server is running. Agent knows a valid camera
  index (e.g., from `list_cameras`).
- **Main Flow**:
  1. Agent sends `open_camera` with `{"index": 0}`.
  2. Server creates a `cv2.VideoCapture` for the given index.
  3. Server verifies `isOpened()` returns true.
  4. Server registers the capture in `CameraRegistry` under a new UUID
     handle.
  5. Server returns `{"camera_id": "<uuid>"}`.
- **Postconditions**: A camera is open and held by the registry. The agent
  has a handle for subsequent `capture_frame` / `close_camera` calls.
- **Alternative Flows**:
  - **Open by name pattern**: Agent sends `{"pattern": "FaceTime"}`.
    Server calls `select_camera_by_pattern()` to resolve the index, then
    proceeds as above.
  - **Open screen capture**: Agent sends `{"source": "screen"}` (or
    `{"source": "screen", "monitor": 1}`). Server creates a
    `ScreenCaptureMSS` instance instead of `VideoCapture`.
  - **Open with backend**: Agent sends `{"index": 0, "backend": "avfoundation"}`.
    Server uses the specified backend API.
- **Error Flows**:
  - Camera index does not exist or fails to open: server returns an error
    with a descriptive message.
  - Pattern matches no camera: server returns an error.
- **Acceptance Criteria**:
  - [ ] `open_camera` returns a `camera_id` string (UUID)
  - [ ] Supports opening by `index`, by `pattern`, or by `source: "screen"`
  - [ ] Optional `backend` parameter selects the capture backend
  - [ ] Returns a clear error when the camera cannot be opened
  - [ ] Multiple cameras can be open simultaneously with different handles

## SUC-003: Agent Captures a Frame

Parent: UC-002 (Agent-Driven Frame Capture)

- **Actor**: AI agent
- **Preconditions**: Agent holds a valid `camera_id` from a prior
  `open_camera` call. The camera is still open.
- **Main Flow**:
  1. Agent sends `capture_frame` with `{"camera_id": "<uuid>"}`.
  2. Server retrieves the capture object from `CameraRegistry`.
  3. Server calls `read()` on the capture object.
  4. Server encodes the frame as JPEG.
  5. Server returns the image as base64-encoded data using MCP
     `ImageContent` type (default `format: "base64"`).
- **Alternative Flows**:
  - **File format**: Agent sends `{"camera_id": "<uuid>", "format": "file"}`.
    Server writes the JPEG to a temp file and returns the file path as
    `TextContent`.
  - **Quality parameter**: Agent optionally passes `{"quality": 90}` to
    control JPEG compression (default 85).
- **Error Flows**:
  - Invalid or expired `camera_id`: server returns an error.
  - `read()` returns `(False, ...)`: server returns an error indicating
    the camera failed to produce a frame.
- **Postconditions**: Agent has received image data (inline or as a file
  path). The camera remains open for further captures.
- **Acceptance Criteria**:
  - [ ] Default format returns base64 JPEG as MCP `ImageContent`
  - [ ] `format: "file"` writes a temp file and returns the path
  - [ ] JPEG quality is configurable (default 85)
  - [ ] Returns error for invalid `camera_id`
  - [ ] Returns error when `read()` fails
  - [ ] Works with both `VideoCapture` and `ScreenCaptureMSS` handles

## SUC-004: Agent Closes a Camera

Parent: UC-001 (Agent-Driven Camera Management)

- **Actor**: AI agent
- **Preconditions**: Agent holds a valid `camera_id`.
- **Main Flow**:
  1. Agent sends `close_camera` with `{"camera_id": "<uuid>"}`.
  2. Server retrieves the capture object from `CameraRegistry`.
  3. Server calls `release()` on the capture object.
  4. Server removes the handle from the registry.
  5. Server returns `{"status": "closed"}`.
- **Error Flows**:
  - Invalid or already-closed `camera_id`: server returns an error.
- **Postconditions**: The camera resource is released. The handle is
  invalidated; subsequent calls with this `camera_id` will fail.
- **Acceptance Criteria**:
  - [ ] `close_camera` releases the underlying capture object
  - [ ] Handle is removed from registry after close
  - [ ] Subsequent use of the same `camera_id` returns an error
  - [ ] Double-close returns a clear error (not a crash)

## SUC-005: Agent Starts the MCP Server

Parent: UC-003 (MCP Server Lifecycle)

- **Actor**: Host system or user launching the server
- **Preconditions**: `aprilcam` package is installed. The `mcp` Python
  SDK is available.
- **Main Flow**:
  1. User runs `aprilcam mcp` (or the dedicated entry point).
  2. The CLI creates an MCP `Server` instance and registers all tool
     handlers.
  3. The server enters the stdio event loop, reading JSON-RPC from stdin
     and writing responses to stdout.
  4. On receiving `initialize`, the server responds with its capabilities
     (including the four camera tools).
- **Shutdown Flow**:
  - When stdin closes (or the process receives SIGTERM), the server
    releases all open cameras via `CameraRegistry` cleanup and exits
    cleanly.
- **Postconditions**: The MCP server is running and ready to accept
  tool calls.
- **Acceptance Criteria**:
  - [ ] `aprilcam mcp` starts without error when `mcp` SDK is installed
  - [ ] Server responds to MCP `initialize` handshake
  - [ ] `tools/list` returns all four camera tools with schemas
  - [ ] Server exits cleanly when stdin is closed
  - [ ] All open cameras are released on shutdown
