---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 007 Use Cases

## SUC-001: End-to-End MCP Server Lifecycle Test
Parent: UC-002 (Core MCP Server & Camera Tools)

- **Actor**: Developer / CI system
- **Preconditions**: AprilCam package is installed; no real camera required
- **Main Flow**:
  1. Run `pytest` which launches the MCP server as a subprocess over stdio.
  2. Send an MCP `initialize` request and receive a successful response.
  3. Send `tools/list` and verify all expected tools are listed.
  4. Call `list_cameras` (returns empty list or mocked cameras).
  5. Call `open_camera` with a mocked camera index, receive a camera_id.
  6. Call `capture_frame` with the camera_id, receive a base64 image.
  7. Call `close_camera` with the camera_id, receive success.
  8. Shut down the MCP server cleanly.
- **Postconditions**: All MCP tool calls returned structured responses;
  server exited with code 0; no unhandled exceptions in server logs
- **Acceptance Criteria**:
  - [ ] Integration test exercises initialize, tools/list, list_cameras,
        open_camera, capture_frame, close_camera in sequence
  - [ ] Test passes with mocked camera (no real hardware)
  - [ ] Server shuts down cleanly without orphan processes

## SUC-002: Graceful Error on Missing Camera
Parent: UC-002 (Core MCP Server & Camera Tools)

- **Actor**: AI agent calling MCP tools
- **Preconditions**: MCP server is running; requested camera index does
  not exist
- **Main Flow**:
  1. Agent calls `open_camera` with an invalid camera index (e.g., 99).
  2. Server attempts to open the camera, fails.
  3. Server returns a structured MCP error response with a descriptive
     message (e.g., "Camera index 99 not found").
- **Postconditions**: Agent receives a parseable error; server remains
  operational and can handle subsequent requests
- **Acceptance Criteria**:
  - [ ] `open_camera` with invalid index returns structured error, not
        an unhandled exception
  - [ ] Server continues to accept requests after the error
  - [ ] Error message includes the requested camera index

## SUC-003: Camera Disconnection Mid-Detection-Loop
Parent: UC-004 (Tag Detection Loop & Ring Buffer)

- **Actor**: AI agent running a detection loop
- **Preconditions**: Detection loop is running on an open camera
- **Main Flow**:
  1. Agent has started a detection loop via `start_detection`.
  2. Camera is physically disconnected (simulated by mock returning
     None frames).
  3. Detection loop detects the read failure.
  4. Detection loop stops automatically.
  5. Server sends/stores an error event indicating camera disconnection.
  6. Agent calls `query_tags` and receives an error indicating the loop
     has stopped due to camera disconnection.
- **Postconditions**: Detection loop is cleanly stopped; no thread leaks;
  agent can re-open a camera and start a new loop
- **Acceptance Criteria**:
  - [ ] Detection loop stops within 1 second of camera disconnection
  - [ ] Agent receives a structured error explaining the disconnection
  - [ ] No background threads are leaked after disconnection
  - [ ] Agent can start a new detection loop after re-opening a camera

## SUC-004: Clean pipx Installation and MCP Startup
Parent: UC-001 (Project Restructure & CLI Foundation)

- **Actor**: Developer installing AprilCam
- **Preconditions**: Python >= 3.9 and pipx are installed; no prior
  AprilCam installation
- **Main Flow**:
  1. Developer runs `pipx install .` from the project root.
  2. pipx creates an isolated venv and installs the package.
  3. Developer runs `aprilcam mcp` from the command line.
  4. MCP server starts, prints no errors, and waits for stdio input.
  5. Developer sends a JSON-RPC `initialize` request via stdin.
  6. Server responds with capabilities and server info.
  7. Developer sends EOF; server exits cleanly.
- **Postconditions**: `aprilcam` is the only installed entry point;
  `aprilcam mcp` works; no leftover `taggen`, `arucogen`, etc. scripts
- **Acceptance Criteria**:
  - [ ] `pipx install .` completes without errors
  - [ ] Only `aprilcam` appears as an installed script (no legacy
        standalone commands)
  - [ ] `aprilcam mcp` starts the MCP server and responds to initialize
  - [ ] `aprilcam taggen`, `aprilcam arucogen`, `aprilcam cameras`
        subcommands still work

## SUC-005: MCP Tool Documentation Completeness
Parent: UC-002 (Core MCP Server & Camera Tools)

- **Actor**: AI agent discovering available tools
- **Preconditions**: MCP server is running
- **Main Flow**:
  1. Agent sends `tools/list` to the MCP server.
  2. Server returns the tool list with descriptions for every tool.
  3. Each tool description includes: purpose, parameter names with
     types and descriptions, return value shape.
  4. Agent can determine how to call any tool from the description
     alone, without external documentation.
- **Postconditions**: All tools are self-documenting via MCP metadata
- **Acceptance Criteria**:
  - [ ] Every MCP tool has a non-empty description
  - [ ] Every parameter on every tool has a type and description
  - [ ] A reference doc (`docs/mcp-tools.md`) exists with example
        request/response JSON for each tool
  - [ ] Test validates that `tools/list` returns descriptions for all
        registered tools

## SUC-006: Detection Loop Performance Under Sustained Load
Parent: UC-004 (Tag Detection Loop & Ring Buffer)

- **Actor**: AI agent running continuous detection
- **Preconditions**: Detection loop is running with a mocked camera
  producing frames at 30fps
- **Main Flow**:
  1. Detection loop processes 1000 consecutive frames.
  2. Ring buffer stores detection results up to configured max.
  3. Agent calls `query_tags` and `query_tag_history` during the loop.
  4. MCP tool responses return within 50ms despite ongoing detection.
  5. Ring buffer memory stays within configured cap.
- **Postconditions**: Detection loop maintained frame rate; MCP
  responses were not blocked; memory usage stayed bounded
- **Acceptance Criteria**:
  - [ ] Per-frame detection latency is profiled and documented
  - [ ] MCP tool responses during active detection complete in < 50ms
  - [ ] Ring buffer memory does not grow unbounded (configurable cap)
  - [ ] Frame capture does not block MCP request handling
