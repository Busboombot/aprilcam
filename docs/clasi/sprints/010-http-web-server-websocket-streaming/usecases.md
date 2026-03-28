---
sprint: "010"
status: done
---

# Use Cases — Sprint 010: HTTP Web Server & WebSocket Streaming

## SUC-010-001: REST API Access

**Actor**: AI agent or developer tool (HTTP client)

**Precondition**: `aprilcam web` is running on port 17439.

**Flow**:
1. Client sends `GET /` and receives a JSON document listing all
   available API endpoints with their parameters and descriptions.
2. Client sends `POST /api/list_cameras` with empty JSON body.
   Server returns JSON array of available cameras.
3. Client sends `POST /api/open_camera` with `{"index": 3}`.
   Server returns `{"camera_id": "cam_3"}`.
4. Client sends `POST /api/capture_frame` with
   `{"camera_id": "cam_3", "format": "base64"}`.
   Server returns JSON with base64-encoded JPEG image data.
5. Client sends `POST /api/start_detection` with
   `{"source_id": "cam_3"}`.
   Server returns `{"source_id": "cam_3", "status": "started"}`.
6. Client sends `POST /api/get_tags` with `{"source_id": "cam_3"}`.
   Server returns JSON with current tag detections.
7. Client sends `POST /api/close_camera` with `{"camera_id": "cam_3"}`.

**Postcondition**: Client has used the full camera lifecycle over HTTP.

**Acceptance Criteria**:
- [ ] All 35 MCP tools are available as `POST /api/<tool_name>`
- [ ] Request bodies are JSON matching tool parameter signatures
- [ ] Responses are JSON matching the tool's return structure
- [ ] Image endpoints with `format=file` return binary JPEG with
  `Content-Type: image/jpeg`

## SUC-010-002: API Discovery

**Actor**: AI agent or developer

**Precondition**: `aprilcam web` is running.

**Flow**:
1. Client sends `GET /`.
2. Server returns a JSON document containing:
   - Server name and version.
   - List of all endpoints with path, method, parameters (name, type,
     default, description), and return type.
   - Brief usage instructions suitable for an AI agent.

**Postcondition**: Client has enough information to use any endpoint
without external documentation.

**Acceptance Criteria**:
- [ ] Response is valid JSON
- [ ] Every REST endpoint is listed with its parameters
- [ ] Parameter types and defaults are included

## SUC-010-003: MCP over SSE Transport

**Actor**: Remote MCP client (e.g., Claude Code on another machine)

**Precondition**: `aprilcam web` is running.

**Flow**:
1. MCP client connects to `http://host:17439/mcp` using the MCP
   Streamable HTTP transport.
2. Client sends MCP tool calls (e.g., `list_cameras`, `get_tags`).
3. Server processes them using the same tool implementations as the
   stdio MCP server and returns MCP responses over SSE.

**Postcondition**: Remote MCP client has full access to all AprilCam
tools without needing stdio subprocess access.

**Acceptance Criteria**:
- [ ] MCP SDK's SSE transport works at `/mcp`
- [ ] All tools available through stdio are also available through SSE
- [ ] Multiple MCP clients can connect simultaneously

## SUC-010-004: WebSocket Tag Streaming

**Actor**: Web application, dashboard, or real-time client

**Precondition**: `aprilcam web` is running. A detection loop is active
on a camera (started via REST or MCP).

**Flow**:
1. Client opens a WebSocket connection to
   `ws://host:17439/ws/tags/{source_id}`.
2. Server immediately begins pushing JSON messages with tag detection
   data as new frames are processed.
3. Each message contains: `timestamp`, `frame_index`, and `tags` array
   (same structure as `get_tags` response).
4. Client processes messages in real time.
5. Client closes the WebSocket when done.

**Postcondition**: Client has received a real-time stream of tag data.

**Acceptance Criteria**:
- [ ] WebSocket connection is established successfully
- [ ] Messages arrive continuously while detection is running
- [ ] Message format matches `get_tags` JSON structure
- [ ] Server handles client disconnection gracefully
- [ ] Multiple WebSocket clients can connect to the same source
