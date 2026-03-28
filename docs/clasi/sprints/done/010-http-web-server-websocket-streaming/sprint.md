---
id: '010'
title: HTTP Web Server & WebSocket Streaming
status: done
branch: sprint/010-http-web-server-websocket-streaming
use-cases:
- SUC-010-001
- SUC-010-002
- SUC-010-003
- SUC-010-004
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 010: HTTP Web Server & WebSocket Streaming

## Goals

Add a new `aprilcam web` CLI subcommand that starts an HTTP server
exposing all AprilCam capabilities over three interfaces:

1. **REST API** — Plain HTTP endpoints mirroring every existing MCP tool.
   The root page returns human/agent-readable instructions.
2. **MCP over SSE** — Standard MCP protocol using Streamable HTTP / SSE
   transport so remote MCP clients can connect without stdio.
3. **WebSocket tag streaming** — Clients open a WebSocket connection and
   receive live tag detection data pushed in real time.

## Problem

The existing `aprilcam mcp` command uses stdio transport, which requires
the MCP client to spawn the server as a subprocess on the same machine.
There is no way for remote clients, web applications, or non-MCP agents
to access camera/tag data over the network. There is also no push-based
streaming interface for real-time tag data.

## Solution

Create a new module `src/aprilcam/web_server.py` (and optionally
`src/aprilcam/cli/web_cli.py`) that starts a Starlette/Uvicorn HTTP
server on port **17439** with:

- **`GET /`** — Returns a JSON instruction document describing all
  available endpoints, their parameters, and how to use them.
- **`/api/*`** — REST endpoints that mirror the 35 existing MCP tools.
  Each tool becomes a POST endpoint (e.g., `POST /api/list_cameras`).
  Request body is JSON matching the tool's parameters. Response is JSON.
  Image-returning endpoints support `format=base64` (inline in JSON) or
  `format=file` (returns JPEG binary with appropriate Content-Type).
- **`/mcp`** — MCP Streamable HTTP endpoint using the `mcp` Python SDK's
  built-in SSE transport support.
- **`/ws/tags/{source_id}`** — WebSocket endpoint. Once connected, the
  server pushes JSON tag detection frames as they arrive from the
  detection loop's ring buffer.

The web server reuses the same registries (CameraRegistry,
PlayfieldRegistry, detection_registry, etc.) as the MCP server — in
fact, the REST endpoints delegate to the same handler functions.

## Success Criteria

- `aprilcam web` starts a server on port 17439 (configurable via `--port`).
- `GET /` returns valid JSON describing all endpoints.
- All 35 MCP tools are accessible as `POST /api/<tool_name>` with correct
  JSON request/response.
- MCP clients can connect via SSE transport at `/mcp`.
- WebSocket clients can connect to `/ws/tags/{source_id}` and receive
  streaming tag data.
- Existing `aprilcam mcp` (stdio) continues to work unchanged.

## Scope

### In Scope

- New `aprilcam web` CLI subcommand with `--port` and `--host` options.
- REST API endpoints for all existing MCP tools.
- Root endpoint with API documentation/instructions.
- MCP SSE/Streamable HTTP transport at `/mcp`.
- WebSocket endpoint for streaming tag detections.
- Starlette + Uvicorn as the HTTP framework (lightweight, async-native,
  good WebSocket support).
- New dependency: `uvicorn`, `starlette` (or use `mcp` SDK's built-in
  SSE server if it bundles these).

### Out of Scope

- Authentication/authorization (local development tool).
- HTML UI or dashboard.
- HTTPS/TLS (use a reverse proxy if needed).
- Modifying the existing stdio MCP server.
- Rate limiting or request throttling.

## Test Strategy

- Unit tests for REST endpoint routing and JSON serialization.
- Integration tests that start the server, make HTTP requests, and
  verify responses (using httpx or starlette.testclient).
- WebSocket integration test that connects, starts detection on a mock
  source, and verifies streamed messages.
- Manual test with a real camera to verify end-to-end.

## Architecture Notes

- The web server imports and reuses the same internal functions as the
  MCP server — the tool implementations are not duplicated.
- Starlette is chosen because the `mcp` Python SDK already uses it for
  SSE transport (dependency already available).
- WebSocket tag streaming reads from the same ring buffer used by
  `get_tags` / `get_tag_history`.
- Port 17439 chosen to avoid common port conflicts.
- The server runs with Uvicorn in a single process (camera hardware
  requires single-process access).

## GitHub Issues

(None yet.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

1. **001** — Extract shared handler functions from mcp_server.py
2. **002** — Create web server with REST API endpoints (depends: 001)
3. **003** — Add MCP SSE transport endpoint (depends: 002)
4. **004** — Add WebSocket tag streaming endpoint (depends: 002)
5. **005** — CLI subcommand and integration tests (depends: 002, 003, 004)
