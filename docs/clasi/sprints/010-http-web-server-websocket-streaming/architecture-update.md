---
sprint: "010"
status: draft
---

# Architecture Update -- Sprint 010: HTTP Web Server & WebSocket Streaming

## What Changed

### New Files

- **`src/aprilcam/web_server.py`** — Main web server module. Creates a
  Starlette application with three route groups:
  - `GET /` — API discovery endpoint returning JSON instructions.
  - `POST /api/{tool_name}` — REST endpoints mirroring all 35 MCP tools.
  - `/mcp` — MCP Streamable HTTP / SSE transport endpoint.
  - `/ws/tags/{source_id}` — WebSocket endpoint for live tag streaming.

- **`src/aprilcam/cli/web_cli.py`** — CLI subcommand `aprilcam web`
  that starts the Uvicorn server.

### Modified Files

- **`pyproject.toml`** — Add dependencies: `uvicorn`, `starlette`,
  `websockets`. Add `web` CLI entry point.

- **`src/aprilcam/cli/__init__.py`** — Register the `web` subcommand.

- **`src/aprilcam/mcp_server.py`** — Refactor tool handler functions to
  be callable from both MCP and REST contexts. Extract the core logic
  from each `@server.tool()` function into a shared internal function
  that returns plain Python dicts/bytes, with the MCP wrapper handling
  TextContent/ImageContent conversion. The REST endpoints call the same
  internal functions directly.

### Architecture Approach

The key design decision is **shared internals, separate transports**:

```
┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│ stdio MCP   │  │ SSE/HTTP MCP│  │  REST API    │
│ (existing)  │  │ (new)       │  │  (new)       │
└──────┬──────┘  └──────┬──────┘  └──────┬───────┘
       │                │               │
       └────────────────┼───────────────┘
                        │
              ┌─────────┴─────────┐
              │  Tool Handlers    │
              │  (shared logic)   │
              └─────────┬─────────┘
                        │
       ┌────────────────┼───────────────┐
       │                │               │
┌──────┴──────┐  ┌──────┴──────┐  ┌─────┴──────┐
│ Camera      │  │ Detection   │  │ Playfield  │
│ Registry    │  │ Registry    │  │ Registry   │
└─────────────┘  └─────────────┘  └────────────┘
```

The web server creates a single Starlette `app` that mounts:
1. The MCP SDK's SSE transport handler at `/mcp` (reusing the existing
   `server` object from `mcp_server.py`).
2. REST API routes at `/api/*` that call shared handler functions.
3. A WebSocket route at `/ws/tags/{source_id}`.

### WebSocket Tag Streaming

The WebSocket endpoint polls the detection ring buffer at ~30fps and
pushes new frames to all connected clients. Each source_id can have
multiple WebSocket subscribers. The server uses asyncio tasks to manage
per-connection streaming loops.

Message format (same as `get_tags` response):
```json
{
  "timestamp": 12345.678,
  "frame_index": 42,
  "tags": [
    {"id": 0, "center_px": [100, 200], ...},
    {"id": 1, "center_px": [300, 400], ...}
  ],
  "source_id": "cam_3"
}
```

### Server Startup

`aprilcam web` starts Uvicorn with:
- Host: `0.0.0.0` (configurable via `--host`)
- Port: `17439` (configurable via `--port`)
- Single worker (camera hardware requires single-process)
- Log level: info

## Why

The stdio MCP transport requires the client to spawn the server as a
local subprocess. This prevents:
- Remote access from other machines on the network.
- Web applications from consuming camera/tag data.
- Non-MCP clients (plain HTTP tools, curl, scripts) from using the API.
- Real-time push-based streaming (stdio MCP is request/response only).

## Impact on Existing Components

- **`mcp_server.py`** is refactored to extract shared handler logic, but
  the stdio MCP interface remains identical. No breaking changes.
- **New dependencies** (`uvicorn`, `starlette`, `websockets`) are added.
  `starlette` may already be a transitive dependency of the `mcp` SDK.
- **CLI** gains one new subcommand (`web`). Existing subcommands unchanged.

## Migration Concerns

None. This is purely additive. The existing stdio MCP transport continues
to work. No data migration needed.

## Decisions

1. **Handler refactoring scope**: This sprint extracts a subset of
   handlers — camera lifecycle (list/open/close/capture), detection
   (start/stop/get_tags/get_tag_history), and playfield
   (create/get_info) — roughly 12 tools. This is enough to prove the
   pattern and deliver a useful REST API. Remaining tools will be
   extracted in a follow-up sprint.

2. **WebSocket backpressure**: Drop old frames, always send latest.
   This is a real-time stream for live tag data, not a replay log.
   Slow clients miss intermediate frames but always see the current state.

## Registry Sharing

The web server imports module-level singletons directly from
`mcp_server.py` (e.g., `from aprilcam.mcp_server import registry,
detection_registry`). This is safe because the server runs as a single
Uvicorn worker process — camera hardware requires single-process access.
If a future sprint needs multi-process, these registries would need to
move to a shared module.

## REST Error Contract

REST endpoints use standard HTTP status codes:
- **200** — Success. Body is JSON matching the tool's response.
- **400** — Bad request (missing/invalid parameters). Body: `{"error": "..."}`.
- **404** — Unknown endpoint. Body: `{"error": "Unknown tool '...'"}`.
- **500** — Internal server error. Body: `{"error": "..."}`.

Image endpoints with `format=file` return HTTP 200 with
`Content-Type: image/jpeg` and binary body on success.
