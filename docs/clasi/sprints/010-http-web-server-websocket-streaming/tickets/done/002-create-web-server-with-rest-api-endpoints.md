---
id: '002'
title: Create web server with REST API endpoints
status: done
use-cases:
- SUC-010-001
- SUC-010-002
depends-on:
- '001'
github-issue: ''
todo: ''
---

# Create web server with REST API endpoints

## Description

Create `src/aprilcam/web_server.py` with a Starlette application that:

1. `GET /` — Returns JSON API discovery document listing all endpoints,
   their parameters (name, type, default), descriptions, and usage
   instructions for AI agents.
2. `POST /api/<tool_name>` — REST endpoints for each of the ~12
   extracted handler functions. Request body is JSON matching the tool's
   parameters. Response is JSON matching the tool's return structure.
   Image endpoints with `format=file` return binary JPEG with
   `Content-Type: image/jpeg`.

Error contract: 400 for bad params, 404 for unknown tool, 500 for
internal errors. Body: `{"error": "..."}`.

Add `starlette` and `uvicorn` to pyproject.toml dependencies (may
already be transitive deps of `mcp`).

## Acceptance Criteria

- [ ] `web_server.py` creates a Starlette app with REST routes
- [ ] `GET /` returns valid JSON API documentation
- [ ] All ~12 tools accessible as `POST /api/<tool_name>`
- [ ] JSON request/response matches MCP tool signatures
- [ ] Image endpoints support binary JPEG response
- [ ] Proper HTTP status codes for errors

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test REST endpoints using `starlette.testclient`
- **Verification command**: `uv run pytest`
