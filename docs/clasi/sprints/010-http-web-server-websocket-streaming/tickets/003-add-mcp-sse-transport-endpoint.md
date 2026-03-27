---
id: "003"
title: "Add MCP SSE transport endpoint"
status: todo
use-cases: [SUC-010-003]
depends-on: ["002"]
github-issue: ""
todo: ""
---

# Add MCP SSE transport endpoint

## Description

Mount the MCP SDK's Streamable HTTP / SSE transport handler at `/mcp`
on the Starlette app created in ticket 002. This reuses the existing
`server` object from `mcp_server.py` so all 35 MCP tools (not just
the extracted subset) are available over SSE.

The MCP Python SDK provides built-in SSE transport support. This
ticket wires it into the Starlette app.

## Acceptance Criteria

- [ ] MCP SSE transport is mounted at `/mcp`
- [ ] Remote MCP clients can connect and call tools via SSE
- [ ] All 35 MCP tools are accessible (not just the REST subset)
- [ ] Multiple MCP clients can connect simultaneously

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test that the `/mcp` endpoint responds to
  SSE connection attempts
- **Verification command**: `uv run pytest`
