---
id: "001"
title: "Extract shared handler functions from mcp_server.py"
status: todo
use-cases: [SUC-010-001]
depends-on: []
github-issue: ""
todo: ""
---

# Extract shared handler functions from mcp_server.py

## Description

Refactor ~12 core MCP tool handlers to separate business logic from
MCP-specific TextContent/ImageContent wrapping. Each tool becomes:

- `_handle_<name>(**params) -> dict | bytes` — Pure logic, returns
  plain Python dicts or bytes.
- `@server.tool() async def <name>(...) -> list[...]` — Thin MCP
  wrapper that calls the handler and wraps the result.

Tools to extract: list_cameras, open_camera, close_camera,
capture_frame, create_playfield, get_playfield_info, start_detection,
stop_detection, get_tags, get_tag_history, get_frame,
start_live_view, stop_live_view.

## Acceptance Criteria

- [ ] Each of the ~12 tools has a `_handle_*` function returning plain dicts
- [ ] MCP `@server.tool()` wrappers call the handler functions
- [ ] All existing MCP tests pass without modification
- [ ] stdio MCP transport works identically to before

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: None (refactor; existing tests verify behavior)
- **Verification command**: `uv run pytest`
