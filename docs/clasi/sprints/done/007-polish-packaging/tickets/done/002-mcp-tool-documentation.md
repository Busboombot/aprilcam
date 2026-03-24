---
id: '002'
title: MCP tool documentation
status: done
use-cases:
- SUC-002
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# MCP tool documentation

## Description

Update every `@server.tool()` function with descriptive docstrings so
that AI agents can discover how to use each tool from the MCP
`tools/list` response alone. Currently, tool descriptions are minimal
or missing, forcing callers to guess parameter semantics and response
shapes.

For each MCP tool:
1. Write a clear one-sentence summary as the first line of the docstring.
2. Add a "Parameters" section listing each parameter with its type,
   whether it is required or optional, and what it controls.
3. Add a "Returns" section describing the response shape (JSON
   structure with field names and types).
4. Note any side effects (e.g., `open_camera` allocates a resource
   that must be closed).

Additionally, create a `docs/mcp-tools.md` reference document with
example request/response JSON for each tool, organized by category
(Camera, Playfield, Detection, Image Processing, Composite).

## Acceptance Criteria

- [ ] Every MCP tool function has a docstring with: summary, parameter
      descriptions (name, type, required/optional, meaning), and
      return value shape
- [ ] `tools/list` MCP response includes non-empty descriptions for
      every registered tool
- [ ] Every parameter on every tool has a type annotation and
      description visible in MCP metadata
- [ ] `docs/mcp-tools.md` exists with example request and response
      JSON for each tool
- [ ] Tools are organized by category in the reference doc (Camera
      Management, Playfield, Detection & Tracking, Image Processing,
      Compositing)
- [ ] Docstrings are consistent in format across all tools (same
      heading style, same parameter notation)

## Testing

- **Existing tests to run**: `uv run pytest` -- full suite
- **New tests to write**:
  - Test that `tools/list` returns a non-empty description for every
    registered tool (integration test via MCP subprocess)
  - Test that every tool has at least one parameter with a description
    in its schema
- **Verification command**: `uv run pytest`
