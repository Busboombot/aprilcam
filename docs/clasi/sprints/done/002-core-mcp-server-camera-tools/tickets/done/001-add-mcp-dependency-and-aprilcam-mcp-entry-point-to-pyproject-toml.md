---
id: "001"
title: "Add mcp dependency and aprilcam-mcp entry point to pyproject.toml"
status: todo
use-cases:
  - SUC-005
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Add mcp dependency and aprilcam-mcp entry point to pyproject.toml

## Description

Move the `mcp` package from `[project.optional-dependencies]` to core
`[project.dependencies]` and add a new `aprilcam-mcp` script entry point
that invokes `aprilcam.mcp_server:main`. This is a prerequisite for all
other tickets because it makes the MCP SDK available at import time and
provides the standalone entry point that host processes (Claude Desktop,
etc.) will use to launch the server.

The `mcp` optional-dependency group should be removed since `mcp` will
now be a core dependency.

## Acceptance Criteria

- [ ] `mcp>=1.0` appears in `[project.dependencies]`
- [ ] `mcp>=1.0` is removed from `[project.optional-dependencies]`
- [ ] `aprilcam-mcp = "aprilcam.mcp_server:main"` appears in `[project.scripts]`
- [ ] `uv sync` or `pip install -e .` installs the `mcp` package
- [ ] Existing entry point `aprilcam` is unchanged

## Implementation Notes

File to modify: `pyproject.toml`

1. In `[project.dependencies]`, add `"mcp>=1.0"` after the existing
   entries.
2. In `[project.scripts]`, add
   `aprilcam-mcp = "aprilcam.mcp_server:main"`.
3. Remove the `mcp` key from `[project.optional-dependencies]` (keep
   `playfield` if it exists).
4. Note: the `aprilcam.mcp_server` module does not exist yet -- that is
   ticket 002/003. This ticket only sets up the packaging metadata. The
   entry point will fail to resolve until ticket 003 is complete, which
   is expected.

## Testing

- **Existing tests to run**: `uv run pytest` -- ensure no regressions
  from dependency changes.
- **New tests to write**: None (packaging metadata only).
- **Verification command**: `uv sync && uv run python -c "import mcp; print(mcp.__version__)"`
