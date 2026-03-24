---
timestamp: '2026-03-24T05:21:47'
parent: team-lead
child: technical-lead
scope: /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/007-polish-packaging
sprint: 007-polish-packaging
context_documents:
- docs/clasi/sprints/007-polish-packaging/sprint.md
- docs/clasi/sprints/007-polish-packaging/architecture-update.md
- docs/clasi/sprints/007-polish-packaging/usecases.md
result: "success \u2014 6 tickets created"
files_modified: []
---

# Dispatch: team-lead → technical-lead

Create tickets for Sprint 007: Polish &amp; Packaging. Use create_ticket with sprint_id="007".

Keep it focused — skip performance profiling and ring buffer memory caps (premature optimization). Focus on the practical stuff.

### Ticket 001: Error handling audit across all MCP tools
- Review every MCP tool handler for unguarded exceptions
- Wrap hardware interactions with try/except
- Return structured {"error": ...} for all failure cases
- Handle camera disconnection gracefully
- depends-on: []

### Ticket 002: MCP tool documentation — docstrings and parameter descriptions
- Update every @server.tool() function with descriptive docstrings
- Include parameter descriptions and example response shapes
- depends-on: []

### Ticket 003: pyproject.toml cleanup and entry point consolidation
- Remove legacy standalone entry points (taggen, arucogen, homocal, cameras, playfield, atscreencap, aprilcap, apriltest)
- Keep only `aprilcam` entry point
- Update metadata: author, description, keywords, classifiers, URLs
- depends-on: []

### Ticket 004: Remove deprecated CLI modules
- Remove or archive dead CLI modules (atscreencap_cli, apriltest_cli, etc.)
- Keep taggen and arucogen as subcommands of aprilcam
- Clean up unused imports
- depends-on: ["003"]

### Ticket 005: README
- Installation instructions (pipx install aprilcam)
- MCP server usage
- CLI subcommand reference
- Quick-start example
- depends-on: ["003"]

### Ticket 006: pipx install verification
- Test that pipx install . works in a clean environment
- Test that aprilcam mcp starts
- Test that aprilcam --help works
- depends-on: ["003", "004"]

Write substantive acceptance criteria for each.

## Context Documents

- `docs/clasi/sprints/007-polish-packaging/sprint.md`
- `docs/clasi/sprints/007-polish-packaging/architecture-update.md`
- `docs/clasi/sprints/007-polish-packaging/usecases.md`
