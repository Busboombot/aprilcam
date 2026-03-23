---
timestamp: '2026-03-23T22:05:21'
parent: team-lead
child: technical-lead
scope: docs/clasi/sprints/001-project-restructure-cli-foundation/
sprint: 001-project-restructure-cli-foundation
context_documents:
- docs/clasi/sprints/001-project-restructure-cli-foundation/sprint.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/architecture-update.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/usecases.md
result: 'success - 5 tickets created: 001-001 (pyproject.toml), 001-002 (CLI dispatcher),
  001-003 (playfield extraction), 001-004 (subcommand registration), 001-005 (pytest
  infrastructure)'
files_modified:
- docs/clasi/sprints/001-project-restructure-cli-foundation/tickets/001-update-pyproject-toml.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/tickets/002-create-unified-cli-dispatcher.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/tickets/003-extract-playfield-simulator.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/tickets/004-register-all-subcommands.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/tickets/005-add-pytest-infrastructure.md
---

# Dispatch: team-lead → technical-lead

You are the technical-lead. Create implementation tickets for Sprint 001: Project Restructure & CLI Foundation.

You may only create or modify files under docs/clasi/sprints/001-project-restructure-cli-foundation/. You may read files from any location.

Read these files first:
- docs/clasi/sprints/001-project-restructure-cli-foundation/architecture-update.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/usecases.md
- docs/clasi/sprints/001-project-restructure-cli-foundation/sprint.md
- src/aprilcam/cli/ (understand existing CLI modules)
- pyproject.toml

Then create tickets in docs/clasi/sprints/001-project-restructure-cli-foundation/tickets/ using the create_ticket MCP tool. Each ticket needs YAML frontmatter (id, title, status: open, use-cases, depends-on) and body (description, acceptance criteria, implementation notes).

The architecture has 5 change areas:
1. Unified CLI dispatcher in cli/__init__.py
2. pyproject.toml changes (single entry point, optional deps)
3. Playfield simulator extraction to contrib/
4. CLI subcommand registration (including capture for aprilcap)
5. Test infrastructure (conftest, smoke tests)

Break these into sequenced tickets ordered by dependency. Every use case (SUC-001 through SUC-004) must be covered. Foundation before features.

## Context Documents

- `docs/clasi/sprints/001-project-restructure-cli-foundation/sprint.md`
- `docs/clasi/sprints/001-project-restructure-cli-foundation/architecture-update.md`
- `docs/clasi/sprints/001-project-restructure-cli-foundation/usecases.md`
