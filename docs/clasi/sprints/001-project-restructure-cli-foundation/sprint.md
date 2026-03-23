---
id: "001"
title: "Project Restructure & CLI Foundation"
status: planning
branch: sprint/001-project-restructure-cli-foundation
use-cases:
  - SUC-001
  - SUC-002
  - SUC-003
  - SUC-004
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 001: Project Restructure & CLI Foundation

## Goals

Reorganize the AprilCam package from a flat collection of independent CLI
scripts into a single unified CLI entry point with subcommands, preparing
the package layout for MCP server integration in Sprint 002. Establish
pytest infrastructure for ongoing development.

## Problem

The current project has 9 separate `[project.scripts]` entry points in
pyproject.toml (`aprilcam`, `taggen`, `arucogen`, `homocal`, `cameras`,
`playfield`, `atscreencap`, `aprilcap`, `apriltest`). Each installs as
its own global command. This creates several problems:

1. **Namespace pollution** -- `pipx install aprilcam` installs 9 top-level
   commands, several with generic names (`cameras`, `playfield`) that
   risk collisions with other tools.
2. **No room for `mcp` subcommand** -- the MCP server needs to be
   launched as `aprilcam mcp`, which requires a subcommand architecture
   that does not exist yet.
3. **pygame is a core dependency** -- the playfield simulator pulls in
   pygame for all users, even though most users only need detection and
   tag generation. pygame is difficult to install in headless environments.
4. **No test infrastructure** -- there are no pytest fixtures, conftest,
   or smoke tests, making it risky to refactor without a safety net.

## Solution

1. Create a unified CLI dispatcher (`src/aprilcam/cli/__init__.py` or
   `src/aprilcam/cli/main.py`) that uses `argparse` subparsers to route
   to existing CLI modules. Each existing `main()` function becomes a
   subcommand handler.
2. Replace all 9 `[project.scripts]` entries with a single entry:
   `aprilcam = "aprilcam.cli:main"`.
3. Move the playfield simulator (`playfield_cli.py` and any
   pygame-specific code) out of the main package into `contrib/playfield/`
   with its own README explaining how to run it.
4. Remove `pygame` from `[project.dependencies]` and make it an optional
   extra (`[project.optional-dependencies]`).
5. Add `mcp` Python SDK to dependencies (needed for Sprint 002, but the
   dependency should be declared now).
6. Add `tests/conftest.py`, `tests/__init__.py`, and basic smoke tests
   that verify the CLI entry point and each subcommand's `--help` output.

## Success Criteria

- `pipx install .` installs a single `aprilcam` command.
- `aprilcam --help` lists all subcommands: `mcp`, `taggen`, `arucogen`,
  `cameras`, `homocal`, `screencap`, `detect`, `test`.
- `aprilcam taggen --help` shows the taggen-specific options.
- `aprilcam mcp` prints a placeholder message (e.g., "MCP server not yet
  implemented") and exits cleanly.
- `pygame` is not required for `pip install aprilcam` (only for the
  optional `contrib/playfield` simulator).
- `pytest` runs from the repo root and all smoke tests pass.
- The playfield simulator code is in `contrib/playfield/` and is not
  importable from `aprilcam.*`.

## Scope

### In Scope

- Unified CLI entry point with argparse subparsers.
- Subcommands: `mcp` (placeholder), `taggen`, `arucogen`, `cameras`,
  `homocal`, `screencap` (was `atscreencap`), `detect` (was `aprilcam`
  main), `test` (was `apriltest`).
- `aprilcap` consolidated into `detect` or kept as a subcommand if
  functionality is distinct enough.
- Moving `playfield_cli.py` and `playfield.py` simulator code to
  `contrib/playfield/`.
- Updating pyproject.toml: single entry point, optional pygame dep,
  add `mcp` SDK dependency.
- pytest infrastructure: conftest.py, smoke tests for CLI help output.
- Removing the old standalone entry points.

### Out of Scope

- MCP server implementation (Sprint 002).
- Changes to core detection, homography, or tracking logic.
- New CLI features or flags.
- Integration tests that require a camera.
- Rewriting any CLI module internals -- only the dispatch layer changes.
- The `playfield.py` core module stays in the package (it contains
  non-simulator logic used by detection); only the pygame simulator
  UI code moves out.

## Test Strategy

- **Smoke tests**: For each subcommand, verify that `aprilcam <sub> --help`
  exits 0 and contains expected strings (description, key flags).
- **Import tests**: Verify that `import aprilcam` works without pygame
  installed. Verify that the CLI module is importable.
- **Entry point test**: Verify that the `aprilcam` console script is
  registered and callable.
- All tests run with `pytest` from the repo root. No camera or display
  hardware required for Sprint 001 tests.

## Architecture Notes

- Use `argparse` with subparsers (not `click`) to avoid adding a new
  dependency. The existing CLI modules already use argparse internally.
- Each subcommand handler receives `argv` so it can parse its own
  arguments independently. The top-level parser handles only the
  subcommand dispatch.
- The `mcp` subcommand will be a thin stub in this sprint that prints
  a not-yet-implemented message. Sprint 002 will replace it with the
  real MCP server entry point.

## GitHub Issues

(None linked yet.)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
