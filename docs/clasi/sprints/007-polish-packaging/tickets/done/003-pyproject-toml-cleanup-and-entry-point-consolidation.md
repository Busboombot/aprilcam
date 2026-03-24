---
id: '003'
title: pyproject.toml cleanup and entry point consolidation
status: done
use-cases:
- SUC-003
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# pyproject.toml cleanup and entry point consolidation

## Description

The current `pyproject.toml` has placeholder metadata and multiple
legacy standalone entry points (`taggen`, `arucogen`, `homocal`,
`cameras`, `playfield`, `atscreencap`, `aprilcap`, `apriltest`) that
were created before the MCP server architecture. These should be
consolidated so that `aprilcam` is the only installed script, with
subcommands for everything else.

Changes required:
1. Remove all `[project.scripts]` entries except `aprilcam`.
2. Verify that `aprilcam taggen`, `aprilcam arucogen`, and
   `aprilcam cameras` subcommands still work after removal of
   standalone entry points.
3. Update `[project]` metadata:
   - `name`: confirm it is `aprilcam`
   - `description`: meaningful one-liner
   - `authors`: real author name and email
   - `keywords`: aprilcam, apriltag, aruco, mcp, robotics, opencv
   - `classifiers`: appropriate PyPI trove classifiers (Development
     Status, Intended Audience, License, Programming Language, Topic)
   - `urls`: Homepage, Repository, Bug Tracker pointing to the GitHub
     repo
4. Verify `[project.optional-dependencies]` and `[build-system]` are
   correct and complete.

## Acceptance Criteria

- [ ] `[project.scripts]` contains only the `aprilcam` entry point
- [ ] All legacy standalone entry points (`taggen`, `arucogen`,
      `homocal`, `cameras`, `playfield`, `atscreencap`, `aprilcap`,
      `apriltest`) are removed
- [ ] `aprilcam taggen --help` still works as a subcommand
- [ ] `aprilcam arucogen --help` still works as a subcommand
- [ ] `aprilcam cameras` still works as a subcommand
- [ ] `aprilcam mcp` still starts the MCP server
- [ ] `[project]` author field contains a real name and email (not
      placeholder)
- [ ] `[project]` description is a meaningful one-liner about the
      project
- [ ] `[project]` keywords include relevant terms (apriltag, aruco,
      mcp, robotics)
- [ ] `[project]` classifiers include at least: Development Status,
      License, Programming Language :: Python :: 3, Topic
- [ ] `[project.urls]` includes Homepage, Repository, and Bug Tracker
- [ ] `uv run pip install -e .` succeeds with the updated
      pyproject.toml
- [ ] No import errors when running `aprilcam --help`

## Testing

- **Existing tests to run**: `uv run pytest` -- full suite
- **New tests to write**:
  - Test that `aprilcam --help` exits 0 and lists subcommands
  - Test that removed entry points are not installed (e.g., `which
    taggen` returns nothing after install)
- **Verification command**: `uv run pytest`
