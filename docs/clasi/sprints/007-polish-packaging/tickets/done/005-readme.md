---
id: '005'
title: README
status: done
use-cases:
- SUC-005
depends-on:
- '003'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# README

## Description

The project currently has no README. Create a `README.md` at the
project root that covers installation, MCP server usage, CLI
subcommand reference, and a quick-start example. The README is the
first thing users and AI agents see when encountering the project, so
it must clearly communicate what AprilCam is, how to install it, and
how to use it.

Sections to include:

1. **Project title and one-liner** -- what AprilCam is (MCP server for
   camera-based AprilTag/ArUco detection and playfield management).
2. **Features** -- bullet list of key capabilities (camera management,
   playfield/homography, tag detection loop, image processing tools,
   multi-camera compositing).
3. **Installation** -- `pipx install aprilcam` (or `pip install
   aprilcam`), prerequisites (Python >= 3.9, OpenCV).
4. **Quick Start** -- minimal example showing how to configure an MCP
   client to use `aprilcam mcp`, and a brief sequence of tool calls.
5. **MCP Server Usage** -- how to start the server (`aprilcam mcp`),
   transport (stdio), how to connect from an MCP client.
6. **CLI Reference** -- `aprilcam --help`, `aprilcam taggen`,
   `aprilcam arucogen`, `aprilcam cameras` with brief descriptions.
7. **Development** -- how to set up for development (`uv sync`,
   `uv run pytest`).
8. **License** -- reference to license file.

## Acceptance Criteria

- [ ] `README.md` exists at the project root
- [ ] README includes an installation section with `pipx install`
      instructions
- [ ] README includes a quick-start section showing MCP client
      configuration and basic tool call sequence
- [ ] README includes MCP server usage section explaining stdio
      transport and how to start the server
- [ ] README includes CLI reference section covering `aprilcam mcp`,
      `aprilcam taggen`, `aprilcam arucogen`, `aprilcam cameras`
- [ ] README includes a development section with setup and test
      commands
- [ ] README includes a features overview listing key capabilities
- [ ] README mentions Python >= 3.9 and OpenCV as prerequisites
- [ ] README is well-formatted Markdown that renders correctly on
      GitHub

## Testing

- **Existing tests to run**: `uv run pytest` -- full suite (no
  regressions from other changes)
- **New tests to write**: None (documentation-only ticket)
- **Verification command**: `uv run pytest`
