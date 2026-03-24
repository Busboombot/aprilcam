---
id: "006"
title: "pipx install verification"
status: todo
use-cases:
  - SUC-006
depends-on:
  - "003"
  - "004"
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# pipx install verification

## Description

Verify that the package installs cleanly via `pipx install .` and that
the installed `aprilcam` command works correctly. This is the final
gate before the package can be considered ready for distribution. The
verification must confirm that installation succeeds, only the
expected entry point is created, the MCP server starts, and CLI
subcommands function.

This ticket depends on ticket 003 (pyproject.toml cleanup) and ticket
004 (deprecated module removal) because the entry points and module
structure must be finalized before installation can be verified.

Steps:
1. Run `pipx install .` in an isolated environment (or `pipx install
   --force .` if already installed).
2. Verify that `aprilcam` is the only script installed (no `taggen`,
   `arucogen`, `homocal`, etc. as standalone commands).
3. Run `aprilcam --help` and verify it exits 0 and lists subcommands.
4. Run `aprilcam mcp` and send an MCP `initialize` JSON-RPC request
   via stdin; verify a valid response is returned.
5. Run `aprilcam taggen --help` and verify it exits 0.
6. Run `aprilcam arucogen --help` and verify it exits 0.
7. Run `aprilcam cameras` and verify it exits 0 (may list no cameras
   in CI, but should not crash).

Consider writing a shell script or pytest test that automates these
checks so they can be repeated easily.

## Acceptance Criteria

- [ ] `pipx install .` completes without errors in a clean environment
- [ ] Only `aprilcam` appears as an installed script (no legacy
      standalone commands like `taggen`, `arucogen`, `homocal`,
      `cameras`, `playfield`, `atscreencap`, `aprilcap`, `apriltest`)
- [ ] `aprilcam --help` exits 0 and displays available subcommands
- [ ] `aprilcam mcp` starts the MCP server and responds to an
      `initialize` JSON-RPC request over stdio
- [ ] `aprilcam taggen --help` exits 0 with usage information
- [ ] `aprilcam arucogen --help` exits 0 with usage information
- [ ] `aprilcam cameras` exits 0 without crashing (even with no
      cameras attached)
- [ ] A verification script or test exists that automates the above
      checks
- [ ] All verification steps are documented so they can be run
      manually if needed

## Testing

- **Existing tests to run**: `uv run pytest` -- full suite
- **New tests to write**:
  - Packaging verification test or script that runs `pipx install .`
    and checks each entry point
  - Test that `aprilcam mcp` responds to MCP `initialize` request
    (can be an integration test via subprocess)
- **Verification command**: `uv run pytest`
