---
id: "004"
title: "Remove deprecated CLI modules"
status: todo
use-cases:
  - SUC-004
depends-on:
  - "003"
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Remove deprecated CLI modules

## Description

Several CLI modules date from the pre-MCP architecture and are no
longer the intended usage path. They duplicate functionality now
available through the MCP server or the `aprilcam` subcommands. These
dead modules should be removed to reduce maintenance burden, confusion,
and package size.

Modules to evaluate for removal:
- `atscreencap_cli` -- screen capture CLI, superseded by MCP tools
- `apriltest_cli` -- test harness CLI, superseded by pytest tests
- `aprilcap_cli` -- capture CLI, superseded by MCP `capture_frame`
- `homocal_cli` -- homography calibration CLI, superseded by MCP
  `calibrate_playfield`
- `playfield_cli` -- playfield CLI, superseded by MCP playfield tools

Modules to keep:
- `taggen` -- tag generation, kept as `aprilcam taggen` subcommand
- `arucogen` -- ArUco generation, kept as `aprilcam arucogen` subcommand
- `cameras` -- camera listing, kept as `aprilcam cameras` subcommand
- `cli` (main) -- the `aprilcam` entry point and subcommand dispatcher

After removing modules:
1. Clean up any imports that reference the removed modules.
2. Verify no remaining code depends on the removed modules.
3. Ensure `aprilcam --help` still lists only the valid subcommands.

## Acceptance Criteria

- [ ] `atscreencap_cli` module is removed from the package
- [ ] `apriltest_cli` module is removed from the package
- [ ] `aprilcap_cli` module is removed from the package
- [ ] Any other dead CLI modules identified during audit are removed
- [ ] `taggen` module is retained and works via `aprilcam taggen`
- [ ] `arucogen` module is retained and works via `aprilcam arucogen`
- [ ] `cameras` module is retained and works via `aprilcam cameras`
- [ ] No remaining imports reference removed modules (clean import
      tree)
- [ ] `aprilcam --help` lists only valid, working subcommands
- [ ] All existing tests still pass after module removal
- [ ] No `ModuleNotFoundError` or `ImportError` when running any
      retained functionality

## Testing

- **Existing tests to run**: `uv run pytest` -- full suite
- **New tests to write**:
  - Test that importing the main package does not trigger imports of
    removed modules
  - Test that `aprilcam --help` output does not reference removed
    commands
- **Verification command**: `uv run pytest`
