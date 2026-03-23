---
id: "001-004"
title: "Register all subcommands in CLI dispatcher"
status: todo
use-cases: [SUC-002, SUC-003]
depends-on: [001-002, 001-003]
github-issue: ""
todo: ""
---

# Register all subcommands in CLI dispatcher

## Description

Wire up all 9 subcommands in the CLI dispatcher created in ticket 001-002.
After ticket 001-002 creates the dispatcher skeleton with the `mcp` stub,
this ticket adds the remaining 8 subcommands with proper help strings and
ensures each one correctly delegates to its target module.

This ticket depends on 001-003 because the playfield subcommand must NOT
be registered (it has been extracted to contrib/). The dispatcher must
only register commands for modules that exist in the package.

## Acceptance Criteria

- [ ] All 9 subcommands are registered: `mcp`, `taggen`, `arucogen`, `cameras`, `homocal`, `screencap`, `detect`, `capture`, `test`
- [ ] `aprilcam --help` lists all 9 subcommands with one-line descriptions
- [ ] `aprilcam taggen --help` shows taggen-specific options (delegated to taggen_cli)
- [ ] `aprilcam detect --help` shows detect-specific options (delegated to aprilcam_cli)
- [ ] `aprilcam screencap --help` shows screencap-specific options (delegated to atscreencap_cli)
- [ ] `aprilcam capture --help` shows capture-specific options (delegated to aprilcap_cli)
- [ ] `aprilcam test --help` shows test-specific options (delegated to apriltest_cli)
- [ ] `aprilcam mcp` prints "MCP server not yet implemented" and exits 0
- [ ] No `playfield` subcommand is registered
- [ ] Each subcommand passes remaining argv to the target module unchanged

## Implementation Notes

Subcommand help strings (suggested):

| Subcommand | Help | Target |
|---|---|---|
| mcp | Launch MCP server | stub |
| taggen | Generate AprilTag images | taggen_cli:main |
| arucogen | Generate ArUco marker images | arucogen_cli:main |
| cameras | List available cameras | cameras_cli:main |
| homocal | Calibrate homography | homocal_cli:main |
| screencap | Capture screen regions with tag detection | atscreencap_cli:main |
| detect | Run live AprilTag detection | aprilcam_cli:main |
| capture | Single-frame capture with detection | aprilcap_cli:main |
| test | Run tag detection tests | apriltest_cli:main |

Use lazy imports to avoid loading cv2/numpy when the user just wants `--help`.

NOTE: Tickets 001-002 and 001-004 may be executed together as a single
implementation pass if the implementer prefers. They are separated for
clarity of scope, but the dispatcher is not useful without subcommand
registration.

## Testing

- **Existing tests to run**: None.
- **New tests to write**: Covered by ticket 001-005 (smoke tests for all subcommands).
- **Verification command**: `aprilcam --help` and `aprilcam <sub> --help` for each subcommand.
