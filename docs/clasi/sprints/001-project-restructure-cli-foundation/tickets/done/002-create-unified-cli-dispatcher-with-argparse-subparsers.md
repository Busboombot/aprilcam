---
id: "001-002"
title: "Create unified CLI dispatcher with argparse subparsers"
status: todo
use-cases: [SUC-002, SUC-003]
depends-on: [001-001]
github-issue: ""
todo: ""
---

# Create unified CLI dispatcher with argparse subparsers

## Description

Create `src/aprilcam/cli/__init__.py` with a `main()` function that
serves as the single CLI entry point. This function uses `argparse` with
subparsers to dispatch to the existing CLI module `main(argv)` functions.

This is the core architectural change of the sprint. After this ticket,
`aprilcam <subcommand> [args]` replaces all standalone commands.

## Acceptance Criteria

- [ ] `src/aprilcam/cli/__init__.py` exists and exports a `main()` function
- [ ] `main()` creates an `ArgumentParser` with `add_subparsers()`
- [ ] `aprilcam --help` exits 0 and prints a usage summary with all subcommands listed
- [ ] When no subcommand is given, `aprilcam` prints help and exits with a non-zero code (or prints help and exits 0 -- choose one and be consistent)
- [ ] The `mcp` subcommand is registered as a stub that prints "MCP server not yet implemented" and exits 0
- [ ] Each subcommand handler passes remaining `argv` through to the target module's `main(argv)` unchanged
- [ ] The `main()` function returns an int exit code and calls `sys.exit()` appropriately

## Implementation Notes

The dispatcher pattern:

```python
import argparse
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(prog="aprilcam", description="AprilTag camera tools")
    subparsers = parser.add_subparsers(dest="command")

    # Register each subcommand with a one-line help string
    # Example: subparsers.add_parser("taggen", help="Generate AprilTag images")

    # mcp stub
    subparsers.add_parser("mcp", help="Launch MCP server (not yet implemented)")

    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to the appropriate module's main(remaining)
    ...
```

All existing CLI modules already accept `main(argv: Optional[List[str]] = None) -> int`.
The subcommand names and their target modules are:

| Subcommand | Module |
|---|---|
| mcp | stub (print message, return 0) |
| taggen | `aprilcam.cli.taggen_cli:main` |
| arucogen | `aprilcam.cli.arucogen_cli:main` |
| cameras | `aprilcam.cli.cameras_cli:main` |
| homocal | `aprilcam.cli.homocal_cli:main` |
| screencap | `aprilcam.cli.atscreencap_cli:main` |
| detect | `aprilcam.cli.aprilcam_cli:main` |
| capture | `aprilcam.cli.aprilcap_cli:main` |
| test | `aprilcam.cli.apriltest_cli:main` |

Use lazy imports (import inside the dispatch function) so that heavy
dependencies like `cv2` are not loaded just to print `--help`.

## Testing

- **Existing tests to run**: None.
- **New tests to write**: Covered by ticket 001-005 (smoke tests).
- **Verification command**: `python -m aprilcam.cli --help`
