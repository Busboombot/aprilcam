---
sprint: "001"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 001: Project Restructure & CLI Foundation

## What Changed

### 1. Unified CLI Entry Point

**Added**: `src/aprilcam/cli/main.py` (or `src/aprilcam/cli/__init__.py`)
containing the top-level argparse dispatcher.

The dispatcher creates an `ArgumentParser` with subparsers. Each
subcommand is registered with a handler function that delegates to the
existing CLI module's `main(argv)` function:

```
aprilcam (top-level parser)
  ├── mcp        → placeholder stub (exits with message)
  ├── taggen     → aprilcam.cli.taggen_cli:main
  ├── arucogen   → aprilcam.cli.arucogen_cli:main
  ├── cameras    → aprilcam.cli.cameras_cli:main
  ├── homocal    → aprilcam.cli.homocal_cli:main
  ├── screencap  → aprilcam.cli.atscreencap_cli:main
  ├── detect     → aprilcam.cli.aprilcam_cli:main
  └── test       → aprilcam.cli.apriltest_cli:main
```

Each handler passes remaining argv to the submodule, so existing
argument parsing is untouched.

### 2. pyproject.toml Changes

**Modified**: `[project.scripts]` reduced from 9 entries to 1:

```toml
[project.scripts]
aprilcam = "aprilcam.cli:main"
```

**Modified**: `[project.dependencies]` -- removed `pygame>=2.5`.

**Added**: `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
playfield = ["pygame>=2.5"]
mcp = ["mcp>=1.0"]
```

The `mcp` dependency is declared as optional for now. Sprint 002 will
move it to core dependencies when the MCP server is implemented.

### 3. Playfield Simulator Extraction

**Moved**: `src/aprilcam/cli/playfield_cli.py` to
`contrib/playfield/playfield_sim.py`.

**Moved**: pygame-specific rendering code from `src/aprilcam/playfield.py`
to `contrib/playfield/`. The core playfield data model and non-pygame
logic (if any) remains in `src/aprilcam/playfield.py`.

**Added**: `contrib/playfield/README.md` with installation and usage
instructions.

**Note**: `src/aprilcam/playfield.py` is retained if it contains
non-simulator logic (e.g., playfield coordinate models used by detection).
Only the pygame UI and simulator CLI are extracted.

### 4. CLI Module Renaming

**Removed**: `src/aprilcam/cli/aprilcap_cli.py` entry point. Its
functionality is merged into the `detect` subcommand, or it becomes
an internal module if its logic is distinct from `aprilcam_cli.py`.
(To be determined during implementation based on code analysis.)

**Renamed** (logical, not file rename): The subcommand names are
simplified:
- `atscreencap` becomes `screencap`
- `aprilcam` (the old standalone) becomes `detect`
- `apriltest` becomes `test`

The underlying CLI module files keep their original names to minimize
diff noise. Only the subparser registration uses the new names.

### 5. Test Infrastructure

**Added**:
- `tests/__init__.py`
- `tests/conftest.py` -- shared fixtures (e.g., a fixture that provides
  the CLI runner)
- `tests/test_cli_smoke.py` -- parametrized smoke tests that run
  `aprilcam <subcommand> --help` and verify exit code 0 and expected
  output strings

## Why

The project overview specifies a single CLI entry point (`aprilcam`)
with subcommands, including an `mcp` subcommand to launch the MCP server.
The current 9-entry-point layout cannot support this. This sprint
establishes the CLI architecture that all future sprints build on.

Removing pygame from core dependencies is necessary because the MCP
server will run in headless environments (containers, remote machines)
where pygame cannot be installed. The playfield simulator is a
development/demo tool, not part of the MCP server.

Adding pytest infrastructure now creates the safety net needed for
refactoring in this sprint and all subsequent sprints.

## Impact on Existing Components

- **All existing CLI modules**: No internal changes. Each module's
  `main(argv)` function signature is preserved. They are simply called
  from the new dispatcher instead of from their own entry points.
- **playfield.py core module**: Retained in the package. Only the
  pygame simulator UI is extracted. If the core module imports pygame
  at the module level, those imports need to be made conditional or
  lazy.
- **Users of standalone commands**: Breaking change. Users who previously
  ran `taggen` must now run `aprilcam taggen`. Since the package is
  pre-1.0 and not published to PyPI, this is acceptable.

## Migration Concerns

- **Existing installations**: Users who installed with `pipx install .`
  will have stale entry points for `taggen`, `cameras`, etc. They need
  to run `pipx uninstall aprilcam && pipx install .` to clean up.
  This is documented but not automated.
- **playfield.py imports**: If `src/aprilcam/playfield.py` has a
  top-level `import pygame`, it will break for users without pygame.
  This must be resolved by either making the import conditional or
  splitting the module. The implementation ticket will handle this.
- **No backward compatibility shims**: We are not providing wrapper
  scripts for the old command names. Clean break.
