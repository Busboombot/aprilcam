---
id: '011'
title: 'CLI live loop: move AprilCam.run() to cli/live_cli.py'
status: todo
use-cases:
  - SUC-008
depends-on:
  - "004"
  - "010"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# CLI live loop: move AprilCam.run() to cli/live_cli.py

## Description

Move the interactive detection loop from `AprilCam.run()` into `cli/live_cli.py`.
`live_cli.py` should use `Playfield` and `TagTableTUI` directly, removing all
dependency on `AprilCam` from the CLI layer.

After this ticket, `aprilcam live` (or equivalent CLI command) works identically
from the user's perspective, but the implementation no longer touches `AprilCam`.

## Acceptance Criteria

- [ ] `cli/live_cli.py` contains the live detection loop using `Playfield` +
      `TagTableTUI` + `PlayfieldDisplay`.
- [ ] `AprilCam.run()` is removed or reduced to a one-line shim calling the
      CLI-layer function.
- [ ] `uv run aprilcam live` (or the equivalent subcommand) works end-to-end.
- [ ] TUI renders tag data during live camera operation.
- [ ] Keyboard interrupt (`Ctrl+C`) stops the pipeline cleanly.
- [ ] No import of `AprilCam` in `cli/live_cli.py`.

## Implementation Plan

### Approach

1. Read the existing `AprilCam.run()` implementation in `aprilcam.py`.
2. Rewrite the equivalent logic in `cli/live_cli.py` using:
   - `Camera.find(pattern)` to open the camera.
   - `Playfield(camera, ...)` to set up the pipeline.
   - `TagTableTUI` for display.
   - `PlayfieldDisplay` for the OpenCV window (if applicable).
3. Wire to the CLI argument parser in `cli/__init__.py` or `init_cli.py`.
4. In `aprilcam.py`, `run()` becomes: raise `DeprecationWarning` or simply
   call the new CLI function.

### Files to Modify

- `src/aprilcam/cli/live_cli.py` — full rewrite using Playfield API
- `src/aprilcam/core/aprilcam.py` — remove/shim `run()`
- `src/aprilcam/cli/__init__.py` (or `init_cli.py`) — ensure routing unchanged

### Key Implementation Notes

- Argument parsing for camera pattern, calibration file, family, etc. stays in
  CLI layer.
- `PlayfieldDisplay` (OpenCV window) is wired to the `on_frame` callback.
- Signal handling: `try/finally` with `field.stop()` guarantees cleanup.

### Testing Plan

- Smoke: `uv run aprilcam --help` includes the live subcommand.
- Smoke: `uv run aprilcam live --help` shows expected options.
- (End-to-end live test requires hardware; deferred to manual verification.)

### Documentation Updates

- Update `AGENT_GUIDE.md` if it references the CLI command.
- Docstring on the new `live_cli` entry point function.
