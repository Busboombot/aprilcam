---
id: '004'
title: 'TagTableTUI: extract Rich TUI dashboard from AprilCam'
status: done
use-cases:
  - SUC-008
depends-on: []
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# TagTableTUI: extract Rich TUI dashboard from AprilCam

## Description

Create `src/aprilcam/ui/tui.py` with `TagTableTUI`.

`TagTableTUI` is a Rich-based terminal dashboard that displays a live table of
tag IDs, positions, speeds, and headings. It is extracted from the TUI rendering
methods inside `AprilCam` (`_build_tui_layout`, `_print_tui`, `_stop_tui`,
`_ema_smooth`). After extraction, `TagTableTUI` must have no dependency on
`AprilCam`; it reads tag data only through the public `Playfield` / `Tag` API.

This ticket does not wire `TagTableTUI` to the new `Playfield` — that happens
in ticket 010. This ticket extracts the code and puts it in its own module with
a clean interface.

## Acceptance Criteria

- [ ] `ui/tui.py` exists with `TagTableTUI` class.
- [ ] `TagTableTUI` has no import of `AprilCam`.
- [ ] `TagTableTUI.__init__(title: str = "AprilCam")` initializes the Rich Live display.
- [ ] `TagTableTUI.update(tags: list[dict])` renders a row per tag with columns:
      `id`, `cx`, `cy`, `wx`, `wy`, `speed`, `heading`, `orientation`.
- [ ] `TagTableTUI.start()` and `TagTableTUI.stop()` control the Live context.
- [ ] Context manager support: `with TagTableTUI() as tui:`.
- [ ] EMA smoothing for display-only values stays inside `TagTableTUI` (not
      exported to business logic).
- [ ] `AprilCam` TUI methods replaced with calls to `TagTableTUI`.
- [ ] `ui/__init__.py` exports `TagTableTUI`.

## Implementation Plan

### Approach

1. Create `ui/tui.py`.
2. Lift `_build_tui_layout`, `_print_tui`, `_stop_tui`, `_ema_smooth` from
   `aprilcam.py` into `TagTableTUI` methods.
3. `update(tags)` accepts a list of dicts (not `TagRecord` objects) so the
   TUI has no dependency on `core` data types.
4. Update `AprilCam` to instantiate `TagTableTUI` and delegate to it.

### Files to Create

- `src/aprilcam/ui/tui.py`

### Files to Modify

- `src/aprilcam/core/aprilcam.py` — delegate TUI rendering to `TagTableTUI`
- `src/aprilcam/ui/__init__.py` — export `TagTableTUI`

### Key Implementation Notes

- `tags` parameter to `update()` is `list[dict]` with keys matching `TagRecord.to_dict()`.
- EMA display smoothing is private state inside `TagTableTUI`.
- If Rich is not installed, `TagTableTUI` should raise `ImportError` with a clear message.

### Testing Plan

- Smoke: `from aprilcam.ui import TagTableTUI` succeeds.
- Smoke: `TagTableTUI()` constructs without error.
- Unit: `update([])` renders an empty table without raising.
- Unit: `update([{"id": 1, "cx": 100.0, ...}])` renders a one-row table.
- (No hardware needed; TUI output can be captured to a string buffer.)

### Documentation Updates

- Docstrings on `TagTableTUI`, `update()`, `start()`, `stop()`.
