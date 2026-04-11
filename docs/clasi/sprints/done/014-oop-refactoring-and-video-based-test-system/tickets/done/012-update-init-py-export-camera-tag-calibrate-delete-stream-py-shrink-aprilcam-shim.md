---
id: '012'
title: 'Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim'
status: done
use-cases:
  - SUC-001
  - SUC-006
  - SUC-009
depends-on:
  - "010"
  - "011"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Update __init__.py: export Camera, Tag, calibrate; delete stream.py; shrink AprilCam shim

## Description

Update `src/aprilcam/__init__.py` to export the new public API (`Camera`, `Tag`,
`calibrate`, `Playfield`). Delete `stream.py` and replace its top-level exports
with re-exports pointing to `Playfield`-based implementations. Shrink `AprilCam`
in `core/aprilcam.py` to a minimal compatibility shim.

After this ticket, `import aprilcam; aprilcam.Camera`, `aprilcam.Tag`,
`aprilcam.Playfield`, `aprilcam.calibrate` all resolve. Legacy names
`detect_tags`, `detect_objects`, `AprilCam` remain available but may emit
`DeprecationWarning`.

## Acceptance Criteria

- [ ] `import aprilcam; aprilcam.Camera` resolves to `camera.Camera`.
- [ ] `import aprilcam; aprilcam.Tag` resolves to `core.tag.Tag`.
- [ ] `import aprilcam; aprilcam.calibrate` resolves to `calibration.calibrate`.
- [ ] `import aprilcam; aprilcam.Playfield` resolves to new `core.playfield.Playfield`.
- [ ] `from aprilcam import detect_tags` still works (compatibility re-export).
- [ ] `from aprilcam import detect_objects` still works (compatibility re-export).
- [ ] `from aprilcam import AprilCam` still works (shim, deprecated).
- [ ] `stream.py` is deleted from the source tree.
- [ ] `core/aprilcam.py` is reduced: all logic extracted; only shim wrappers remain
      that delegate to the new classes.
- [ ] `uv run pytest` passes (no import errors from removals).

## Implementation Plan

### Approach

1. Update `__init__.py`: add `Camera`, `Tag`, `calibrate` imports; update
   `Playfield` import; add `__all__` entries.
2. For `detect_tags`/`detect_objects` compatibility, write thin wrappers in
   `__init__.py` that open a `Playfield` and stream results — or point to
   `stream.py`'s replacement function.
3. Delete `stream.py`.
4. `AprilCam` in `core/aprilcam.py`: strip all extracted logic; keep constructor
   signature for backward compat; delegate methods to new classes.

### Files to Delete

- `src/aprilcam/stream.py`

### Files to Modify

- `src/aprilcam/__init__.py` — full update
- `src/aprilcam/core/aprilcam.py` — strip to shim

### Key Implementation Notes

- The compatibility `detect_tags()` in `__init__.py` needs to match the old
  generator signature: `detect_tags(camera, ...) -> Generator[list[TagRecord]]`.
  Implement as a thin wrapper over `Playfield.stream()`.
- `AprilCam` shim: keep `__init__` signature; in body, construct `TagDetector`,
  `OpticalFlowTracker`, `DetectionPipeline`; delegate `process_frame()`,
  `detect_apriltags()`, etc. to those objects.
- Verify all CLI and MCP server paths still import cleanly after deletion.

### Testing Plan

- Smoke: `import aprilcam` succeeds.
- Smoke: `aprilcam.Camera`, `aprilcam.Tag`, `aprilcam.Playfield`,
  `aprilcam.calibrate` all resolve.
- Smoke: `from aprilcam import detect_tags, detect_objects, AprilCam` succeeds.
- Smoke: `uv run aprilcam --help` works.
- Unit: `uv run pytest` passes with no import errors.

### Documentation Updates

- `AGENT_GUIDE.md`: update API section to show new exports.
- `__init__.py`: update module docstring to list public API.
