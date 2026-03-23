---
id: "001-003"
title: "Extract playfield simulator to contrib/playfield/"
status: todo
use-cases: [SUC-004]
depends-on: [001-001]
github-issue: ""
todo: ""
---

# Extract playfield simulator to contrib/playfield/

## Description

Move the pygame-based playfield simulator out of the main package into
`contrib/playfield/`. This removes pygame as a runtime dependency for
normal aprilcam users while keeping the simulator available for
development and demos.

The core `src/aprilcam/playfield.py` module stays in the package -- it
contains the `Playfield` data model used by detection code and has no
pygame imports. Only the simulator UI (`playfield_cli.py`) moves out.

## Acceptance Criteria

- [ ] `src/aprilcam/cli/playfield_cli.py` is moved to `contrib/playfield/playfield_sim.py`
- [ ] `contrib/playfield/README.md` exists with usage instructions (how to install pygame, how to run the simulator)
- [ ] `import aprilcam` does not trigger a pygame import
- [ ] `src/aprilcam/playfield.py` remains in the package (it has no pygame imports)
- [ ] The simulator script in `contrib/playfield/` can run standalone with `python contrib/playfield/playfield_sim.py` (when pygame is installed)
- [ ] The `playfield` entry in the old `[project.scripts]` is removed (already handled by ticket 001-001, but verify it is not re-added)

## Implementation Notes

1. Create `contrib/playfield/` directory.

2. Move `src/aprilcam/cli/playfield_cli.py` to
   `contrib/playfield/playfield_sim.py`. Update any internal imports
   from relative (`from ..playfield import ...`) to absolute or adjust
   `sys.path` so the script can find the `aprilcam` package when run
   standalone. The simplest approach: add an `if __name__ == "__main__"`
   block and document that users should run it from the repo root with
   the aprilcam package installed.

3. Check `src/aprilcam/playfield.py` for any top-level `import pygame`
   statements. If found, make them conditional (`try: import pygame
   except ImportError: pygame = None`) or remove them. Currently this
   file does NOT import pygame, so this should be a no-op verification.

4. Delete `src/aprilcam/cli/playfield_cli.py` after the move.

5. Create `contrib/playfield/README.md` with:
   - What the simulator does
   - Prerequisites: `pip install pygame>=2.5`
   - How to run: `python contrib/playfield/playfield_sim.py`

## Testing

- **Existing tests to run**: None.
- **New tests to write**: A test in ticket 001-005 verifies that `import aprilcam` works without pygame.
- **Verification command**: `python -c "import aprilcam"` (should succeed without pygame installed).
