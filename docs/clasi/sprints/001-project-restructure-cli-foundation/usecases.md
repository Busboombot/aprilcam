---
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 001 Use Cases

## SUC-001: Install aprilcam via pipx
Parent: (none -- foundational)

- **Actor**: Robotics developer
- **Preconditions**: Python 3.9+ and pipx are installed. No prior
  aprilcam installation exists.
- **Main Flow**:
  1. Developer runs `pipx install .` from the repo root (or
     `pipx install aprilcam` from PyPI once published).
  2. pipx creates an isolated venv and installs the package.
  3. A single `aprilcam` command is placed on the PATH.
  4. No other commands (taggen, cameras, etc.) are installed globally.
- **Postconditions**: `which aprilcam` resolves. `which taggen` does not
  resolve (no namespace pollution). pygame is not installed in the venv.
- **Acceptance Criteria**:
  - [ ] `pipx install .` completes without error.
  - [ ] Only the `aprilcam` command is installed on the PATH.
  - [ ] pygame is not in the installed dependencies.

## SUC-002: Discover available subcommands
Parent: (none -- foundational)

- **Actor**: Robotics developer (new to the tool)
- **Preconditions**: aprilcam is installed.
- **Main Flow**:
  1. Developer runs `aprilcam --help`.
  2. The CLI prints a usage summary listing all subcommands: `mcp`,
     `taggen`, `arucogen`, `cameras`, `homocal`, `screencap`, `detect`,
     `test`.
  3. Each subcommand has a one-line description.
  4. Developer runs `aprilcam taggen --help` to see taggen-specific
     options.
- **Postconditions**: Developer understands the available commands and
  how to get help for each one.
- **Acceptance Criteria**:
  - [ ] `aprilcam --help` exits 0 and lists all 8 subcommands.
  - [ ] `aprilcam <subcommand> --help` exits 0 for every subcommand.
  - [ ] Each subcommand help includes a description and its specific flags.

## SUC-003: Run a subcommand that was previously a standalone tool
Parent: (none -- migration)

- **Actor**: Existing aprilcam user
- **Preconditions**: aprilcam is installed. User previously ran `taggen`
  as a standalone command.
- **Main Flow**:
  1. User runs `aprilcam taggen --ids 0-10 --outdir ./tags`.
  2. The CLI dispatches to the taggen handler.
  3. Tag images are generated in `./tags/` exactly as the old standalone
     `taggen` command would have produced.
- **Postconditions**: Output is identical to the old standalone command.
  All flags and behavior are preserved.
- **Acceptance Criteria**:
  - [ ] `aprilcam taggen` accepts the same flags as the old `taggen` command.
  - [ ] Output files are identical to what the standalone command produced.
  - [ ] Same applies to `arucogen`, `cameras`, `homocal`, `screencap`,
        `detect`, and `test` subcommands.

## SUC-004: Run the playfield simulator independently
Parent: (none -- separation of concerns)

- **Actor**: Developer who wants to test with the pygame playfield simulator
- **Preconditions**: aprilcam repo is cloned. pygame is installed in the
  developer's environment (not installed by default with aprilcam).
- **Main Flow**:
  1. Developer navigates to `contrib/playfield/`.
  2. Developer reads the README for instructions.
  3. Developer installs pygame (`pip install pygame`) if not already present.
  4. Developer runs the simulator script directly (e.g.,
     `python -m contrib.playfield` or `python contrib/playfield/playfield_sim.py`).
  5. The pygame window opens with the playfield simulation.
- **Postconditions**: The simulator runs independently of the main aprilcam
  package. The main package does not depend on pygame.
- **Acceptance Criteria**:
  - [ ] Playfield simulator code is in `contrib/playfield/`, not in
        `src/aprilcam/`.
  - [ ] `import aprilcam` does not import pygame.
  - [ ] The simulator can be run from `contrib/playfield/` with pygame
        installed.
  - [ ] `contrib/playfield/` has a README with run instructions.
