---
id: '013'
title: 'Smoke tests: import and construction sanity checks'
status: done
use-cases:
  - SUC-010
depends-on:
  - "012"
github-issue: ''
todo:
  - docs/clasi/todo/video-based-test-system.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Smoke tests: import and construction sanity checks

## Description

Create `tests/smoke/` with fast, hardware-free sanity checks. Smoke tests
verify that the package imports cleanly, all new public classes are accessible,
and basic construction does not raise. They must run in under 5 seconds total.

These tests are the first line of defense against import errors, missing
modules, and broken `__init__.py` wiring after the refactor.

## Acceptance Criteria

- [ ] `tests/smoke/` directory exists with `__init__.py` and at least one test file.
- [ ] `tests/smoke/test_imports.py` verifies:
      - `import aprilcam` succeeds.
      - `aprilcam.Camera` is accessible.
      - `aprilcam.Tag` is accessible.
      - `aprilcam.Playfield` is accessible.
      - `aprilcam.calibrate` is callable.
      - `aprilcam.VideoCamera` is accessible (via `aprilcam.camera` or top-level).
      - `from aprilcam import TagRecord, AprilCam, detect_tags` succeeds (legacy).
- [ ] `tests/smoke/test_construction.py` verifies:
      - `TagDetector()` constructs without error.
      - `DetectorConfig()` constructs with defaults.
      - `VelocityEstimator()` constructs without error.
      - `OpticalFlowTracker()` constructs without error.
      - `TagTableTUI()` constructs without error.
      - `VideoCamera("tests/movies/bright-gsc.mov")` constructs and is open.
- [ ] All smoke tests pass with `uv run pytest tests/smoke/`.
- [ ] Total smoke test runtime under 5 seconds.
- [ ] No hardware required to run smoke tests.

## Implementation Plan

### Approach

Write minimal `pytest` tests — no fixtures, no mocks. Each test is a single
`assert` or `import` statement. The point is to catch wiring errors quickly.

### Files to Create

- `tests/smoke/__init__.py`
- `tests/smoke/test_imports.py`
- `tests/smoke/test_construction.py`

### Files to Modify

- `tests/conftest.py` (create if needed) — add path to `tests/movies/` fixture.

### Key Implementation Notes

- Use `pytest.importorskip` or `importlib` for optional dependencies.
- `VideoCamera` construction test needs the path to `tests/movies/bright-gsc.mov`;
  use `pathlib.Path(__file__).parent.parent / "movies" / "bright-gsc.mov"`.
- Do not call `read()` in smoke tests — construction only.
- If Rich is not installed, `TagTableTUI` construction should be `pytest.skip`-ed
  rather than fail.

### Testing Plan

- These tests are themselves the test plan.
- Verification: `uv run pytest tests/smoke/ -v` all green.

### Documentation Updates

- Add comment at top of each smoke test file explaining its purpose.
