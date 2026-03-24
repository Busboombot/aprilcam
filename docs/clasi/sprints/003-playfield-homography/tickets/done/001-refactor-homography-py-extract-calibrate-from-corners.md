---
id: "001"
title: "Refactor homography.py — extract calibrate_from_corners()"
status: todo
use-cases: [SUC-005]
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Refactor homography.py — extract calibrate_from_corners()

## Description

Extract the calibration math from `homography.py`'s `main()` function
(lines 217-232) into a standalone library function
`calibrate_from_corners(pixel_corners, field_spec)` that takes four
pixel corner positions and a `FieldSpec`, builds world-coordinate
correspondences, and returns the 3x3 homography matrix.

This enables the MCP server to compute calibration without invoking CLI
code, argparse, or file I/O. The existing `main()` is then refactored
to call this new function internally, preserving all CLI behavior.

The `pixel_corners` parameter accepts a dict with keys `upper_left`,
`upper_right`, `lower_left`, `lower_right`, each a `(float, float)`
tuple. This matches the `CORNER_ID_MAP` naming convention already used
in `run_once()`.

## Acceptance Criteria

- [ ] New function `calibrate_from_corners(pixel_corners, field_spec)` exists in `homography.py`
- [ ] `pixel_corners` accepts a dict with keys `upper_left`, `upper_right`, `lower_left`, `lower_right`
- [ ] Returns a 3x3 `np.ndarray` homography matrix
- [ ] `main()` calls `calibrate_from_corners()` instead of inline math (lines 218-232)
- [ ] Existing `homocal` CLI command works unchanged
- [ ] No new imports of `argparse`, `config`, or `screencap` in the new function
- [ ] Unit test: `calibrate_from_corners()` with known pixel/world points produces correct homography (within tolerance)
- [ ] Unit test: `FieldSpec` inch-to-cm conversion is correct

## Testing

- **Existing tests to run**: `uv run pytest tests/` (all existing tests pass)
- **New tests to write**: `tests/test_homography.py` — unit tests for `calibrate_from_corners()` with synthetic data, `FieldSpec` unit conversions
- **Verification command**: `uv run pytest`
