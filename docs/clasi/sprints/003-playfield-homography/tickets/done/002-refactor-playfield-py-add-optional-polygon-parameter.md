---
id: "002"
title: "Refactor playfield.py — add optional polygon parameter"
status: todo
use-cases: [SUC-005]
depends-on: []
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Refactor playfield.py — add optional polygon parameter

## Description

Add an optional `polygon` parameter to the `Playfield` constructor so
that the MCP tool handlers (and tests) can inject a pre-detected polygon
directly, bypassing the frame-based detection flow.

When `polygon` is provided (a 4x2 float32 ndarray in UL, UR, LR, LL
order), it is stored as `_poly` immediately and `update()` becomes a
no-op. When `polygon` is `None` (default), existing behavior is
preserved -- the polygon is detected from frames via `update()`.

This is a non-breaking change: all existing callers that do not pass
`polygon` continue to work identically.

## Acceptance Criteria

- [ ] `Playfield(polygon=np.array(...))` initializes with `_poly` set to the provided array
- [ ] `Playfield()` (no polygon) behaves identically to current code
- [ ] `update()` is a no-op when polygon was injected via constructor
- [ ] `get_polygon()` returns the injected polygon
- [ ] `deskew()` works correctly with an injected polygon
- [ ] Existing CLI commands (`playfield`, `aprilcam`) work unchanged
- [ ] Unit test: constructor with polygon sets `_poly`
- [ ] Unit test: `_order_poly()` produces UL, UR, LR, LL order regardless of input ordering

## Testing

- **Existing tests to run**: `uv run pytest tests/` (all existing tests pass)
- **New tests to write**: `tests/test_playfield.py` — unit tests for constructor with polygon, `_order_poly()` ordering, `deskew()` output dimensions with synthetic data
- **Verification command**: `uv run pytest`
