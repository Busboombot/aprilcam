---
id: '001'
title: Add world-velocity fields to AprilTagFlow
status: done
use-cases:
- SUC-003
depends-on: []
github-issue: ''
todo: tag-position-estimation-tag-estimate-t-method.md
---

# Add world-velocity fields to AprilTagFlow

## Description

`AprilTagFlow` currently tracks pixel-space velocity only (`_vel_px`,
`_speed_px`). This ticket adds the parallel world-space velocity fields
and a setter method so that `Playfield` (ticket 002) can store computed
world velocity on the flow, and `aprilcam.py` (also ticket 002) can
read it when constructing `TagRecord` objects.

This is pure model-layer work — no computation logic, just the new
fields and their interface.

## Acceptance Criteria

- [ ] `AprilTagFlow` has three new private fields initialized to
  `(0.0, 0.0)`, `0.0`, and `0.0`:
  `_vel_world`, `_speed_world`, `_heading_rad`.
- [ ] New method `set_world_velocity(vel_world, speed_world, heading_rad)`
  stores all three fields. Mirrors the signature of `set_velocity()`.
- [ ] Three new read-only properties expose the fields:
  `vel_world -> tuple[float, float]`,
  `speed_world -> float`,
  `heading_rad -> float`.
- [ ] All existing `AprilTagFlow` tests pass unchanged.
- [ ] `uv run pytest` passes.

## Implementation Plan

### Approach

Add fields and method to `AprilTagFlow` in `src/aprilcam/models.py`,
following the exact same pattern as the existing pixel-velocity
implementation.

### Files to Modify

- `src/aprilcam/models.py`
  - In `__init__`: add `self._vel_world: Tuple[float, float] = (0.0, 0.0)`,
    `self._speed_world: float = 0.0`, `self._heading_rad: float = 0.0`.
  - Add method `set_world_velocity(self, vel_world, speed_world, heading_rad)`.
  - Add properties `vel_world`, `speed_world`, `heading_rad`.

### Testing Plan

No new tests required for this ticket — it is pure scaffolding. Ticket 002
will exercise the setter via `Playfield.add_tag()` tests. Confirm the
existing test suite passes: `uv run pytest`.

### Documentation Updates

None required. The new API is documented via docstrings on the setter and
properties.
