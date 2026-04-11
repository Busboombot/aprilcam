---
id: '003'
title: 'VelocityEstimator: per-tag EMA velocity with deadband'
status: todo
use-cases:
  - SUC-005
depends-on: []
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# VelocityEstimator: per-tag EMA velocity with deadband

## Description

Create `src/aprilcam/core/motion.py` with `VelocityEstimator`.

`VelocityEstimator` is per-tag stateful object tracking position history with
EMA smoothing and deadband suppression. It absorbs the velocity EMA block from
`Playfield.add_tag()` in `playfield.py`. Designed so callers can swap EMA for
a Kalman filter without changing the interface.

## Acceptance Criteria

- [ ] `core/motion.py` exists with `VelocityEstimator` class.
- [ ] `VelocityEstimator(alpha: float = 0.3, deadband: float = 50.0)` constructor.
- [ ] `update(position: tuple[float, float], timestamp: float) -> tuple[tuple[float, float], float]`
      returns `(velocity_vec, speed)`. Returns `((0.0, 0.0), 0.0)` on first call.
- [ ] EMA: `v_ema = alpha * v_raw + (1 - alpha) * v_prev`.
- [ ] Deadband: if raw `speed < deadband`, reports `((0.0, 0.0), 0.0)`.
- [ ] `predict_position(t: float) -> tuple[float, float]` extrapolates linearly from
      last known position using last computed velocity.
- [ ] `reset()` clears all internal state.
- [ ] `Playfield.add_tag()` updated to delegate EMA velocity to `VelocityEstimator`.
- [ ] `core/__init__.py` exports `VelocityEstimator`.

## Implementation Plan

### Approach

Lift `_vel_ema`, `_last_seen` EMA logic from `Playfield.add_tag()` into this
class. The world-velocity homography transform stays in `Playfield`/`pipeline.py`
since it needs the homography matrix; `VelocityEstimator` handles pixel-space only.

### Files to Create

- `src/aprilcam/core/motion.py`

### Files to Modify

- `src/aprilcam/core/playfield.py` — delegate velocity EMA to `VelocityEstimator`
- `src/aprilcam/core/__init__.py` — export `VelocityEstimator`

### Key Implementation Notes

- One `VelocityEstimator` instance per tracked tag ID.
- `speed = hypot(vx, vy)`. Deadband applied to raw speed before EMA.
- Store `_last_pos`, `_last_time`, `_vel_ema` as instance attributes.

### Testing Plan

- Smoke: `from aprilcam.core import VelocityEstimator` succeeds.
- Unit: first `update()` returns zero velocity.
- Unit: constant velocity across two updates returns correct vector.
- Unit: sub-deadband speed returns zero.
- Unit: `predict_position(t)` extrapolates correctly.
- Unit: `reset()` clears state so next `update()` gives zero velocity.

### Documentation Updates

- Docstrings on `VelocityEstimator`, `update()`, `predict_position()`, `reset()`.
