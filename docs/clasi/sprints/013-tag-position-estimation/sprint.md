---
id: "013"
title: "Tag Position Estimation"
status: planning
branch: sprint/013-tag-position-estimation
use-cases: [SUC-001, SUC-002, SUC-003]
todo:
  - tag-position-estimation-tag-estimate-t-method.md
---

# Sprint 013: Tag Position Estimation

## Goals

Enable callers to estimate where a tag is right now (or at a future
moment) by extrapolating from the last known position using velocity.
Wire world-space velocity through the entire pipeline so that both
pixel and world positions are extrapolated correctly.

## Problem

`TagRecord` positions are stale by the time the caller receives them.
The detection pipeline runs at 10-30 Hz and processing adds latency.
PID controllers and trajectory planners need a current or future position,
not a past one. World-space velocity (`vel_world`) is never populated today
even though the homography is available to compute it. The data to
extrapolate exists in `AprilTagFlow`; it just is not exposed via an
estimation API.

## Solution

1. Add `_vel_world`, `_speed_world`, `_heading_rad` fields to
   `AprilTagFlow` and a new `set_world_velocity()` method.
2. Compute world-space velocity in `Playfield.add_tag()` using the
   homography matrix (passed as a parameter). Store on the flow.
3. Wire `vel_world`, `speed_world`, `heading_rad` through
   `TagRecord` construction in `aprilcam.py`.
4. Add `TagRecord.estimate(t)` — linear extrapolation of pixel and
   world position using velocity * dt, returning a new `TagRecord`.

## Success Criteria

- `TagRecord.estimate()` returns a correctly extrapolated position
  for both pixel and world coordinates.
- `vel_world` is non-None for tags on a calibrated playfield.
- Unit tests cover `estimate()` with and without world velocity.
- Unit tests cover world-velocity computation from known homography.
- All existing tests continue to pass.

## Scope

### In Scope

- `models.py`: add world-velocity fields and `set_world_velocity()` to
  `AprilTagFlow`.
- `playfield.py`: compute world-space velocity in `add_tag()` when
  homography is available; accept homography as a parameter.
- `aprilcam.py`: pass `self.homography` to `playfield.add_tag()`;
  wire `vel_world`, `speed_world`, `heading_rad` into `TagRecord`
  construction from the flow.
- `detection.py`: add `estimate(t)` method to `TagRecord`.

### Out of Scope

- Phase 2 estimators (Kalman filter, weighted linear regression).
- Orientation extrapolation (`orientation_yaw` unchanged by estimate).
- Tag class OOP refactoring (separate future work).
- Any UI or display changes.

## Test Strategy

Unit tests for:
- `TagRecord.estimate()` with known position and velocity.
- World-velocity computation using a synthetic homography matrix.
- Edge cases: `vel_px=None`, `vel_world=None`, past and future `t`.

Integration test: run `process_frame()` with a synthetic frame,
confirm `TagRecord.vel_world` is populated when homography is set.

## Architecture Notes

Homography linearization: the world-velocity transform uses the
current tag position as the linearization point. For a pure
translation in pixel space, this is exact. For perspective distortion,
it is a first-order approximation valid for small dt (< 200ms).

## GitHub Issues

(None)

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

| # | Title | Depends On | Group |
|---|-------|------------|-------|
| 001 | Add world-velocity fields to AprilTagFlow | — | 1 |
| 002 | Compute world velocity in Playfield and wire through aprilcam.py | 001 | 2 |
| 003 | Add TagRecord.estimate(t) method and tests | 001 | 2 |

**Groups**: Tickets in the same group can execute in parallel.
Groups execute sequentially (1 before 2, etc.).

Group 1 (foundation): ticket 001 must complete first — it defines the
`set_world_velocity()` interface that both 002 and 003 depend on.

Group 2 (parallel): tickets 002 and 003 are independent of each other
and can execute concurrently once 001 is done.
