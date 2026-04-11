---
sprint: '013'
status: approved
---

# Architecture Update -- Sprint 013: Tag Position Estimation

## What Changed

### Modified: `aprilcam/models.py` — `AprilTagFlow`

Three new private fields added to `AprilTagFlow`:
- `_vel_world: tuple[float, float]` — EMA-smoothed velocity in world units/s.
- `_speed_world: float` — magnitude of `_vel_world`.
- `_heading_rad: float` — `atan2(vy, vx)` of world velocity.

New method added:
```
AprilTagFlow.set_world_velocity(vel_world, speed_world, heading_rad)
```
Called by `Playfield.add_tag()` after the world-velocity transform.
Exposes three new read-only properties: `vel_world`, `speed_world`,
`heading_rad`.

### Modified: `aprilcam/playfield.py` — `Playfield.add_tag()`

Signature change:
```
add_tag(tag: AprilTag, homography: np.ndarray | None = None) -> None
```
New `homography` parameter defaults to None so all existing callers
are unaffected.

After the pixel-velocity EMA block, a new block runs when
`homography is not None` and pixel velocity is non-zero:

1. Linearize the homography at the tag's current pixel position.
2. Map `(cx, cy)` and `(cx + vx*dt_ref, cy + vy*dt_ref)` through `H @ p`.
3. Divide by `dt_ref` to recover world velocity components.
4. Compute `speed_world = hypot(wvx, wvy)` and `heading_rad = atan2(wvy, wvx)`.
5. Call `flow.set_world_velocity(vel_world, speed_world, heading_rad)`.

The reference time `dt_ref` is a unit constant (1.0 s) — only the
direction and magnitude of the pixel vector matter; dividing by the
same constant recovers the world rate.

### Modified: `aprilcam/aprilcam.py` — `process_frame()`

Two changes in the `TagRecord` construction block (lines ~609-649):

1. Call `self.playfield.add_tag(tag, homography=self.homography)` instead
   of `self.playfield.add_tag(tag)`.
2. When reading velocity from the flow to pass to `TagRecord.from_apriltag()`,
   also read `flow.vel_world`, `flow.speed_world`, `flow.heading_rad`
   instead of hard-coding `None`. Applied to both current and stale tag
   construction paths.

### Modified: `aprilcam/detection.py` — `TagRecord`

New method added to the frozen dataclass:
```
def estimate(self, t: float | None = None) -> TagRecord
```
Returns a new `TagRecord` with:
- `center_px` shifted by `vel_px * dt` (if `vel_px` is not None).
- `corners_px` — all four corners shifted by the same `(dx, dy)` delta.
- `world_xy` shifted by `vel_world * dt` (if both are not None).
- `timestamp` set to `t`.
- `age` set to `self.age + dt`.
- All other fields copied unchanged.

When `t` defaults to None, uses `time.monotonic()`.

---

## Component Diagram

```mermaid
graph LR
    AC[aprilcam.py<br/>AprilCam] -->|add_tag(tag, homography)| PF[playfield.py<br/>Playfield]
    PF -->|set_velocity()| ATF[models.py<br/>AprilTagFlow]
    PF -->|set_world_velocity()| ATF
    AC -->|reads vel_world/speed_world/heading_rad| ATF
    AC -->|from_apriltag(...vel_world...)| TR[detection.py<br/>TagRecord]
    TR -->|estimate(t)| TR2[TagRecord<br/>extrapolated]
```

---

## Data Flow — World Velocity

```
Camera frame
  -> AprilTag detected (pixel corners, center)
  -> Playfield.add_tag(tag, homography)
       pixel vel computed (EMA + deadband)
       if homography:
           linearize H at (cx, cy)
           vel_world = H-transformed pixel vel / dt_ref
           flow.set_world_velocity(vel_world, speed_world, heading_rad)
  -> AprilCam reads flow.vel_world
  -> TagRecord.from_apriltag(..., vel_world=flow.vel_world, ...)
  -> Caller: tag.estimate(t)
       new_center_px = center_px + vel_px * dt
       new_world_xy  = world_xy  + vel_world * dt
       return TagRecord(timestamp=t, ...)
```

---

## Why

Tag positions delivered to callers are always stale. For PID controllers
and trajectory planners operating at high frequency, the latency between
detection and use (10-100ms at 10-30 Hz) causes meaningful position error.
Linear extrapolation from smoothed velocity is a simple, zero-dependency
fix that covers the common case (short horizon, roughly constant velocity).
World-space velocity is the natural unit for robot control and is already
derivable from the existing homography — it just was never computed.

## Impact on Existing Components

- `Playfield.add_tag()` gains an optional `homography` parameter.
  Default is `None`. All existing callers that pass no homography are
  unaffected — no world velocity is computed.
- `AprilTagFlow` gains three new properties. They default to
  `(0.0, 0.0)`, `0.0`, and `0.0` respectively, so any code that reads
  them today will see zero rather than crashing.
- `TagRecord` is a frozen dataclass. The new `estimate()` method is
  additive — no existing attribute or method changes.
- `process_frame()` in `aprilcam.py` passes `self.homography` to
  `add_tag()`. When `self.homography` is None (default), behavior is
  identical to before.

## Migration Concerns

None. All changes are additive or guarded by None-checks. Existing
callers continue to work without modification.

---

## Design Rationale

### Homography linearization for velocity

**Decision**: Transform a unit pixel displacement vector through H to
recover world velocity, rather than transforming two successive world
positions.

**Context**: The homography maps pixel coordinates to world coordinates
non-linearly (perspective). Velocity requires a derivative.

**Alternatives considered**:
- Track world_xy across frames and difference them: simpler but
  introduces noise from the rounding in world_xy storage.
- Full Jacobian of H at (cx, cy): mathematically equivalent to
  the approach used here for a first-order approximation.

**Why this choice**: Linearizing at the current position gives a
first-order approximation that is exact for pure translations and
accurate to O(dt^2) for perspective distortion. It requires no
additional state.

**Consequences**: Velocity accuracy degrades slightly near the edges
of a strongly distorted field. Acceptable for Phase 1.

### `estimate()` as a method on `TagRecord`

**Decision**: Place `estimate()` on `TagRecord` rather than as a
free function or on `AprilTagFlow`.

**Context**: `TagRecord` is the public-facing immutable value type.
Callers work with `TagRecord` — they should not need to know about
`AprilTagFlow`.

**Why this choice**: Follows the principle that the object that owns
the data owns the operations on that data. `TagRecord` already contains
all inputs needed for estimation.

**Consequences**: `TagRecord` gains a dependency on `time.monotonic()`
(standard library only).

---

## Open Questions

None. The design is fully specified in the TODO and confirmed by the
existing code structure.
