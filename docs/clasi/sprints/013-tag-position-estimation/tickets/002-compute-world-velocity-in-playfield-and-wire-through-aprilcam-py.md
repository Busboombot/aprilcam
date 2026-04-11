---
id: '002'
title: Compute world velocity in Playfield and wire through aprilcam.py
status: done
use-cases:
- SUC-003
depends-on:
- '001'
github-issue: ''
todo: ''
---

# Compute world velocity in Playfield and wire through aprilcam.py

## Description

Today `vel_world`, `speed_world`, and `heading_rad` on `TagRecord` are
always `None` (hard-coded in `aprilcam.py` lines ~618-628). This ticket
has two parts:

1. **`playfield.py`**: Extend `Playfield.add_tag()` to accept a `homography`
   parameter. When provided, transform the computed pixel velocity through
   the homography to get world-space velocity, then call
   `flow.set_world_velocity()` (added in ticket 001).

2. **`aprilcam.py`**: Pass `self.homography` to `playfield.add_tag()` and
   read `flow.vel_world`, `flow.speed_world`, `flow.heading_rad` when
   constructing `TagRecord` objects — for both current and stale tags.

## Acceptance Criteria

- [ ] `Playfield.add_tag()` signature is
  `add_tag(self, tag: AprilTag, homography: np.ndarray | None = None) -> None`.
- [ ] When `homography` is None, behavior is identical to before —
  `set_world_velocity()` is not called, world fields remain `(0.0, 0.0) / 0.0 / 0.0`.
- [ ] When `homography` is provided and pixel velocity is non-zero,
  `flow.vel_world`, `flow.speed_world`, and `flow.heading_rad` are set
  to non-trivially-zero values consistent with the pixel velocity direction.
- [ ] `aprilcam.py` passes `homography=self.homography` in the `add_tag()` call
  inside `process_frame()`.
- [ ] Both current-tag and stale-tag `TagRecord` construction paths in
  `process_frame()` pass `vel_world`, `speed_world`, `heading_rad` from
  the flow (not hardcoded `None`).
- [ ] A unit test verifies that a known pixel velocity + known homography
  produces the expected `vel_world` on the flow after `add_tag()`.
- [ ] `uv run pytest` passes.

## Implementation Plan

### Approach

**Playfield world-velocity computation:**

In `Playfield.add_tag()`, after the existing EMA block, insert:

```python
if homography is not None and speed_px_val > 0.0:
    vx, vy = vel_px_val
    cx, cy = tag.center_px
    # Linearize H at (cx, cy): map center and center + unit velocity
    dt_ref = 1.0
    p1 = np.array([cx, cy, 1.0], dtype=float)
    p2 = np.array([cx + vx * dt_ref, cy + vy * dt_ref, 1.0], dtype=float)
    w1 = homography @ p1; w1 = w1 / w1[2]
    w2 = homography @ p2; w2 = w2 / w2[2]
    wvx = (w2[0] - w1[0]) / dt_ref
    wvy = (w2[1] - w1[1]) / dt_ref
    speed_world = math.hypot(wvx, wvy)
    heading_rad = math.atan2(wvy, wvx)
    flow.set_world_velocity((wvx, wvy), speed_world, heading_rad)
```

**aprilcam.py wiring:**

In `process_frame()`, change the single `add_tag` call:
```python
# Before:
self.playfield.add_tag(tag)
# After:
self.playfield.add_tag(tag, homography=self.homography)
```

In both TagRecord construction blocks, replace the three `None` literals:
```python
# Before:
vel_world=None,
speed_world=None,
heading_rad=None,
# After:
vel_world=flow.vel_world if flow else None,
speed_world=flow.speed_world if flow else None,
heading_rad=flow.heading_rad if flow else None,
```

Note: `flow.vel_world` defaults to `(0.0, 0.0)` when not set. This means
when homography is absent, TagRecord gets `(0.0, 0.0)` rather than `None`.
Adjust the default in models.py or keep a sentinel check — prefer keeping
`None` semantics. Solution: initialize `_vel_world` as `None` in the Flow
and check for None in the property. Update the acceptance criteria in ticket
001 accordingly during implementation if needed (the implementor may choose
the cleanest approach).

### Files to Modify

- `src/aprilcam/playfield.py` — extend `add_tag()` signature and add
  world-velocity computation block after the EMA block.
- `src/aprilcam/aprilcam.py` — pass homography in `add_tag()` call;
  wire world velocity into both TagRecord construction blocks.

### Testing Plan

Write a unit test in `tests/` (new file `tests/test_world_velocity.py`
or add to an appropriate existing test file):

1. Create a simple identity-like or scale-only homography matrix.
2. Create an `AprilTag` with a known `center_px` and `last_ts`.
3. Call `playfield.add_tag(tag)` twice with different positions to
   establish velocity.
4. Call `playfield.add_tag(tag3, homography=H)` with a third position.
5. Assert `flow.vel_world` is not `(0.0, 0.0)` and its direction is
   consistent with the pixel velocity direction.
6. For a pure scaling homography (scale=2), verify `speed_world ≈ 2 * speed_px`.

### Documentation Updates

None required. The behavior is documented via the method docstring update.
