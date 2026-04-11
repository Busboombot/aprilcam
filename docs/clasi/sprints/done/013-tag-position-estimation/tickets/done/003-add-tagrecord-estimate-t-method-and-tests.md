---
id: '003'
title: Add TagRecord.estimate(t) method and tests
status: done
use-cases:
- SUC-001
- SUC-002
depends-on:
- '001'
github-issue: ''
todo: ''
---

# Add TagRecord.estimate(t) method and tests

## Description

`TagRecord` is a frozen dataclass holding a tag's last observed state.
By the time a caller receives it, the tag has moved. This ticket adds
`TagRecord.estimate(t)` — a method that linearly extrapolates the tag's
position to time `t` (defaulting to now) using the stored velocity.

The method requires no changes to call sites — it is purely additive.
It uses only fields already present on `TagRecord`: `vel_px`, `vel_world`,
`world_xy`, `center_px`, `corners_px`, `timestamp`, `age`.

## Acceptance Criteria

- [ ] `TagRecord.estimate()` (no arg) returns a `TagRecord` with
  `center_px` shifted by `vel_px * (now - self.timestamp)`.
- [ ] `TagRecord.estimate(t)` with explicit `t` returns a `TagRecord`
  with `timestamp == t`.
- [ ] `corners_px` — all four corners are shifted by the same `(dx, dy)`.
- [ ] `world_xy` is shifted by `vel_world * dt` when both are non-None;
  otherwise `world_xy` is copied unchanged.
- [ ] When `vel_px` is None, `center_px` and `corners_px` are copied
  unchanged.
- [ ] Returned record's `age` equals `self.age + dt`.
- [ ] All non-positional fields (id, orientation_yaw, vel_px, vel_world,
  speed_px, speed_world, heading_rad, frame_index, in_playfield) are
  identical to the source record.
- [ ] Unit tests for all above acceptance criteria pass.
- [ ] `uv run pytest` passes.

## Implementation Plan

### Approach

Add `estimate()` as a regular method on `TagRecord`. Because `TagRecord`
is frozen, the method constructs and returns a new instance — it does not
mutate self.

```python
import time

def estimate(self, t: float | None = None) -> "TagRecord":
    """Return a new TagRecord with position extrapolated to time t.

    Args:
        t: Target monotonic time. Defaults to time.monotonic().

    Returns:
        New TagRecord with center_px, corners_px, and world_xy adjusted
        by velocity * dt. timestamp is set to t. age is updated.
        All other fields are preserved.
    """
    if t is None:
        t = time.monotonic()
    dt = t - self.timestamp

    new_center = self.center_px
    new_corners = self.corners_px
    if self.vel_px is not None:
        dx = self.vel_px[0] * dt
        dy = self.vel_px[1] * dt
        new_center = (self.center_px[0] + dx, self.center_px[1] + dy)
        new_corners = [[c[0] + dx, c[1] + dy] for c in self.corners_px]

    new_world = self.world_xy
    if self.world_xy is not None and self.vel_world is not None:
        new_world = (
            self.world_xy[0] + self.vel_world[0] * dt,
            self.world_xy[1] + self.vel_world[1] * dt,
        )

    return TagRecord(
        id=self.id,
        center_px=new_center,
        corners_px=new_corners,
        orientation_yaw=self.orientation_yaw,
        world_xy=new_world,
        in_playfield=self.in_playfield,
        vel_px=self.vel_px,
        speed_px=self.speed_px,
        vel_world=self.vel_world,
        speed_world=self.speed_world,
        heading_rad=self.heading_rad,
        timestamp=t,
        frame_index=self.frame_index,
        age=self.age + dt,
    )
```

Add `import time` at the top of `detection.py` (check if already present).

### Files to Modify

- `src/aprilcam/detection.py` — add `estimate()` method to `TagRecord`.
  Add `import time` if not already imported.

### Files to Create

- `tests/test_tag_estimate.py` — unit tests for `TagRecord.estimate()`.

### Testing Plan

Tests in `tests/test_tag_estimate.py`:

```
test_estimate_default_t_shifts_center
  - Create TagRecord with known center, vel_px=(10.0, 0.0), timestamp=now-0.1
  - Call estimate() with no arg
  - Assert new center_px[0] ≈ original + 10*0.1 = original + 1.0 (within 0.01)

test_estimate_explicit_t_shifts_center
  - vel_px=(5.0, -3.0), timestamp=T, call estimate(T + 0.2)
  - Assert center_px[0] += 1.0, center_px[1] -= 0.6

test_estimate_shifts_all_corners
  - Verify all 4 corners shift by same (dx, dy) as center

test_estimate_world_xy_shifted
  - TagRecord with world_xy=(100.0, 200.0), vel_world=(50.0, 0.0)
  - estimate(T + 0.1) -> world_xy[0] ≈ 105.0

test_estimate_no_vel_px_unchanged
  - vel_px=None, any world_xy
  - estimate() returns same center_px and corners_px

test_estimate_no_vel_world_world_xy_unchanged
  - vel_world=None, non-None world_xy
  - estimate() returns same world_xy

test_estimate_timestamp_set
  - Call estimate(T + 0.5), confirm result.timestamp == T + 0.5

test_estimate_age_updated
  - age=0.0, timestamp=T, call estimate(T + 0.3)
  - result.age ≈ 0.3

test_estimate_preserves_other_fields
  - Confirm id, orientation_yaw, vel_px, speed_px, frame_index
    are identical between source and result
```

### Documentation Updates

Add a docstring to `estimate()` as shown in the implementation above.
