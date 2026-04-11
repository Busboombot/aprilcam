---
status: done
sprint: '013'
tickets:
- 013-001
---

# Tag Position Estimation: `tag.estimate(t)` Method

## Context

When a tag's position is returned to the caller, it's already stale. The camera
captures at 10-30 Hz, detection takes some processing time, and by the time the
caller gets the TagRecord, the tag has moved. For PID controllers and trajectory
planning, you need to know where the tag is *right now* (or where it will be at
a specific future time when your calculation completes). 

The system already computes EMA-smoothed velocity per tag in `Playfield.add_tag()`
(pixel-space only; world-space velocity is always None today). The history is in
`AprilTagFlow` (deque of 5 AprilTag snapshots). The data to extrapolate exists —
it's just not exposed through an estimation interface.

## Design

### The `estimate()` method

Add an `estimate(t=None)` method to TagRecord (and later to the Tag class from
the OOP refactoring). It returns a **new TagRecord** with position values
extrapolated to time `t`.

```python
# Estimate where tag is right now
current = tag.estimate()

# Estimate where tag will be 50ms from now (when my PID calc finishes)
future = tag.estimate(time.monotonic() + 0.05)

# Use the same interface — cx, cy, wx, wy all adjusted
print(f"Estimated position: ({current.world_xy[0]:.1f}, {current.world_xy[1]:.1f})")
```

**Signature:**
```python
def estimate(self, t: float | None = None) -> TagRecord:
    """Return a new TagRecord with position extrapolated to time t.
    
    Args:
        t: Target time (monotonic). Defaults to time.monotonic() if None.
    
    Returns:
        New TagRecord with center_px, corners_px, and world_xy adjusted
        by velocity * dt. All other fields preserved. The timestamp on the
        returned record is set to t.
    """
```

**What changes in the returned TagRecord:**
- `center_px` — shifted by `vel_px * dt`
- `corners_px` — all 4 corners shifted by `vel_px * dt`
- `world_xy` — shifted by `vel_world * dt` (if calibrated)
- `timestamp` — set to `t` (the estimation target time)
- `age` — updated to reflect time since last actual detection
- Everything else (id, orientation, velocity, speed, heading) — preserved as-is

### Estimation Algorithm (Phase 1: Linear Extrapolation)

Start simple. Linear extrapolation from current velocity:

```
dt = t - self.timestamp
estimated_cx = self.center_px[0] + self.vel_px[0] * dt
estimated_cy = self.center_px[1] + self.vel_px[1] * dt
```

This works well for short horizons (< 200ms) where velocity is roughly constant.

### Prerequisite: World-Space Velocity

Today `vel_world` and `speed_world` are always `None` — they're fields on TagRecord
but never populated (lines 622-623 in aprilcam.py). We need to compute them.

**In `Playfield.add_tag()` or the future `VelocityEstimator`:**
- If homography is available, transform `vel_px` through the homography to get
  `vel_world` in cm/s
- Also compute `speed_world = hypot(vel_world)` and `heading_rad = atan2(vy, vx)`
- These get passed through to TagRecord via `from_apriltag()`

This is a small change in `aprilcam.py` lines 615-623 where TagRecords are built.

### Phase 2 (Future): Better Estimators

The linear extrapolation can be upgraded to:
- **Weighted linear regression** on the last N positions (already in AprilTagFlow deque)
- **Kalman filter** — maintains state (position + velocity), predicts forward,
  corrects on each measurement. Better for noisy/irregular updates.
- **Orientation extrapolation** — rotate `orientation_yaw` by `rotation_rate * dt`

The `estimate()` interface stays the same regardless of the algorithm behind it.
The algorithm choice can be a configuration option on the Playfield or a strategy
object on the VelocityEstimator.

## Implementation

### Files to modify

1. **`detection.py`** — Add `estimate(t)` method to `TagRecord`
2. **`aprilcam.py`** — Populate `vel_world`, `speed_world`, `heading_rad` when
   building TagRecords (lines ~615-623). Requires passing homography into the
   velocity→world conversion.
3. **`playfield.py`** — Compute world-space velocity in `add_tag()` when
   homography is available. Store on the flow alongside `vel_px`.

### Changes in detail

#### detection.py — `TagRecord.estimate()`

```python
def estimate(self, t: float | None = None) -> TagRecord:
    if t is None:
        t = time.monotonic()
    dt = t - self.timestamp
    
    # Extrapolate pixel position
    new_cx, new_cy = self.center_px
    if self.vel_px is not None:
        new_cx += self.vel_px[0] * dt
        new_cy += self.vel_px[1] * dt
    
    # Shift all 4 corners by same delta
    new_corners = self.corners_px
    if self.vel_px is not None:
        dx, dy = self.vel_px[0] * dt, self.vel_px[1] * dt
        new_corners = [[c[0] + dx, c[1] + dy] for c in self.corners_px]
    
    # Extrapolate world position
    new_world = self.world_xy
    if self.world_xy is not None and self.vel_world is not None:
        new_world = (
            self.world_xy[0] + self.vel_world[0] * dt,
            self.world_xy[1] + self.vel_world[1] * dt,
        )
    
    return TagRecord(
        id=self.id,
        center_px=(new_cx, new_cy),
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

#### playfield.py — World-space velocity in `add_tag()`

In the existing velocity computation block, after computing pixel velocity,
transform through homography to get world velocity:

```python
# After computing vel_px = (dx, dy) and speed_px...
if homography is not None:
    # Transform velocity vector through homography
    # Use the tag's current position as the linearization point
    p1 = np.array([cx, cy, 1.0])
    p2 = np.array([cx + dx * dt, cy + dy * dt, 1.0])  # small displacement
    w1 = homography @ p1; w1 /= w1[2]
    w2 = homography @ p2; w2 /= w2[2]
    vel_world = ((w2[0] - w1[0]) / dt, (w2[1] - w1[1]) / dt)
    speed_world = math.hypot(*vel_world)
    heading_rad = math.atan2(vel_world[1], vel_world[0])
```

This needs the homography matrix passed into `add_tag()` or stored on the
Playfield. Currently Playfield doesn't have it — it's on AprilCam.

#### aprilcam.py — Wire up vel_world in TagRecord construction

Lines ~615-623 currently set `vel_world=None`. Change to pass through the
world velocity from the flow (once Playfield computes it).

### New fields on AprilTagFlow

Add `_vel_world`, `_speed_world`, `_heading_rad` alongside existing `_vel_px`
and `_speed_px`. Updated via an expanded `set_velocity()`.

## Testing

1. **Unit test for `estimate()`**: Create a TagRecord with known position and
   velocity, call `estimate(t + 0.1)`, verify position shifted correctly.
2. **Unit test for world velocity**: Create tags with known homography, verify
   `vel_world` is correctly transformed from `vel_px`.
3. **Integration test**: Run `detect_tags()` generator, get a tag with velocity,
   call `estimate()`, verify result is a valid TagRecord with adjusted position.
4. **Accuracy test**: Use test video with known tag motion, verify estimated
   positions are within expected error bounds.

## Verification

1. `uv run pytest` — all existing tests pass
2. New tests for `estimate()` and world-space velocity pass
3. Manual test: run `detect_tags()`, print `tag.estimate().world_xy` vs
   `tag.world_xy` — estimated position should be slightly ahead in the
   direction of motion
