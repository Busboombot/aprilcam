---
sprint: '013'
status: approved
---

# Use Cases — Sprint 013: Tag Position Estimation

## SUC-001: Estimate Current Tag Position

- **Actor**: AI agent or application code
- **Preconditions**: A `TagRecord` has been obtained from `get_tags()`
  or the ring buffer. The tag had non-zero velocity on at least one axis.
- **Main Flow**:
  1. Caller receives a `TagRecord` from the detection pipeline.
  2. Caller calls `tag.estimate()` with no argument.
  3. The method computes `dt = time.monotonic() - tag.timestamp`.
  4. Returns a new `TagRecord` with `center_px` and `corners_px` shifted
     by `vel_px * dt`, and `world_xy` shifted by `vel_world * dt` (if
     calibrated). `timestamp` is set to the current monotonic time.
- **Postconditions**: Returned record has updated position fields.
  Non-positional fields (id, orientation, velocity, heading) are
  identical to the source record.
- **Acceptance Criteria**:
  - [ ] `estimate()` returns a `TagRecord` instance.
  - [ ] `center_px` is shifted by `vel_px * dt` within floating-point tolerance.
  - [ ] `corners_px` are all shifted by the same `(dx, dy)` delta.
  - [ ] `world_xy` is shifted by `vel_world * dt` when both are non-None.
  - [ ] When `vel_px` is None, position fields are copied unchanged.

---

## SUC-002: Estimate Future Tag Position

- **Actor**: PID controller or trajectory planner
- **Preconditions**: A `TagRecord` with velocity has been obtained.
  The caller has a known future target time.
- **Main Flow**:
  1. Caller computes `t_future = time.monotonic() + lookahead_seconds`.
  2. Caller calls `tag.estimate(t_future)`.
  3. Method extrapolates position by `vel * (t_future - tag.timestamp)`.
  4. Returns a `TagRecord` with `timestamp = t_future`.
- **Postconditions**: Returned record reflects predicted position at
  `t_future`. `age` is updated to `original_age + dt`.
- **Acceptance Criteria**:
  - [ ] `estimate(t)` with `t > tag.timestamp` produces forward-extrapolated position.
  - [ ] `estimate(t)` with `t < tag.timestamp` produces backward-extrapolated position.
  - [ ] Returned `timestamp` equals the `t` argument exactly.
  - [ ] Returned `age` equals `original_age + (t - original_timestamp)`.

---

## SUC-003: World-Space Velocity on Calibrated Playfield

- **Actor**: Detection pipeline
- **Preconditions**: `AprilCam` has a non-None `self.homography`.
  A tag has been tracked for at least two frames.
- **Main Flow**:
  1. `aprilcam.py` calls `playfield.add_tag(tag, homography=self.homography)`.
  2. `Playfield.add_tag()` computes pixel velocity, then transforms
     through the homography to obtain `vel_world` in world units/second.
  3. `vel_world`, `speed_world`, `heading_rad` are stored on the flow
     via `flow.set_world_velocity()`.
  4. `aprilcam.py` reads world velocity from the flow when constructing
     `TagRecord` objects (both current and stale tags).
- **Postconditions**: `TagRecord.vel_world` is a `(vx, vy)` tuple in
  world units/second. `speed_world` and `heading_rad` are also non-None.
  When homography is absent, these fields remain None.
- **Acceptance Criteria**:
  - [ ] `vel_world` is non-None for a moving tag when homography is set.
  - [ ] `speed_world` and `heading_rad` are non-None alongside `vel_world`.
  - [ ] When homography is None, all three fields remain None.
  - [ ] World velocity direction is consistent with pixel velocity direction.
