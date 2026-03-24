---
id: 008
title: "Fix velocity noise \u2014 port EMA smoothing from run() to process_frame()"
status: done
use-cases:
- SUC-003
depends-on: []
github-issue: ''
todo: ''
---

# Fix velocity noise — port EMA smoothing from run() to process_frame()

## Description

`process_frame()` reports velocity using raw `AprilTagFlow.vel_px`
which is a single frame-to-frame delta. On static tags, sub-pixel
detector jitter produces fake velocities of 10-100+ px/s.

The original `run()` had EMA smoothing via `speed_alpha` and a
`vel_ema` dict that accumulated over many frames. This smoothing was
left behind as a "UI-only" local when `process_frame()` was extracted.

Fix: move the EMA velocity state (`_vel_ema`, `_last_seen`) into
instance attributes on AprilCam, compute smoothed velocity inside
`process_frame()`, and use the smoothed values in the TagRecords
returned to callers (including the MCP detection loop).

Also add a dead-band threshold: velocities below ~2 px/s should be
reported as 0.0 to suppress detector jitter on truly static tags.

## Acceptance Criteria

- [ ] `process_frame()` uses EMA-smoothed velocity, not raw frame-to-frame delta
- [ ] `_vel_ema` and `_last_seen` are instance attributes, reset by `reset_state()`
- [ ] Static tags report near-zero velocity (<2 px/s → 0.0)
- [ ] Moving tags still report accurate non-zero velocity
- [ ] `run()` uses the same smoothed velocities from process_frame (no duplication)
- [ ] All existing tests pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test that process_frame returns near-zero velocity for static image fed repeatedly
- **Verification command**: `uv run pytest`
