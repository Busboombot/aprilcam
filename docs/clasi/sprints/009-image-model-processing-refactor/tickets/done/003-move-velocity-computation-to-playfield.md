---
id: '003'
title: Move velocity computation to Playfield
status: done
use-cases:
- SUC-003
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Move velocity computation to Playfield

## Description

Relocate the EMA (exponential moving average) + dead-band velocity computation
from `AprilCam.process_frame()` to `Playfield`. Currently, velocity is computed
in AprilCam and also partially self-computed in AprilTagFlow. This creates
duplicated logic and unclear ownership.

After this change:
- **Playfield** is the single owner of velocity computation
- **AprilTagFlow** retains its history deque and velocity fields but no longer
  self-computes velocity; it receives velocity externally from Playfield
- **AprilCam** no longer maintains velocity state (`_vel_ema`, `_last_seen`)

### Changes required

1. **`Playfield`** (`src/aprilcam/playfield.py`):
   - Add EMA alpha and dead-band threshold as configurable `__init__()` parameters
   - Add internal velocity state tracking (`_vel_ema`, `_last_seen` dicts)
   - Add `add_tag(tag, timestamp)` method that computes velocity using EMA +
     dead-band and stores it on the associated `AprilTagFlow`
   - Adopt the exact algorithm currently in `AprilCam.process_frame()`

2. **`AprilTagFlow`** (`src/aprilcam/models.py`):
   - Remove any self-computation of velocity
   - Keep `vel_px`, `speed_px` fields; these are now set externally by Playfield
   - Keep the history deque for position tracking

3. **`AprilCam`** (`src/aprilcam/aprilcam.py`):
   - Remove `_vel_ema` and `_last_seen` state variables
   - Remove velocity computation from `process_frame()`
   - `process_frame()` returns detection results only; velocity is Playfield's job

## Acceptance Criteria

- [x] Playfield has configurable EMA alpha and dead-band threshold in `__init__()`
- [x] `Playfield.add_tag()` computes velocity using EMA + dead-band algorithm
- [x] Velocity values match what AprilCam previously produced (behavioral parity)
- [x] `AprilTagFlow.vel_px` and `speed_px` are set by Playfield, not self-computed
- [x] `AprilCam` no longer has `_vel_ema` or `_last_seen` state
- [x] `AprilCam.process_frame()` returns detections without velocity
- [x] `get_tags()` MCP tool returns velocity sourced from Playfield
- [x] No velocity-related code remains in AprilCam

## Implementation Notes

### Key files
- `src/aprilcam/playfield.py` -- Playfield class: add velocity logic
- `src/aprilcam/models.py` -- AprilTagFlow: remove self-computation
- `src/aprilcam/aprilcam.py` -- AprilCam: remove velocity state and computation
- `src/aprilcam/mcp_server.py` -- verify get_tags still works correctly

### Design decisions
- Behavioral parity: the EMA + dead-band algorithm must be identical to what
  AprilCam currently uses, just relocated
- Playfield stores per-tag velocity state internally (dict keyed by tag_id)
- Dead-band threshold prevents jitter on stationary tags
- EMA alpha controls smoothing (higher = more responsive, lower = smoother)

### Migration approach
1. Copy the velocity algorithm from AprilCam to Playfield
2. Wire Playfield.add_tag() into the detection/processing flow
3. Remove velocity code from AprilCam
4. Remove self-computation from AprilTagFlow
5. Verify existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest` (full suite, ensure no regressions)
- **New tests to write**:
  - `test_playfield_velocity_ema` -- verify EMA computation matches expected values
  - `test_playfield_velocity_deadband` -- verify dead-band suppresses jitter
  - `test_playfield_add_tag_sets_flow_velocity` -- verify flow gets velocity
  - `test_aprilcam_no_velocity_state` -- verify _vel_ema/_last_seen removed
  - `test_velocity_behavioral_parity` -- same inputs produce same outputs as before
- **Verification command**: `uv run pytest`
