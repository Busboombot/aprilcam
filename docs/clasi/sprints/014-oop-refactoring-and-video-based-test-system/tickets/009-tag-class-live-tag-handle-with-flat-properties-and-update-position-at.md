---
id: "009"
title: 'Tag class: live tag handle with flat properties and update/position_at'
status: todo
use-cases:
  - SUC-006
  - SUC-007
depends-on:
  - "008"
github-issue: ''
todo:
  - docs/clasi/todo/oop-refactoring.md
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Tag class: live tag handle with flat properties and update/position_at

## Description

Create `src/aprilcam/core/tag.py` with `Tag`.

`Tag` is the user-facing live tag object obtained from `Playfield`. It holds an
atomically-replaced frozen `TagRecord` snapshot. Callers use flat properties
(`cx`, `cy`, `wx`, `wy`, etc.) instead of navigating nested `TagRecord` fields.
`update()` pulls the latest snapshot from the pipeline; `position_at(t)` extrapolates.

`Tag` has no detection logic — it is purely a read-only view of the ring buffer.

## Acceptance Criteria

- [ ] `core/tag.py` exists with `Tag` class.
- [ ] `Tag(tag_id: int, pipeline: DetectionPipeline)` constructor.
- [ ] `tag.id: int` — tag ID.
- [ ] `tag.cx: float` — pixel center x (from latest snapshot).
- [ ] `tag.cy: float` — pixel center y.
- [ ] `tag.wx: float | None` — world x (cm), or None if uncalibrated.
- [ ] `tag.wy: float | None` — world y (cm), or None if uncalibrated.
- [ ] `tag.orientation: float` — yaw in radians.
- [ ] `tag.velocity: tuple[float, float] | None` — (vx, vy) in world units/s.
- [ ] `tag.speed: float | None` — scalar speed.
- [ ] `tag.heading: float | None` — `atan2(vy, vx)` in radians.
- [ ] `tag.rotation_rate: float | None` — yaw delta per second.
- [ ] `tag.timestamp: float` — capture time of latest snapshot.
- [ ] `tag.age: float` — seconds since last detected.
- [ ] `tag.is_visible: bool` — True if `age == 0.0`.
- [ ] `tag.update() -> Tag` — atomically replaces internal snapshot with latest
      from ring buffer; returns self for chaining.
- [ ] `tag.position_at(t: float) -> tuple[float, float]` — extrapolates pixel
      position; delegates to `TagRecord.estimate(t)`.
- [ ] `tag.to_dict() -> dict` — returns flat dict matching `TagRecord.to_dict()` keys.
- [ ] `core/__init__.py` exports `Tag`.

## Implementation Plan

### Approach

1. Create `core/tag.py`. `Tag` stores a reference to `DetectionPipeline`
   (or to `RingBuffer` directly) and a cached `TagRecord`.
2. `update()`: search ring buffer for latest `TagRecord` with `id == self.id`;
   atomically assign to `self._snapshot` using `threading.Lock`.
3. All flat properties read from `self._snapshot`; return sensible defaults
   if `_snapshot is None` (tag not yet seen).
4. `position_at(t)` calls `self._snapshot.estimate(t).center_px`.

### Files to Create

- `src/aprilcam/core/tag.py`

### Files to Modify

- `src/aprilcam/core/__init__.py` — export `Tag`

### Key Implementation Notes

- `Tag` is created by `Playfield`, not by callers directly.
- `_snapshot: TagRecord | None` — None if tag never seen.
- Thread safety: `_snapshot` replaced atomically via assignment (GIL protects
  single assignment in CPython; use a lock for correctness in edge cases).
- `is_visible`: True when `age == 0.0` (detected in the latest frame).

### Testing Plan

- Smoke: `from aprilcam.core import Tag` succeeds.
- Unit: `Tag(id=42, pipeline=mock_pipeline)` constructs.
- Unit: properties return None/zero before first `update()`.
- Unit: after `update()` with a seeded ring buffer, properties match `TagRecord` values.
- Unit: `position_at(t)` extrapolates correctly from a known velocity snapshot.

### Documentation Updates

- Docstrings on `Tag`, all properties, `update()`, `position_at()`, `to_dict()`.
