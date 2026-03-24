---
id: '001'
title: TagRecord and FrameRecord dataclasses
status: done
use-cases:
- SUC-003
- SUC-004
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# TagRecord and FrameRecord dataclasses

## Description

Add two new dataclasses to `src/aprilcam/detection.py` (new file) that
serve as the structured output format for the detection loop and MCP
query tools.

**TagRecord** captures the full state of a single detected tag at one
point in time. It is the serializable counterpart to the existing
`AprilTag` model in `models.py`, but designed for ring buffer storage
and JSON output rather than in-loop mutation.

Fields:
- `id: int` -- tag ID
- `center_px: tuple[float, float]` -- pixel center
- `corners_px: list[list[float]]` -- 4x2 corner coordinates (plain lists, not ndarray)
- `orientation_yaw: float` -- yaw angle in radians
- `world_xy: tuple[float, float] | None` -- world coordinates (cm), None if uncalibrated
- `in_playfield: bool` -- whether tag center is inside the playfield polygon
- `vel_px: tuple[float, float] | None` -- pixel velocity (px/s), None if no prior frame
- `speed_px: float | None` -- pixel speed (px/s)
- `vel_world: tuple[float, float] | None` -- world velocity (units/s)
- `speed_world: float | None` -- world speed (units/s)
- `heading_rad: float | None` -- heading angle in radians from velocity vector
- `timestamp: float` -- `time.monotonic()` value
- `frame_index: int` -- frame counter within the detection loop

**FrameRecord** bundles a timestamp, frame index, and a list of
`TagRecord` objects representing all tags detected in that frame.

Fields:
- `timestamp: float` -- `time.monotonic()` value
- `frame_index: int` -- frame counter
- `tags: list[TagRecord]` -- all tags detected in this frame

Both classes must provide a `to_dict()` method that returns a plain
Python dict suitable for `json.dumps()`. No numpy arrays in the output.

A `TagRecord.from_apriltag()` class method should convert from the
existing `AprilTag` model (plus velocity/heading info) to a `TagRecord`.

## Acceptance Criteria

- [ ] `TagRecord` dataclass exists in `src/aprilcam/detection.py` with all fields listed above
- [ ] `FrameRecord` dataclass exists in `src/aprilcam/detection.py` with all fields listed above
- [ ] `TagRecord.to_dict()` returns a plain dict with all fields; numpy arrays are converted to nested lists
- [ ] `FrameRecord.to_dict()` returns a dict with `timestamp`, `frame_index`, and `tags` (list of tag dicts)
- [ ] `TagRecord.from_apriltag(tag, vel_px, speed_px, vel_world, speed_world, heading_rad, frame_index)` class method works correctly
- [ ] `json.dumps(record.to_dict())` round-trips without error for both classes
- [ ] Optional fields (`world_xy`, `vel_px`, `vel_world`, etc.) serialize as `null` when `None`
- [ ] All existing tests still pass (`uv run pytest`)

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite, verify no regressions
- **New tests to write** (in `tests/test_detection.py`):
  - `test_tagrecord_construction` -- create a TagRecord with all fields, verify attribute access
  - `test_tagrecord_optional_none` -- create with optional fields as None, verify defaults
  - `test_tagrecord_to_dict` -- verify to_dict output matches expected structure
  - `test_tagrecord_to_dict_json_roundtrip` -- `json.loads(json.dumps(tr.to_dict()))` produces equivalent dict
  - `test_framerecord_construction` -- create FrameRecord with multiple TagRecords
  - `test_framerecord_to_dict` -- verify nested serialization
  - `test_tagrecord_from_apriltag` -- create an `AprilTag` from `models.py`, convert via `from_apriltag()`, verify field mapping
- **Verification command**: `uv run pytest tests/test_detection.py -v`
