---
id: '002'
title: AprilTag family field and detection pipeline update
status: done
use-cases:
- SUC-001
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# AprilTag family field and detection pipeline update

## Description

Add a `family: str` field to the `AprilTag` dataclass so that each detected tag
carries its family name (e.g., "36h11", "25h9"). Currently, the detection pipeline
knows which family it's using but doesn't record this on individual tag objects.

This is a small, isolated change that threads the family name from the detector
configuration through to each detection result. It supports the broader sprint
goal of richer tag metadata in FrameEntry results.

### Changes required

1. **`AprilTag` dataclass** (`src/aprilcam/models.py`): Add `family: str` field
   with a sensible default (e.g., `"36h11"` since that's the primary family used).

2. **`AprilTag.from_corners()`** (`src/aprilcam/models.py`): Accept an optional
   `family` parameter and pass it through to the constructor.

3. **`AprilCam.detect_apriltags()`** (`src/aprilcam/aprilcam.py`): After detection,
   extract the family name from the detector's configuration and pass it to
   `AprilTag.from_corners()` for each detection.

## Acceptance Criteria

- [x] `AprilTag` dataclass has a `family: str` field
- [x] `AprilTag.from_corners()` accepts and passes through `family` parameter
- [x] `AprilCam.detect_apriltags()` populates `family` from detector config
- [x] Default value is `"36h11"` for backward compatibility
- [x] Existing code that constructs `AprilTag` without `family` still works
- [x] Family name appears in tag detection results returned by MCP tools

## Implementation Notes

### Key files
- `src/aprilcam/models.py` -- `AprilTag` dataclass, `from_corners()` classmethod
- `src/aprilcam/aprilcam.py` -- `detect_apriltags()` method on `AprilCam`

### Design decisions
- Use a default value for `family` so existing call sites don't break
- The family name comes from the detector object's configuration, not from
  individual detections (the detector is configured for one family at a time)
- This is a data-only change; no behavioral logic changes

## Testing

- **Existing tests to run**: `uv run pytest` (full suite, ensure no regressions)
- **New tests to write**:
  - `test_apriltag_family_field` -- verify field exists and default value
  - `test_apriltag_from_corners_with_family` -- verify family threads through
  - `test_detect_apriltags_populates_family` -- integration test with detector
- **Verification command**: `uv run pytest`
