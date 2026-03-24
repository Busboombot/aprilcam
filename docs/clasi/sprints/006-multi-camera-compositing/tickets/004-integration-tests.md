---
id: "004"
title: "Integration tests"
status: todo
use-cases:
  - SUC-001
  - SUC-002
  - SUC-003
  - SUC-004
  - SUC-005
depends-on:
  - "003"
github-issue: ""
todo: ""
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Integration tests

## Description

Write end-to-end integration tests that exercise the full multi-camera
compositing pipeline: creating a composite from two synthetic camera
feeds, querying composite frames and tags, and verifying that individual
cameras remain independently accessible while a composite exists.

These tests use programmatically generated images (numpy arrays) with
ArUco markers drawn at known positions using `cv2.aruco.generateImageMarker`.
Both "cameras" are synthetic -- no physical hardware is needed. The tests
mock the camera manager to return these synthetic frames on capture.

### Key scenarios to cover

1. **Full round-trip with automatic alignment**: Open two mock cameras,
   place shared ArUco markers in both frames at known (but different)
   pixel positions, call `create_composite`, then `get_composite_tags`,
   and verify that tag positions are correctly mapped from secondary to
   primary coordinates.

2. **Full round-trip with manual correspondence**: Same as above but
   supply explicit correspondence points instead of relying on ArUco
   marker detection.

3. **Composite frame rendering**: Call `get_composite_frame` and verify
   the returned image is non-empty, is the correct resolution (matching
   the primary camera), and contains overlay pixels at the expected
   mapped tag positions.

4. **Individual camera independence**: While a composite exists, call
   `capture_frame` on each camera individually and verify the responses
   are unaffected by the composite (no overlays, original coordinates).

5. **Composite lifecycle**: Create a composite, verify it appears in
   listing, destroy it, verify it is gone, and confirm both cameras
   still work independently.

## Acceptance Criteria

- [ ] Integration test creates two synthetic camera feeds with ArUco markers at known positions
- [ ] Test verifies `create_composite` auto-detect mode produces a valid composite with low reprojection error (<10px)
- [ ] Test verifies `create_composite` manual mode produces a valid composite from supplied correspondences
- [ ] Test verifies `get_composite_tags` returns tags with centers mapped to primary camera coordinates (within 5px of expected positions)
- [ ] Test verifies `get_composite_tags` returns correct tag IDs from the secondary camera
- [ ] Test verifies `get_composite_frame` returns a non-empty image matching primary camera resolution
- [ ] Test verifies `get_composite_frame` output contains overlay pixels (differs from raw primary frame)
- [ ] Test verifies `capture_frame(primary_camera_id)` returns raw frame without composite overlays while composite exists
- [ ] Test verifies `capture_frame(secondary_camera_id)` returns raw frame while composite exists
- [ ] Test verifies tag detection on individual cameras returns tags in that camera's own coordinate space (not mapped)
- [ ] Test verifies composite lifecycle: create -> list (present) -> destroy -> list (absent)
- [ ] Test verifies both cameras remain accessible after composite is destroyed
- [ ] All tests run without physical cameras (fully synthetic/mocked)
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite must pass
- **New tests to write**: `tests/test_composite_integration.py`
  - `test_auto_detect_round_trip` -- create composite with ArUco markers, query tags, verify mapped coordinates
  - `test_manual_correspondence_round_trip` -- create composite with explicit points, query tags, verify
  - `test_composite_frame_has_overlays` -- get composite frame, compare with raw primary frame
  - `test_individual_camera_unaffected` -- capture from each camera while composite exists, verify no overlays
  - `test_composite_lifecycle` -- create, list, destroy, list, verify cameras still work
  - `test_tag_detection_independence` -- run tag detection on individual cameras, verify own-coordinate-space results
- **Verification command**: `uv run pytest tests/test_composite_integration.py -v`
