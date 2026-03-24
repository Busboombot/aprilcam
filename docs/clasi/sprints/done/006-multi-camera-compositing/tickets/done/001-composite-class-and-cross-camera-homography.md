---
id: '001'
title: Composite class and cross-camera homography
status: done
use-cases:
- SUC-001
- SUC-002
- SUC-004
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Composite class and cross-camera homography

## Description

Create the core `src/aprilcam/composite.py` module containing the
`Composite` dataclass, `CompositeManager` class, and the cross-camera
homography computation and tag-mapping functions. This module is the
foundation for all multi-camera compositing functionality -- it defines
the data structures, lifecycle management, and coordinate transformation
logic that the MCP tools (tickets 002-003) will expose.

### Components

1. **`Composite` dataclass** with fields:
   - `composite_id: str` -- unique identifier
   - `primary_camera_id: str` -- camera providing visual frames
   - `secondary_camera_id: str` -- camera providing tag detections
   - `homography: np.ndarray` -- 3x3 matrix mapping secondary pixels
     to primary pixels
   - `playfield_id: Optional[str]` -- optional associated playfield
   - `reprojection_error: float` -- RMS error from homography computation

2. **`CompositeManager` class** with methods:
   - `create(primary_id, secondary_id, homography, reprojection_error,
     playfield_id?) -> Composite`
   - `get(composite_id) -> Composite` (raises if not found)
   - `destroy(composite_id)` (raises if not found)
   - `list() -> list[Composite]`

3. **`compute_cross_camera_homography(primary_points, secondary_points)
   -> tuple[np.ndarray, float]`** -- Uses `cv2.findHomography` to
   compute a 3x3 homography mapping secondary camera pixel coordinates
   to primary camera pixel coordinates. Returns the matrix and the RMS
   reprojection error. Raises `ValueError` if fewer than 4 point pairs
   are provided or if the resulting homography is degenerate.

4. **`map_tags_to_primary(detections, homography) -> list[dict]`** --
   Transforms tag corner points and centers from secondary camera space
   to primary camera space using `cv2.perspectiveTransform`. Each result
   dict includes `id`, `center_px`, `corners_px`, `orientation_yaw`.

## Acceptance Criteria

- [ ] `Composite` dataclass exists with all specified fields (`composite_id`, `primary_camera_id`, `secondary_camera_id`, `homography`, `playfield_id`, `reprojection_error`)
- [ ] `CompositeManager.create()` stores a composite and returns it with a unique ID
- [ ] `CompositeManager.get()` retrieves a composite by ID and raises `KeyError` for unknown IDs
- [ ] `CompositeManager.destroy()` removes a composite and raises `KeyError` for unknown IDs
- [ ] `CompositeManager.list()` returns all active composites
- [ ] `compute_cross_camera_homography` returns a valid 3x3 ndarray and a float reprojection error
- [ ] `compute_cross_camera_homography` raises `ValueError` when given fewer than 4 point pairs
- [ ] `compute_cross_camera_homography` raises `ValueError` for degenerate point configurations (e.g., collinear points)
- [ ] `map_tags_to_primary` correctly transforms tag centers and corners from secondary to primary coordinates using `cv2.perspectiveTransform`
- [ ] `map_tags_to_primary` preserves tag ID and orientation in the output
- [ ] `map_tags_to_primary` returns an empty list when given no detections
- [ ] Module has no import-time dependency on camera hardware (pure computation)

## Testing

- **Existing tests to run**: `uv run pytest tests/` -- full suite to verify no regressions
- **New tests to write**: `tests/test_composite.py`
  - Test `Composite` dataclass instantiation and field access
  - Test `CompositeManager` CRUD lifecycle (create, get, list, destroy)
  - Test `CompositeManager.get` with invalid ID raises `KeyError`
  - Test `CompositeManager.destroy` with invalid ID raises `KeyError`
  - Test `compute_cross_camera_homography` with known point correspondences (verify matrix correctness and low reprojection error)
  - Test `compute_cross_camera_homography` with <4 points raises `ValueError`
  - Test `compute_cross_camera_homography` with collinear points raises `ValueError`
  - Test `map_tags_to_primary` with synthetic tag detections and a known homography (verify transformed coordinates match expected values)
  - Test `map_tags_to_primary` with empty input returns empty list
- **Verification command**: `uv run pytest tests/test_composite.py -v`
