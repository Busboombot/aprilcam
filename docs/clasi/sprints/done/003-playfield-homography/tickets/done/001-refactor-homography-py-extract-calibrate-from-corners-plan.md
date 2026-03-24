# Plan: 001 — Refactor homography.py — extract calibrate_from_corners()

## Approach

Extract the calibration logic currently inline in `main()` (lines 218-232
of `homography.py`) into a new public function `calibrate_from_corners()`.
The function builds world-coordinate correspondences from the corner
positions and field spec, then delegates to the existing
`compute_homography()` function.

The `main()` function is refactored to call `calibrate_from_corners()`
instead of computing the correspondences inline.

## Files to Modify

| File | Change |
|------|--------|
| `src/aprilcam/homography.py` | Add `calibrate_from_corners(pixel_corners, field_spec)` function; refactor `main()` to call it |
| `tests/test_homography.py` | **New file** — unit tests for `calibrate_from_corners()` and `FieldSpec` |

## Implementation Details

### `calibrate_from_corners(pixel_corners, field_spec)`

```python
def calibrate_from_corners(
    pixel_corners: Dict[str, Tuple[float, float]],
    field_spec: FieldSpec,
) -> np.ndarray:
    """Compute homography from four corner positions and a field spec.

    Args:
        pixel_corners: Dict with keys 'upper_left', 'upper_right',
            'lower_left', 'lower_right', each a (x, y) tuple.
        field_spec: FieldSpec with physical dimensions.

    Returns:
        3x3 homography matrix mapping [u,v,1] pixels to [X,Y,W] world cm.
    """
```

- Build `world_pts_cm` array: UL=(0,0), UR=(w_cm,0), LL=(0,h_cm), LR=(w_cm,h_cm)
- Build `pixel_pts` array from the dict values in matching order
- Call `compute_homography(pixel_pts, world_pts_cm)` and return result

### Refactor `main()`

Replace lines 218-232 with:
```python
field = FieldSpec(width_in=float(args.width), ...)
H = calibrate_from_corners(found, field)
```

The `found` dict from `run_once()` already uses the correct key names.

## Testing Plan

1. **`test_calibrate_from_corners_identity`** — Pass corners at
   (0,0), (100,0), (0,100), (100,100) with field_spec 100cm x 100cm.
   Homography should be close to identity.

2. **`test_calibrate_from_corners_scaled`** — Pass pixel corners at
   known positions with field_spec 40in x 35in. Verify mapping of
   each corner pixel to its expected world coordinate.

3. **`test_field_spec_inch_to_cm`** — FieldSpec(40, 35, "inch") yields
   width_cm=101.6, height_cm=88.9.

4. **`test_field_spec_cm_passthrough`** — FieldSpec(100, 80, "cm")
   yields width_cm=100, height_cm=80.

## Documentation Updates

None required -- this is an internal refactoring. The function docstring
provides API documentation.
