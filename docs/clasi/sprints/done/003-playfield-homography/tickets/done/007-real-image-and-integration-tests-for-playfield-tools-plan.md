# Plan: 007 — Real-image and integration tests for playfield tools

## Approach

Create a dedicated test file using the real playfield images from
`tests/data/` to validate the full pipeline end-to-end. Mock the camera
to return these images. Compare computed homography against the reference
in `data/homography.json`.

## Files to Modify

| File | Change |
|------|--------|
| `tests/test_playfield_real.py` | **New file** — real-image unit tests for detection, deskew, homography |
| `tests/test_mcp_playfield.py` | Add end-to-end MCP integration tests with real images |

## Implementation Details

### Camera mock helper

Create a reusable mock that wraps an image file as a camera-like object:

```python
class MockCamera:
    def __init__(self, image_path: str):
        self._frame = cv2.imread(image_path)

    def read(self):
        return True, self._frame.copy()

    def isOpened(self):
        return True

    def release(self):
        pass
```

### test_playfield_real.py tests

1. **`test_detect_corners_cam3`** — Load `playfield_cam3.jpg`, run
   `Playfield._detect_corners()`, verify all 4 ArUco IDs (0-3) found.

2. **`test_detect_corners_cam3_moved`** — Same for `playfield_cam3_moved.jpg`.

3. **`test_create_playfield_from_real_image`** — Create Playfield,
   call `update()` with real frame, verify polygon locked.

4. **`test_deskew_real_image`** — Deskew the real frame, verify output
   is rectangular and dimensions are reasonable (width > height for
   landscape playfield).

5. **`test_homography_matches_reference`** — Load `data/homography.json`,
   create playfield from `playfield_cam3.jpg`, calibrate with same
   field spec (40in x 35in), compare computed H against reference
   within tolerance (element-wise atol=0.5).

6. **`test_apriltag_detection_after_deskew`** — Deskew the real frame,
   run AprilTag 36h11 detection on the deskewed image, verify IDs
   0-6 and 30 are found.

### MCP integration tests (in test_mcp_playfield.py)

7. **`test_full_flow_real_image`** — Register mock camera, call
   `create_playfield`, `calibrate_playfield(40, 35, "inch")`,
   `get_playfield_info`, `capture_frame(playfield_id)`. Verify each
   step returns expected results.

## Testing Plan

All tests in this ticket are the testing plan. Run with:
```
uv run pytest tests/test_playfield_real.py tests/test_mcp_playfield.py -v
```

## Documentation Updates

None required.
