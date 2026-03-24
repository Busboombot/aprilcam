---
id: 009
title: Mask source image to playfield polygon before deskew warp
status: done
use-cases: []
depends-on: []
github-issue: ''
todo: ''
---

# Mask source image to playfield polygon before deskew warp

## Description

`Playfield.deskew()` uses `cv2.warpPerspective` to transform the full
source image. Pixels outside the playfield polygon (defined by the 4
ArUco corner marker centers) bleed into the output image.

The output boundary should be exactly at the corner marker centers.
Nothing outside those centers should be visible.

Fix: mask the source image to the playfield polygon before warping.

## Acceptance Criteria

- [ ] `Playfield.deskew()` masks source image to playfield polygon before warping
- [ ] Output contains only content from inside the playfield boundary
- [ ] Pixels outside the polygon are black (0,0,0) in the output
- [ ] All existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Test that verifies edge pixels outside the polygon are masked to black
- **Verification command**: `uv run pytest`
