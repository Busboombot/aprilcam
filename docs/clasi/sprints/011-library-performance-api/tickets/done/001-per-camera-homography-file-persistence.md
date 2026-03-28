---
id: "001"
title: "Per-camera homography file persistence"
status: done
use-cases: [SUC-011-002]
depends-on: []
github-issue: ""
todo: ""
---

# Per-camera homography file persistence

## Description

Add per-camera homography file naming and auto-discovery. Files are
keyed by slugified device name + resolution:
`data/homography-brio-501-1920x1080.json`.

### Changes

1. **`camutil.py`**: Add `camera_slug(device_name, width, height)` that
   returns a slug like `brio-501-1920x1080`. Slugify: lowercase, replace
   spaces/special chars with hyphens, strip leading/trailing hyphens.

2. **`homography.py`**: Add `homography_path(slug, data_dir="data")` →
   returns `Path(data_dir) / f"homography-{slug}.json"`. Add
   `discover_homography(device_name, width, height, data_dir="data")`
   that finds the matching file or falls back to `data/homography.json`.

3. **`homography.py`**: Update `save_homography()` to accept a slug
   and save to the per-camera path.

4. **`mcp_server.py`**: Update `calibrate_playfield` to save with
   per-camera naming when camera metadata is available.

## Acceptance Criteria

- [ ] `camera_slug("Brio 501", 1920, 1080)` → `"brio-501-1920x1080"`
- [ ] `discover_homography(...)` finds per-camera file if it exists
- [ ] Falls back to `data/homography.json` if no per-camera file found
- [ ] Calibration saves to per-camera path
- [ ] Existing `data/homography.json` files still work

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Unit tests for `camera_slug()`,
  `homography_path()`, `discover_homography()` with tmp directories
- **Verification command**: `uv run pytest`
