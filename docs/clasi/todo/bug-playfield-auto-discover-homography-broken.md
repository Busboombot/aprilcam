---
status: pending
---

# Bug: Playfield._auto_discover_homography is broken — calibration="auto" never loads saved homography

## Symptom

`Playfield(cam, width_cm=101, height_cm=89)` — the documented usage in
the package docstring and README — silently runs without loading the
saved calibration from `data/calibration.json`. Downstream world-
coordinate queries return whatever the Playfield falls back to (corner
detection at construction time, or nothing), not the persisted
calibration.

## Root cause

Three stacked bugs in [src/aprilcam/core/playfield.py:478-486](../../src/aprilcam/core/playfield.py#L478-L486):

```python
def _auto_discover_homography(self) -> Optional[np.ndarray]:
    try:
        from ..calibration.homography import discover_homography
        result = discover_homography()              # <-- no args
        if result is not None:
            return np.array(result, dtype=float) ...
    except Exception:                                # <-- swallows TypeError
        pass
    return None
```

1. `discover_homography` at [src/aprilcam/calibration/homography.py:68-73](../../src/aprilcam/calibration/homography.py#L68-L73)
   requires `device_name: str`, `width: int`, `height: int` — all
   positional, no defaults. The zero-arg call raises `TypeError`.
2. The bare `except Exception:` silently swallows that TypeError.
3. Default `calibration="auto"` in `Playfield.__init__` means every
   typical construction hits this path.

`discover_homography` also returns a `Path` (not a matrix), so even if
the args were right, the subsequent `np.array(result, ...)` would also
be wrong — the result needs to be loaded from the returned JSON file.

## Fix

`_auto_discover_homography` should:

1. Pull device_name from `self._camera` (`camera.device_name` or
   similar) and width/height from the camera's resolution.
2. Call `discover_homography(device_name, width, height, data_dir)`
   properly.
3. If it returns a `Path`, load the JSON and extract the homography
   matrix (mirror the logic in [src/aprilcam/config.py:165-175](../../src/aprilcam/config.py#L165-L175)
   and [src/aprilcam/stream.py:57](../../src/aprilcam/stream.py#L57)).
4. Narrow the except to `FileNotFoundError` / `json.JSONDecodeError`,
   not bare `Exception`. Let programming errors surface.

There's already a parallel loader at `Playfield._load_homography`
(line 488) that handles the file→matrix conversion —
`_auto_discover_homography` should call it once the Path is resolved.

## Downstream check

Navigator (consumer project, separate repo) uses
`Playfield(cam, width_cm=..., height_cm=...)` in the same idiomatic
way. Verify `navigate_to()` is actually operating on calibrated world
coords — this bug has likely been masked by corner-detection fallback
giving "close enough" numbers.

## Acceptance

- `Playfield(cam, width_cm=101, height_cm=89)` on the Global Shutter
  Camera loads the homography from `data/calibration.json` and
  `playfield.homography` matches the stored matrix (within float
  tolerance).
- A unit test covers the auto-discovery path end-to-end (mock camera
  with device_name + resolution, calibration.json on disk).
- Narrow the `except` clause in `_auto_discover_homography` so
  programming errors aren't silently swallowed.
- Navigator's `navigate_to()` verified to use calibrated coords, not
  fallback.
