# Plan: 002 — Refactor playfield.py — add optional polygon parameter

## Approach

Add an optional `polygon` keyword parameter to the `Playfield` dataclass.
Since `Playfield` uses `@dataclass`, add a new field with `default=None`
and handle it in `__post_init__` to set `_poly` if provided.

## Files to Modify

| File | Change |
|------|--------|
| `src/aprilcam/playfield.py` | Add `polygon` parameter to constructor, validate and store in `__post_init__` |
| `tests/test_playfield.py` | **New file** — unit tests for constructor, `_order_poly()`, `deskew()` |

## Implementation Details

### Constructor change

Add a new dataclass field after `detect_inverted`:

```python
polygon: Optional[np.ndarray] = None  # shape (4,2) float32, UL/UR/LR/LL
```

Add `__post_init__` method:

```python
def __post_init__(self):
    if self.polygon is not None:
        poly = np.asarray(self.polygon, dtype=np.float32).reshape(4, 2)
        self._poly = poly
```

### `update()` behavior

No change needed -- `update()` already returns early if `self._poly is not None`:
```python
def update(self, frame_bgr):
    if self._poly is not None:
        return  # already locked
```

This naturally handles the injected polygon case.

## Testing Plan

1. **`test_constructor_with_polygon`** — Create `Playfield(polygon=arr)`,
   verify `get_polygon()` returns the array.

2. **`test_constructor_without_polygon`** — Create `Playfield()`, verify
   `get_polygon()` returns `None`.

3. **`test_update_noop_with_injected_polygon`** — Create with polygon,
   call `update()` with a dummy frame, verify polygon unchanged.

4. **`test_order_poly_canonical`** — Call `_order_poly()` with corners
   in various ID-to-position mappings, verify output is always
   UL, UR, LR, LL.

5. **`test_deskew_output_dimensions`** — Inject a polygon with known
   geometry, call `deskew()` on a synthetic image, verify output
   width/height match expected values.

## Documentation Updates

None required.
