# Plan: 006 — Playfield-as-camera pass-through in capture tool

## Approach

Add a resolution step at the top of the `capture_frame` tool handler
that checks whether the `camera_id` is actually a `playfield_id`. If
so, capture from the underlying camera and apply deskew before encoding.

## Files to Modify

| File | Change |
|------|--------|
| `src/aprilcam/mcp_server.py` | Modify `capture_frame` to resolve playfield IDs and apply deskew |
| `tests/test_mcp_playfield.py` | Add pass-through capture tests |

## Implementation Details

### capture_frame modification

At the top of `capture_frame`, before the existing camera registry
lookup, add:

```python
# Resolve playfield_id -> underlying camera + deskew
pf_entry = None
try:
    pf_entry = playfield_registry.get(camera_id)
    # Override camera_id to the underlying camera
    actual_camera_id = pf_entry.camera_id
except KeyError:
    actual_camera_id = camera_id
```

Then use `actual_camera_id` for the camera registry lookup. After
capturing the frame, if `pf_entry` is not None, apply deskew:

```python
if pf_entry is not None:
    frame = pf_entry.playfield.deskew(frame)
```

This is 4-5 lines of code inserted before the existing logic.

### Encoding

The rest of the function (JPEG encode, base64/file return) works
unchanged on the deskewed frame.

## Testing Plan

1. **`test_capture_via_playfield_returns_deskewed`** — Create playfield
   from test image, capture via playfield_id. Verify returned image
   dimensions differ from raw camera frame (deskew produces different
   size).

2. **`test_capture_via_camera_unchanged`** — Verify existing camera
   capture still works after the modification.

3. **`test_capture_via_playfield_file_format`** — Capture with
   `format="file"`, verify file is written and path returned.

4. **`test_capture_unknown_id`** — Verify error when ID is not in
   either registry.

## Documentation Updates

None required.
