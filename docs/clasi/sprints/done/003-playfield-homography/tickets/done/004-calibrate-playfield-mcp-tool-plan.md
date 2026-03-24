# Plan: 004 — calibrate_playfield MCP tool

## Approach

Register a `calibrate_playfield` tool on the MCP server that takes a
playfield ID and physical measurements, computes the homography using
`calibrate_from_corners()` from ticket 001, and stores the result in
the playfield entry.

## Files to Modify

| File | Change |
|------|--------|
| `src/aprilcam/mcp_server.py` | Add `calibrate_playfield` tool |
| `tests/test_mcp_playfield.py` | Add calibration integration tests |

## Implementation Details

### calibrate_playfield tool

```python
@server.tool()
async def calibrate_playfield(
    playfield_id: str,
    width: float,
    height: float,
    units: str = "inch",
) -> list[TextContent]:
```

Flow:
1. Look up playfield entry: `playfield_registry.get(playfield_id)`
2. Validate units is "inch" or "cm"
3. Create `FieldSpec(width_in=width, height_in=height, units=units)`
4. Get polygon from `entry.playfield.get_polygon()` -- shape (4,2) in
   UL, UR, LR, LL order
5. Build corner dict mapping polygon points to CORNER_ID_MAP keys:
   - `poly[0]` -> `upper_left` (UL)
   - `poly[1]` -> `upper_right` (UR)
   - `poly[2]` -> `lower_right` (LR) -- NOTE: polygon order is UL,UR,LR,LL
   - `poly[3]` -> `lower_left` (LL)
6. Call `calibrate_from_corners(corner_dict, field_spec)`
7. Store `field_spec` and `homography` in the PlayfieldEntry
8. Return `{playfield_id, calibrated: true, width_cm, height_cm}`

### Error handling

- Unknown playfield_id: return error JSON
- Invalid units: return error JSON
- Homography computation failure: return error with details

## Testing Plan

1. **`test_calibrate_playfield_success`** — Create playfield from test
   image, calibrate with 40in x 35in. Verify `calibrated: true`,
   `width_cm` close to 101.6, `height_cm` close to 88.9.

2. **`test_calibrate_unknown_playfield`** — Call with bad ID, verify error.

3. **`test_calibrate_overwrites`** — Calibrate twice with different
   measurements. Verify second calibration replaces first.

4. **`test_calibrate_homography_accuracy`** — After calibration, apply
   homography to each corner pixel and verify world coords within 0.1 cm.

## Documentation Updates

None required.
