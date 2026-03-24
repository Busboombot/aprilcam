# Plan: 005 — get_playfield_info MCP tool

## Approach

Register a `get_playfield_info` read-only tool on the MCP server that
returns the full state of a playfield entry as JSON.

## Files to Modify

| File | Change |
|------|--------|
| `src/aprilcam/mcp_server.py` | Add `get_playfield_info` tool |
| `tests/test_mcp_playfield.py` | Add info query tests |

## Implementation Details

### get_playfield_info tool

```python
@server.tool()
async def get_playfield_info(
    playfield_id: str,
) -> list[TextContent]:
```

Flow:
1. Look up entry: `playfield_registry.get(playfield_id)`
2. Build response dict:
   - `playfield_id`: entry.playfield_id
   - `camera_id`: entry.camera_id
   - `corners`: entry.playfield.get_polygon().tolist() -- 4x2 list
   - `calibrated`: entry.homography is not None
3. If calibrated, add:
   - `width_cm`: entry.field_spec.width_cm
   - `height_cm`: entry.field_spec.height_cm
   - `homography`: entry.homography.tolist() -- 3x3 nested list
4. Return as JSON TextContent

### Error handling

- Unknown playfield_id: return error JSON

## Testing Plan

1. **`test_get_info_uncalibrated`** — Create playfield, query info.
   Verify `calibrated: false`, corners present, no width_cm/height_cm.

2. **`test_get_info_calibrated`** — Create + calibrate playfield, query
   info. Verify all fields present including width_cm, height_cm, homography.

3. **`test_get_info_unknown`** — Query non-existent playfield, verify error.

4. **`test_get_info_corners_order`** — Verify corners are in UL, UR, LR, LL
   order matching the polygon.

## Documentation Updates

None required.
