# Plan: 003 — PlayfieldRegistry and create_playfield MCP tool

## Approach

Add `PlayfieldEntry` dataclass and `PlayfieldRegistry` class to
`mcp_server.py`, following the same pattern as `CameraRegistry`. Then
register a `create_playfield` tool on the MCP server.

## Files to Modify

| File | Change |
|------|--------|
| `src/aprilcam/mcp_server.py` | Add `PlayfieldEntry`, `PlayfieldRegistry`, `playfield_registry` instance, `create_playfield` tool |
| `tests/test_mcp_playfield.py` | **New file** — integration tests for `create_playfield` |

## Implementation Details

### PlayfieldEntry

```python
@dataclass
class PlayfieldEntry:
    playfield_id: str
    camera_id: str
    playfield: Playfield
    field_spec: Optional[FieldSpec] = None
    homography: Optional[np.ndarray] = None
```

### PlayfieldRegistry

```python
class PlayfieldRegistry:
    def __init__(self):
        self._playfields: dict[str, PlayfieldEntry] = {}

    def register(self, entry: PlayfieldEntry) -> None: ...
    def get(self, playfield_id: str) -> PlayfieldEntry: ...  # raises KeyError
    def list(self) -> list[str]: ...
    def remove(self, playfield_id: str) -> None: ...
    def find_by_camera(self, camera_id: str) -> Optional[str]: ...
```

### create_playfield tool

```python
@server.tool()
async def create_playfield(
    camera_id: str,
    max_frames: int = 30,
) -> list[TextContent]:
```

Flow:
1. `registry.get(camera_id)` -- validate camera exists
2. Create `Playfield(detect_inverted=True)`
3. Loop up to `max_frames`: capture frame, call `pf.update(frame)`
4. If `pf.get_polygon()` is not None: register and return success
5. Else: detect which corner IDs are missing, return error

Playfield ID pattern: `pf_{camera_id}`

### Module-level instance

```python
playfield_registry = PlayfieldRegistry()
```

## Testing Plan

1. **`test_create_playfield_success`** — Mock camera to return
   `tests/data/playfield_cam3.jpg`. Call `create_playfield`. Verify
   response contains `playfield_id`, `corners` (4 points), `calibrated: false`.

2. **`test_create_playfield_missing_markers`** — Mock camera to return
   a blank image. Verify error response lists missing corner IDs.

3. **`test_create_playfield_unknown_camera`** — Call with non-existent
   camera_id. Verify error.

4. **`test_create_playfield_replaces_existing`** — Call twice for same
   camera. Verify registry has exactly one entry.

5. **`test_playfield_registry_crud`** — Unit test register/get/list/remove.

## Documentation Updates

None required.
