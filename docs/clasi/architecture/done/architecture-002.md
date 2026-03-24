---
sprint: "002"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 002: Core MCP Server & Camera Tools

## What Changed

### New Module: `src/aprilcam/mcp_server.py`

This is the primary deliverable. It contains:

- **`CameraRegistry`** class: manages a dictionary mapping UUID string
  handles to open capture objects (`cv2.VideoCapture` or
  `ScreenCaptureMSS`). Provides `open()`, `close()`, `get()`, and
  `close_all()` methods. On deletion or shutdown, automatically releases
  any remaining open captures.

- **MCP Server setup**: creates an `mcp.server.Server` instance (named
  `"aprilcam"`) and registers four tools via `@server.tool()` decorators:

  | Tool             | Parameters                                      | Returns                        |
  |------------------|-------------------------------------------------|--------------------------------|
  | `list_cameras`   | (none)                                          | JSON array of `{index, name, backend}` |
  | `open_camera`    | `index?`, `pattern?`, `source?`, `backend?`     | `{camera_id: "<uuid>"}`       |
  | `close_camera`   | `camera_id`                                     | `{status: "closed"}`          |
  | `capture_frame`  | `camera_id`, `format?` ("base64"\|"file"), `quality?` | base64 image or file path |

- **`main()` function**: entry point that runs the server using
  `mcp.server.stdio.stdio_server()` as the async transport. Called from
  the CLI.

### New CLI Entry Point

A new script entry `aprilcam-mcp` is added to `[project.scripts]` in
`pyproject.toml`:

```
aprilcam-mcp = "aprilcam.mcp_server:main"
```

This keeps the MCP server entry point separate from the existing
`aprilcam` CLI, avoiding the need to refactor `aprilcam_cli.py` into a
subcommand dispatcher in this sprint.

### New Dependency

`mcp` (Python MCP SDK) is added to `[project.dependencies]` in
`pyproject.toml`:

```
"mcp>=1.0",
```

### New Test File: `tests/test_mcp_server.py`

Unit tests for `CameraRegistry` and the tool handler functions. Uses
mock capture objects to avoid needing real cameras in CI.

## Why

The project's vision is to make AprilCam's capabilities available to AI
agents via MCP. Sprint 001 restructured the CLI foundation. This sprint
delivers the core MCP server and the first set of tools (camera
management), which are prerequisites for all subsequent MCP tools
(detection, homography, image processing).

Camera management is the natural first tool set because every other
operation (capturing frames, detecting tags, calibrating homography)
requires an open camera. The `CameraRegistry` pattern establishes the
handle-based resource management model that later sprints will build on.

## Impact on Existing Components

### `camutil.py` -- No Changes

`list_cameras()` and `select_camera_by_pattern()` are called as-is from
the MCP tool handlers. No modifications needed. The functions already
return structured data (`List[CameraInfo]`) that maps directly to JSON.

### `config.py` -- No Changes

`AppConfig.get_camera()` is not used by the MCP server. The MCP tools
manage cameras directly through `CameraRegistry` rather than through
`AppConfig`, because the MCP server needs explicit handle-based lifecycle
management rather than the "open and return" pattern that `get_camera()`
provides. `AppConfig` remains available for the traditional CLI commands.

### `screencap.py` -- No Changes

`ScreenCaptureMSS` already implements the `isOpened()`/`read()`/
`release()` interface that `CameraRegistry` expects. It is used as-is
when the agent opens a screen capture source.

### `pyproject.toml` -- Modified

- New dependency: `mcp>=1.0`
- New script entry: `aprilcam-mcp`

### CLI Commands -- No Changes

Existing CLI commands (`aprilcam`, `cameras`, `homocal`, etc.) are
unaffected. The MCP server is a separate entry point that does not
modify or depend on the CLI modules.

## New Interfaces

### CameraRegistry API

```python
class CameraRegistry:
    def open(self, capture: Any) -> str:
        """Register an open capture object. Returns a UUID handle."""

    def get(self, camera_id: str) -> Any:
        """Retrieve capture by handle. Raises KeyError if invalid."""

    def close(self, camera_id: str) -> None:
        """Release and unregister. Raises KeyError if invalid."""

    def close_all(self) -> None:
        """Release all open captures."""

    def list_open(self) -> list[str]:
        """Return list of active camera_id handles."""
```

### MCP Tool Schemas (JSON-RPC)

Each tool is registered with the MCP SDK's `@server.tool()` decorator,
which automatically generates the JSON schema from the Python function
signature and type annotations.

## Data Flow

```
Agent (Claude)
    |  stdio (JSON-RPC)
    v
MCP Server (mcp_server.py)
    |
    +-- list_cameras --> camutil.list_cameras() --> [CameraInfo, ...]
    |
    +-- open_camera --> cv2.VideoCapture(idx) or ScreenCaptureMSS()
    |                   --> CameraRegistry.open(cap) --> camera_id
    |
    +-- capture_frame --> CameraRegistry.get(id) --> cap.read()
    |                     --> cv2.imencode('.jpg') --> base64 or tmpfile
    |
    +-- close_camera --> CameraRegistry.close(id) --> cap.release()
```

## Migration Concerns

None. This sprint adds new modules and a new entry point without
modifying any existing code. Existing CLI commands continue to work
unchanged. The new `mcp` dependency is additive.

Users who do not need the MCP server can ignore the `aprilcam-mcp`
entry point entirely. The `mcp` package will be installed as a
dependency but has no side effects unless the server is explicitly
started.
