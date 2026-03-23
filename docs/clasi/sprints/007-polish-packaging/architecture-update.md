---
sprint: "007"
status: draft
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Architecture Update -- Sprint 007: Polish & Packaging

## What Changed

### 1. Test Infrastructure (new)

- **`tests/` directory** added at project root with pytest configuration.
- **`tests/conftest.py`** provides shared fixtures: `mock_camera` (a
  fake `cv2.VideoCapture` that yields synthetic frames with embedded
  AprilTag patterns), `mcp_server_process` (launches `aprilcam mcp` as
  a subprocess for integration tests), and `synthetic_frame` (a numpy
  array with known tag positions).
- **`tests/test_mcp_lifecycle.py`** — integration tests for MCP server
  startup, tool listing, camera open/close, frame capture.
- **`tests/test_error_handling.py`** — tests for invalid inputs, missing
  cameras, camera disconnection during detection loop.
- **`tests/test_detection_loop.py`** — tests for detection loop
  start/stop, ring buffer queries, tag history.
- **`tests/test_packaging.py`** — tests that verify entry point
  resolution and import paths.
- **`tests/test_ring_buffer.py`** — unit tests for ring buffer
  insert, query, memory cap enforcement.

### 2. Error Handling Layer (modified)

- **MCP tool error decorator** — a `@mcp_error_handler` decorator (or
  context manager) added to `src/aprilcam/mcp/` that wraps every tool
  handler. Catches exceptions and returns structured MCP error responses
  with appropriate error codes and human-readable messages.
- **Camera disconnection handling** — the detection loop thread checks
  for `None` frames from `VideoCapture.read()`. On three consecutive
  failures, it stops the loop, sets an error flag, and stores a
  disconnect error that subsequent `query_tags` calls can report.
- **Parameter validation** — input validation added to each MCP tool
  handler (type checks, range checks, required-field checks) with
  descriptive error messages returned before any processing begins.

### 3. MCP Tool Documentation (modified)

- Every MCP tool function gains a structured docstring following a
  consistent format: one-line summary, Parameters section with types,
  Returns section describing the response shape, Errors section listing
  possible error codes.
- Tool descriptions registered with the MCP framework are updated to
  match the docstrings.
- **`docs/mcp-tools.md`** (new) — a reference document with example
  JSON request and response for every tool.

### 4. Entry Point Consolidation (modified)

- **`pyproject.toml` `[project.scripts]`** reduced to a single entry
  point: `aprilcam = "aprilcam.cli.aprilcam_cli:main"`.
- Legacy standalone entry points removed: `taggen`, `arucogen`,
  `homocal`, `cameras`, `playfield`, `atscreencap`, `aprilcap`,
  `apriltest`.
- **`src/aprilcam/cli/aprilcam_cli.py`** updated to use argparse
  subcommands: `aprilcam mcp`, `aprilcam taggen`, `aprilcam arucogen`,
  `aprilcam cameras`. Each subcommand delegates to the existing CLI
  module's `main()` function.

### 5. Ring Buffer Memory Cap (modified)

- Ring buffer class gains an optional `max_bytes` parameter. When total
  stored frame data exceeds this limit, oldest entries are evicted
  regardless of the frame-count limit. Default: 256 MB.
- Memory tracking is approximate (based on numpy array `nbytes`).

### 6. Deprecated Code Removal (removed)

- **`src/aprilcam/cli/atscreencap_cli.py`** — removed. Screen capture
  functionality is available through the MCP `capture_frame` tool with
  screen source.
- **`src/aprilcam/cli/apriltest_cli.py`** — removed. Testing is now
  done through pytest.
- **`src/aprilcam/cli/aprilcap_cli.py`** — removed. Capture
  functionality is available through MCP tools.
- **`src/aprilcam/cli/playfield_cli.py`** — evaluated for removal or
  integration as `aprilcam playfield` subcommand. If the module provides
  functionality not covered by MCP tools (e.g., interactive pygame
  display), it is kept as a subcommand; otherwise removed.

### 7. pyproject.toml Metadata (modified)

- `description` updated to reflect MCP server purpose.
- `authors` updated with real name/email.
- `keywords` expanded: `apriltag`, `aruco`, `fiducial`, `mcp`,
  `robotics`, `computer-vision`, `opencv`.
- `classifiers` added: Development Status, Framework, Topic, License,
  Python version.
- `project.urls` added: Homepage, Repository, Documentation.
- `pygame` dependency moved to optional `[project.optional-dependencies]`
  since the MCP server does not require it.

### 8. README.md (new)

- Installation instructions (pipx, pip, development setup).
- MCP server configuration (Claude Desktop, generic MCP client).
- CLI subcommand reference.
- Quick-start example for AI agent integration.

## Why

- **Testing (SUC-001, SUC-003, SUC-006):** Six sprints of functionality
  with zero test coverage is a reliability risk. Integration tests with
  mocked cameras enable CI without hardware.
- **Error handling (SUC-002, SUC-003):** AI agents need structured
  errors to recover gracefully. Unhandled exceptions break agent
  workflows.
- **Documentation (SUC-005):** Self-documenting MCP tools are essential
  for agent usability. Agents discover tools via `tools/list` and need
  complete parameter/return descriptions.
- **Packaging (SUC-004):** Clean `pipx install` is a project requirement
  from the overview. Legacy entry points confuse the install.
- **Performance (SUC-006):** Detection loops must not block MCP
  responses or the agent cannot query state during active detection.
- **Code hygiene:** Deprecated CLI modules add confusion and maintenance
  burden.

## Impact on Existing Components

- **`aprilcam_cli.py`** — rewritten to use argparse subcommands.
  Breaking change for users who invoke the old CLI directly, but the
  MCP server interface (the primary usage path) is unchanged.
- **`pyproject.toml`** — entry point changes mean a reinstall is
  required. Users who had `taggen` or `arucogen` as standalone commands
  must switch to `aprilcam taggen` / `aprilcam arucogen`.
- **Ring buffer** — new `max_bytes` parameter with a sensible default.
  Existing behavior is preserved unless memory pressure triggers
  eviction.
- **MCP tool handlers** — additional error handling wrapping. Return
  values are unchanged for successful calls; error responses gain a
  consistent structure.
- **No changes to core detection logic, homography, or playfield
  modules** — this sprint wraps and documents existing functionality
  without altering algorithms.

## Migration Concerns

- **Entry point rename:** Users or scripts referencing standalone
  commands (`taggen`, `arucogen`, `cameras`, `homocal`) must update to
  `aprilcam <subcommand>`. This is a breaking change for CLI users but
  does not affect MCP tool consumers.
- **Reinstall required:** After this sprint, `pipx reinstall aprilcam`
  (or `pip install -e .`) is needed to pick up the new entry points.
- **pygame optional:** Moving pygame to optional dependencies means
  `pipx install aprilcam` no longer installs pygame by default. Users
  who need the playfield simulator must install with
  `pipx install aprilcam[simulator]` or similar.
- **No data migration.** No persistent data formats are changed.
- **No API breaking changes for MCP tools.** Tool names, parameters,
  and response shapes are preserved. Only error responses gain a
  consistent structure (previously they were inconsistent or missing).
