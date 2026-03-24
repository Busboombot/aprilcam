---
id: '007'
title: Polish & Packaging
status: done
branch: sprint/007-polish-packaging
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
- SUC-006
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Sprint 007: Polish & Packaging

## Goals

Harden AprilCam for production use. The previous six sprints built the
full feature set (camera management, playfield/homography, detection
loop/ring buffer, image processing tools, multi-camera compositing).
This sprint focuses on reliability, usability, and distribution:

1. Add end-to-end integration tests covering the MCP server lifecycle.
2. Audit and improve error handling across all MCP tools.
3. Write clear MCP tool documentation with parameter descriptions and
   example responses.
4. Verify clean `pipx install .` and `aprilcam mcp` startup.
5. Profile and tune the detection loop and ring buffer for performance.
6. Remove deprecated CLI-only code paths superseded by the MCP server.
7. Update pyproject.toml metadata for PyPI readiness.
8. Add a README with installation and usage instructions.

## Problem

AprilCam has working functionality but lacks the hardening needed for
reliable day-to-day use by AI agents:

- **No tests.** There is zero test coverage. Regressions go undetected.
- **Fragile error paths.** Missing cameras, failed detections, invalid
  parameters, and mid-stream disconnections can produce unhandled
  exceptions or cryptic error messages instead of structured MCP errors.
- **Undocumented tools.** MCP tool descriptions are minimal. Agents
  calling the tools have to guess parameter semantics and response
  shapes.
- **Packaging gaps.** pyproject.toml still has placeholder metadata
  (`"Your Name"` as author), multiple legacy entry points (`taggen`,
  `arucogen`, `homocal`, `cameras`, `playfield`, `atscreencap`,
  `aprilcap`, `apriltest`) that should be consolidated under the
  `aprilcam` CLI, and no README.
- **Potential performance issues.** The detection loop runs synchronously
  and the ring buffer has not been profiled for memory pressure under
  sustained use.
- **Dead code.** Several CLI modules (`atscreencap_cli`, `apriltest_cli`,
  `aprilcap_cli`) date from the pre-MCP architecture and are no longer
  the intended usage path.

## Solution

A systematic polish pass organized into eight work areas:

1. **Integration tests** — pytest-based tests using mocked cameras
   (synthetic frames) to exercise MCP server startup, tool dispatch,
   camera lifecycle, playfield creation, detection loop start/stop, and
   ring buffer queries. No real camera hardware required.
2. **Error handling audit** — review every MCP tool handler for
   unguarded exceptions. Wrap hardware interactions (camera open, frame
   grab) with try/except and return structured MCP error responses.
   Handle camera disconnection mid-detection-loop gracefully (stop loop,
   report error, allow restart).
3. **Tool documentation** — add descriptive docstrings and parameter
   annotations to every MCP tool. Include example request/response
   JSON in a `docs/mcp-tools.md` reference.
4. **pipx install verification** — fix pyproject.toml entry points to
   consolidate under `aprilcam`. Verify `pipx install .` in a clean
   venv. Verify `aprilcam mcp` launches the stdio MCP server.
5. **Performance tuning** — profile the detection loop with cProfile.
   Ensure frame capture runs in a background thread so MCP tool
   responses are not blocked. Cap ring buffer memory with configurable
   max entries.
6. **Deprecated code removal** — remove or archive CLI modules that
   duplicate MCP functionality. Keep `taggen` and `arucogen` as
   subcommands of `aprilcam`; remove standalone entry points.
9. **Tag generation PDF output** — update `taggen` and `arucogen` to
   produce multi-page PDF output by default. Each page is 59mm × 102mm
   (label/card size) with one tag per page, centered with quiet zone.
   Individual PNG output available via `--png` flag. Requires adding a
   PDF library dependency (e.g., `reportlab` or `fpdf2`).
7. **pyproject.toml metadata** — update description, author, keywords,
   classifiers, and project URLs for PyPI.
8. **README** — installation instructions (`pipx install aprilcam`),
   MCP server usage, CLI subcommand reference, quick-start example.

## Success Criteria

- `pytest` passes with at least 80% line coverage on the `aprilcam`
  package (excluding vendored/generated code).
- Every MCP tool returns a structured error (not an unhandled exception)
  when given invalid inputs or when hardware is unavailable.
- Every MCP tool has a docstring with parameter descriptions and at
  least one example response shape documented.
- `pipx install .` succeeds in a fresh venv; `aprilcam mcp` starts and
  responds to an MCP `initialize` request.
- The detection loop does not block MCP tool responses for more than
  50ms (measured via profiling).
- No deprecated standalone entry points remain in pyproject.toml (only
  `aprilcam`).
- pyproject.toml has real author, description, classifiers, and URLs.
- README.md exists and covers installation, MCP usage, and CLI usage.

## Scope

### In Scope

- Integration tests with mocked cameras (no real hardware in CI).
- Error handling improvements in all MCP tool handlers.
- MCP tool documentation (docstrings + reference doc).
- pyproject.toml cleanup: consolidate entry points, update metadata.
- pipx install verification script or test.
- Detection loop performance profiling and optimization.
- Ring buffer memory cap configuration.
- Removal of deprecated CLI entry points and dead modules.
- Tag generation multi-page PDF output (59mm × 102mm page size, one tag
  per page) for `taggen` and `arucogen` subcommands.
- README with installation and usage instructions.

### Out of Scope

- Streamable HTTP transport (deferred to a future sprint).
- New MCP tools or new detection capabilities.
- Web UI or dashboard.
- Real-camera hardware tests in CI (manual verification only).
- PyPI publishing (this sprint prepares for it but does not publish).
- Playfield simulator changes.

## Test Strategy

**Unit tests:**
- Test individual utility functions (tag ID parsing, homography math,
  ring buffer insert/query, velocity computation).
- Test error paths with invalid inputs (bad camera index, malformed
  parameters).

**Integration tests:**
- Mock `cv2.VideoCapture` to return captured test images from
  `tests/data/` (`playfield_cam3.jpg`, `playfield_cam3_moved.jpg`)
  so all tests run without camera hardware.
- Test MCP server startup via subprocess, send `initialize` +
  `tools/list` + tool calls over stdio, verify responses.
- Test detection loop lifecycle: start, query, stop, query-after-stop
  using the mock camera with real playfield images.
- Test playfield creation with real captured images containing ArUco
  corner markers.
- Test camera disconnection mid-loop (mock returns None frame).

**Packaging tests:**
- Script that runs `pipx install .` in an isolated environment and
  verifies `aprilcam mcp` starts.

**Performance tests:**
- cProfile the detection loop with 1000 synthetic frames. Assert
  per-frame latency is under target threshold.
- Measure ring buffer memory usage at max capacity.

## Architecture Notes

This sprint does not add new modules or change the core architecture.
Key design decisions:

- **Test infrastructure:** Tests go in `tests/` at the project root.
  Use pytest with `conftest.py` providing camera mocks and synthetic
  frame fixtures.
- **Error handling pattern:** All MCP tool handlers catch exceptions
  and return `{"error": {"code": ..., "message": ...}}` per MCP spec.
  A decorator or context manager standardizes this.
- **Entry point consolidation:** The single `aprilcam` entry point
  dispatches to subcommands: `mcp`, `taggen`, `arucogen`, `cameras`.
  Legacy standalone entry points (`taggen`, `arucogen`, `homocal`,
  `cameras`, `playfield`, `atscreencap`, `aprilcap`, `apriltest`) are
  removed from `[project.scripts]`.
- **Ring buffer memory cap:** Add a `max_bytes` parameter to the ring
  buffer. When exceeded, oldest entries are dropped regardless of
  frame count.

## GitHub Issues

None linked.

## Definition of Ready

Before tickets can be created, all of the following must be true:

- [ ] Sprint planning documents are complete (sprint.md, use cases, architecture)
- [ ] Architecture review passed
- [ ] Stakeholder has approved the sprint plan

## Tickets

(To be created after sprint approval.)
