---
id: "005"
title: "CLI subcommand and integration tests"
status: todo
use-cases: [SUC-010-001, SUC-010-002, SUC-010-003, SUC-010-004]
depends-on: ["002", "003", "004"]
github-issue: ""
todo: ""
---

# CLI subcommand and integration tests

## Description

1. Create `src/aprilcam/cli/web_cli.py` with the `aprilcam web`
   subcommand. Options:
   - `--port` (default 17439)
   - `--host` (default 0.0.0.0)
   Starts Uvicorn with single worker, info log level.

2. Register the `web` subcommand in `src/aprilcam/cli/__init__.py`.

3. Add the `web` entry point to `pyproject.toml` if needed.

4. Write integration tests covering the full server lifecycle:
   - Start server, hit `GET /`, verify JSON docs.
   - POST to `/api/list_cameras`, verify response.
   - WebSocket connect, verify message format.

## Acceptance Criteria

- [ ] `aprilcam web` starts the server on the configured port
- [ ] `--port` and `--host` options work
- [ ] Server logs startup info (host, port, available endpoints)
- [ ] Integration tests pass
- [ ] All existing tests still pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Integration tests in `tests/test_web_server.py`
- **Verification command**: `uv run pytest`
