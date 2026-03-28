---
id: '006'
title: Live web UI with content-negotiated root route
status: done
use-cases: []
depends-on: []
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Live web UI with content-negotiated root route

## Description

Add a live web UI served from the root route `/` using content negotiation.
When a browser requests HTML (`Accept: text/html`), serve a single-page
JavaScript application that:
- Displays a live camera/playfield image by polling `/api/get_frame`
- Shows real-time tag detections via WebSocket at `/ws/tags/{source_id}`
- Lists detected tags with id, position, and orientation

When the client requests JSON/text (non-HTML), return the existing API
discovery document unchanged.

The UI is a single inline HTML response — no static files, no new
dependencies, no build step.

## Acceptance Criteria

- [x] `GET /` with `Accept: text/html` returns an HTML page with embedded JS
- [x] `GET /` with `Accept: application/json` returns the existing JSON discovery doc
- [x] `GET /` with no Accept header or `*/*` returns the JSON discovery doc (backwards-compatible)
- [x] The HTML UI shows a live image stream from a configurable source
- [x] The HTML UI shows live tag detections via WebSocket
- [x] Existing `/api/*` and `/ws/*` endpoints are unchanged
- [x] All existing tests pass, new tests cover content negotiation

## Testing

- **Existing tests to run**: `tests/test_web_server.py`
- **New tests to write**: Test content negotiation on `/` (HTML vs JSON), test that HTML response contains expected JS/structure
- **Verification command**: `uv run pytest`
