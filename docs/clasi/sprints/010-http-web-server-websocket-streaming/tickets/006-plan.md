# Ticket #006 Plan — Live web UI with content-negotiated root route

## Approach

Modify the `_discovery()` handler in `web_server.py` to check the
`Accept` header. If it contains `text/html`, return an `HTMLResponse`
with an inline single-page application. Otherwise return the existing
JSON discovery document.

## Files to Modify

- `src/aprilcam/web_server.py` — content negotiation + HTML response
- `tests/test_web_server.py` — new tests for content negotiation

## Implementation Details

### Content Negotiation in `_discovery()`

```python
from starlette.responses import HTMLResponse

async def _discovery(request: Request) -> Response:
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(_build_html_ui())
    # existing JSON discovery response...
```

The key rule: `text/html` explicitly requested -> HTML UI. Everything
else (json, */*, missing header) -> JSON discovery doc. This keeps
backwards compatibility.

### HTML UI (`_build_html_ui()`)

A function returning a string of HTML with embedded CSS and JavaScript.
No external dependencies, no static files, no build step.

The UI should:

1. **Source selector** — on load, call `POST /api/list_cameras` to
   populate a dropdown. User picks a source, UI calls
   `POST /api/open_camera`, then `POST /api/start_detection`.

2. **Live image** — poll `POST /api/get_frame` with
   `format=base64` on an interval (e.g. every 500ms). Display in
   an `<img>` tag by setting `src` to a data URL.

3. **Tag overlay** — connect to `ws://{host}/ws/tags/{source_id}`.
   Display tag data in a table below the image: tag ID, center
   position (x, y), orientation. Update on each WebSocket message.

4. **Status bar** — show connection state, frame rate, source info.

### No New Dependencies

- Uses `HTMLResponse` from starlette (already available)
- All JS is vanilla, inline in the HTML string
- No npm, no bundler, no static files directory

### Tests

Add to `tests/test_web_server.py`:

- `test_discovery_html_when_accept_html` — GET `/` with
  `Accept: text/html` returns 200 with `text/html` content type
  and contains expected HTML markers
- `test_discovery_json_when_accept_json` — GET `/` with
  `Accept: application/json` returns JSON (existing behavior)
- `test_discovery_json_when_no_accept` — GET `/` with no Accept
  header returns JSON (backwards compatible)
- `test_discovery_json_when_wildcard` — GET `/` with `Accept: */*`
  returns JSON
