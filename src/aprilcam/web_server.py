"""Starlette web application with REST API endpoints mirroring MCP tools.

Provides a ``create_app()`` factory that returns a Starlette ``Application``
with:

- ``GET /`` — API discovery endpoint returning available tools and usage info.
- ``POST /api/<tool_name>`` — REST endpoints that delegate to the existing
  ``_handle_*`` functions in ``mcp_server.py``.

This module does NOT start a server itself; use ``uvicorn`` or the CLI
subcommand (see ``cli.py``) to run the app.
"""

from __future__ import annotations

import base64
import json
from importlib.metadata import version as _pkg_version
from typing import Any

import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from aprilcam.mcp_server import detection_registry, server as _mcp_server
from aprilcam.mcp_server import (
    _handle_list_cameras,
    _handle_open_camera,
    _handle_close_camera,
    _handle_capture_frame,
    _handle_create_playfield,
    _handle_get_playfield_info,
    _handle_start_detection,
    _handle_stop_detection,
    _handle_get_tags,
    _handle_get_tag_history,
    _handle_get_frame,
    _handle_start_live_view,
    _handle_stop_live_view,
)

# ---------------------------------------------------------------------------
# API endpoint metadata (used for discovery and routing)
# ---------------------------------------------------------------------------

_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_cameras",
        "path": "/api/list_cameras",
        "method": "POST",
        "description": "Enumerate available cameras with names and indices.",
        "parameters": [],
        "returns": "array of {index, name, backend}",
    },
    {
        "name": "open_camera",
        "path": "/api/open_camera",
        "method": "POST",
        "description": "Open a camera by index, name pattern, or source type.",
        "parameters": [
            {"name": "index", "type": "integer", "default": None, "description": "Camera index (0-based)"},
            {"name": "pattern", "type": "string", "default": None, "description": "Substring to match camera name"},
            {"name": "source", "type": "string", "default": None, "description": "Special source type, e.g. 'screen'"},
            {"name": "backend", "type": "string", "default": None, "description": "OpenCV backend constant name"},
        ],
        "returns": "{camera_id} or {error}",
    },
    {
        "name": "close_camera",
        "path": "/api/close_camera",
        "method": "POST",
        "description": "Close an open camera and release its resources.",
        "parameters": [
            {"name": "camera_id", "type": "string", "default": None, "description": "Handle returned by open_camera (required)"},
        ],
        "returns": "{status: 'closed'} or {error}",
    },
    {
        "name": "capture_frame",
        "path": "/api/capture_frame",
        "method": "POST",
        "description": "Capture a single frame from a camera or playfield.",
        "parameters": [
            {"name": "camera_id", "type": "string", "default": None, "description": "Camera or playfield handle (required)"},
            {"name": "format", "type": "string", "default": "base64", "description": "'base64' (JSON) or 'file' (binary JPEG)"},
            {"name": "quality", "type": "integer", "default": 85, "description": "JPEG quality 1-100"},
        ],
        "returns": "JSON with base64 data, or binary JPEG (Content-Type: image/jpeg)",
    },
    {
        "name": "create_playfield",
        "path": "/api/create_playfield",
        "method": "POST",
        "description": "Detect ArUco corner markers and create a playfield for deskewing.",
        "parameters": [
            {"name": "camera_id", "type": "string", "default": None, "description": "Camera handle (required)"},
            {"name": "max_frames", "type": "integer", "default": 30, "description": "Max frames to try for corner detection"},
        ],
        "returns": "{playfield_id, corners, ...} or {error}",
    },
    {
        "name": "get_playfield_info",
        "path": "/api/get_playfield_info",
        "method": "POST",
        "description": "Get information about a registered playfield.",
        "parameters": [
            {"name": "playfield_id", "type": "string", "default": None, "description": "Playfield handle (required)"},
        ],
        "returns": "{playfield_id, camera_id, corners, calibrated, ...} or {error}",
    },
    {
        "name": "start_detection",
        "path": "/api/start_detection",
        "method": "POST",
        "description": "Start a persistent AprilTag/ArUco detection loop on a source.",
        "parameters": [
            {"name": "source_id", "type": "string", "default": None, "description": "Camera or playfield handle (required)"},
            {"name": "family", "type": "string", "default": "36h11", "description": "Tag family: '36h11' or 'aruco4x4'"},
            {"name": "proc_width", "type": "integer", "default": 0, "description": "Processing width (0 = no downscale)"},
            {"name": "detect_interval", "type": "integer", "default": 1, "description": "Detect every N-th frame"},
            {"name": "use_clahe", "type": "boolean", "default": False, "description": "Apply CLAHE contrast enhancement"},
            {"name": "use_sharpen", "type": "boolean", "default": False, "description": "Apply sharpening filter"},
        ],
        "returns": "{source_id, status: 'started'} or {error}",
    },
    {
        "name": "stop_detection",
        "path": "/api/stop_detection",
        "method": "POST",
        "description": "Stop a running detection loop.",
        "parameters": [
            {"name": "source_id", "type": "string", "default": None, "description": "Source handle (required)"},
        ],
        "returns": "{source_id, status: 'stopped'} or {error}",
    },
    {
        "name": "get_tags",
        "path": "/api/get_tags",
        "method": "POST",
        "description": "Get the latest detected tags from a running detection loop.",
        "parameters": [
            {"name": "source_id", "type": "string", "default": None, "description": "Source handle (required)"},
        ],
        "returns": "{source_id, frame, tags: [...]} or {error}",
    },
    {
        "name": "get_tag_history",
        "path": "/api/get_tag_history",
        "method": "POST",
        "description": "Get the last N frames of tag records from the ring buffer.",
        "parameters": [
            {"name": "source_id", "type": "string", "default": None, "description": "Source handle (required)"},
            {"name": "num_frames", "type": "integer", "default": 30, "description": "Number of frames to return"},
        ],
        "returns": "{source_id, frames: [...]} or {error}",
    },
    {
        "name": "get_frame",
        "path": "/api/get_frame",
        "method": "POST",
        "description": "Capture a raw frame from a camera or playfield (no processing).",
        "parameters": [
            {"name": "source_id", "type": "string", "default": None, "description": "Camera or playfield handle (required)"},
            {"name": "format", "type": "string", "default": "base64", "description": "'base64' (JSON) or 'file' (binary JPEG)"},
            {"name": "quality", "type": "integer", "default": 85, "description": "JPEG quality 1-100"},
        ],
        "returns": "JSON with base64 data, or binary JPEG (Content-Type: image/jpeg)",
    },
    {
        "name": "start_live_view",
        "path": "/api/start_live_view",
        "method": "POST",
        "description": "Start a live camera view with tag detection overlay.",
        "parameters": [
            {"name": "camera_id", "type": "string", "default": None, "description": "Camera handle (required)"},
            {"name": "deskew", "type": "boolean", "default": True, "description": "Apply playfield deskew if available"},
            {"name": "family", "type": "string", "default": "36h11", "description": "Tag family: '36h11' or 'aruco4x4'"},
            {"name": "proc_width", "type": "integer", "default": 0, "description": "Processing width (0 = no downscale)"},
            {"name": "use_clahe", "type": "boolean", "default": False, "description": "Apply CLAHE contrast enhancement"},
            {"name": "use_sharpen", "type": "boolean", "default": False, "description": "Apply sharpening filter"},
        ],
        "returns": "{view_id, camera_id, status: 'started'} or {error}",
    },
    {
        "name": "stop_live_view",
        "path": "/api/stop_live_view",
        "method": "POST",
        "description": "Stop a running live view.",
        "parameters": [
            {"name": "view_id", "type": "string", "default": None, "description": "View handle (required)"},
        ],
        "returns": "{view_id, status: 'stopped'} or {error}",
    },
]

# Map tool name -> handler function
_HANDLERS: dict[str, Any] = {
    "list_cameras": _handle_list_cameras,
    "open_camera": _handle_open_camera,
    "close_camera": _handle_close_camera,
    "capture_frame": _handle_capture_frame,
    "create_playfield": _handle_create_playfield,
    "get_playfield_info": _handle_get_playfield_info,
    "start_detection": _handle_start_detection,
    "stop_detection": _handle_stop_detection,
    "get_tags": _handle_get_tags,
    "get_tag_history": _handle_get_tag_history,
    "get_frame": _handle_get_frame,
    "start_live_view": _handle_start_live_view,
    "stop_live_view": _handle_stop_live_view,
}

# Image-returning tools that may produce binary JPEG responses
_IMAGE_TOOLS = {"capture_frame", "get_frame"}


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_response(message: str, status_code: int = 400) -> JSONResponse:
    """Return a JSON error response."""
    return JSONResponse({"error": message}, status_code=status_code)


def _image_or_json(result: dict) -> Response:
    """Convert an image handler result to either binary JPEG or JSON.

    If the result has ``type: "file"``, read the file and return binary
    JPEG with the appropriate Content-Type.  If ``type: "image"``, return
    the base64-encoded data as JSON.  Errors are returned as JSON with
    the appropriate status code.
    """
    rtype = result.get("type")

    if rtype == "error":
        return _error_response(result.get("error", "Unknown error"), 500)

    if rtype == "file":
        path = result["path"]
        try:
            with open(path, "rb") as f:
                data = f.read()
            return Response(content=data, media_type="image/jpeg")
        except Exception as exc:
            return _error_response(f"Failed to read image file: {exc}", 500)

    # Default: base64 JSON
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# HTML UI builder
# ---------------------------------------------------------------------------


def _build_html_ui() -> str:
    """Return a self-contained HTML page with embedded CSS and JavaScript.

    The single-page app provides:
    - A source selector that lists available cameras.
    - A live image stream polled from ``/api/get_frame``.
    - A real-time tag table fed by a WebSocket at ``/ws/tags/{source_id}``.
    - A status bar showing connection state and frame rate.
    """
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AprilCam Live</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; display: flex; flex-direction: column;
         min-height: 100vh; }
  header { background: #16213e; padding: 0.75rem 1.5rem; display: flex;
           align-items: center; justify-content: space-between; }
  header h1 { font-size: 1.25rem; color: #0af; }
  .controls { display: flex; gap: 0.75rem; align-items: center; }
  select, button { padding: 0.4rem 0.75rem; border: 1px solid #334; border-radius: 4px;
                   background: #0d1b2a; color: #e0e0e0; font-size: 0.9rem; cursor: pointer; }
  button:hover { background: #1b2838; }
  button:disabled { opacity: 0.4; cursor: default; }
  main { flex: 1; display: flex; flex-wrap: wrap; padding: 1rem; gap: 1rem; }
  .panel { background: #16213e; border-radius: 8px; padding: 1rem; }
  .video-panel { flex: 2; min-width: 320px; display: flex; flex-direction: column;
                 align-items: center; }
  .video-panel img { max-width: 100%; border-radius: 4px; background: #000; }
  .tags-panel { flex: 1; min-width: 260px; overflow-y: auto; max-height: 80vh; }
  .tags-panel h2 { font-size: 1rem; margin-bottom: 0.5rem; color: #0af; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #223; }
  th { color: #8af; }
  footer { background: #16213e; padding: 0.5rem 1.5rem; font-size: 0.8rem;
           display: flex; gap: 1.5rem; }
  .status-item { display: flex; gap: 0.3rem; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #555; margin-top: 3px; }
  .dot.ok { background: #0f0; }
  .dot.err { background: #f33; }
</style>
</head>
<body>
<header>
  <h1>AprilCam Live</h1>
  <div class="controls">
    <select id="cameraSelect" disabled><option>Loading cameras...</option></select>
    <button id="startBtn" disabled>Start</button>
    <button id="stopBtn" disabled>Stop</button>
  </div>
</header>
<main>
  <div class="panel video-panel">
    <img id="liveImg" alt="No frame" width="640" height="480"
         src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
  </div>
  <div class="panel tags-panel">
    <h2>Detected Tags</h2>
    <table>
      <thead><tr><th>ID</th><th>Center X</th><th>Center Y</th><th>Orientation</th></tr></thead>
      <tbody id="tagBody"><tr><td colspan="4">No data</td></tr></tbody>
    </table>
  </div>
</main>
<footer>
  <div class="status-item"><span class="dot" id="frameDot"></span> Frames:
    <span id="fpsLabel">--</span> fps</div>
  <div class="status-item"><span class="dot" id="wsDot"></span> WebSocket:
    <span id="wsLabel">disconnected</span></div>
  <div class="status-item">Source: <span id="srcLabel">none</span></div>
</footer>
<script>
(function(){
  const cameraSelect = document.getElementById("cameraSelect");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const liveImg = document.getElementById("liveImg");
  const tagBody = document.getElementById("tagBody");
  const frameDot = document.getElementById("frameDot");
  const fpsLabel = document.getElementById("fpsLabel");
  const wsDot = document.getElementById("wsDot");
  const wsLabel = document.getElementById("wsLabel");
  const srcLabel = document.getElementById("srcLabel");

  let cameraId = null;
  let sourceId = null;
  let frameTimer = null;
  let ws = null;
  let frameCount = 0;
  let fpsTimer = null;

  async function api(tool, params) {
    const resp = await fetch("/api/" + tool, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(params || {})
    });
    return resp.json();
  }

  async function loadCameras() {
    try {
      const cams = await api("list_cameras");
      cameraSelect.innerHTML = "";
      if (!cams.length) {
        cameraSelect.innerHTML = "<option>No cameras found</option>";
        return;
      }
      cams.forEach(function(c) {
        const opt = document.createElement("option");
        opt.value = c.index;
        opt.textContent = c.name || ("Camera " + c.index);
        cameraSelect.appendChild(opt);
      });
      cameraSelect.disabled = false;
      startBtn.disabled = false;
    } catch(e) {
      cameraSelect.innerHTML = "<option>Error loading cameras</option>";
    }
  }

  startBtn.addEventListener("click", async function() {
    startBtn.disabled = true;
    cameraSelect.disabled = true;
    const idx = parseInt(cameraSelect.value, 10);
    try {
      const openRes = await api("open_camera", {index: idx});
      if (openRes.error) { alert("Open camera error: " + openRes.error); reset(); return; }
      cameraId = openRes.camera_id;
      sourceId = cameraId;
      srcLabel.textContent = sourceId;

      const detRes = await api("start_detection", {source_id: sourceId});
      if (detRes.error) { alert("Start detection error: " + detRes.error); reset(); return; }

      stopBtn.disabled = false;
      startFramePolling();
      connectWebSocket();
    } catch(e) { alert("Error: " + e.message); reset(); }
  });

  stopBtn.addEventListener("click", async function() {
    await cleanup();
    reset();
  });

  function startFramePolling() {
    frameCount = 0;
    fpsLabel.textContent = "--";
    fpsTimer = setInterval(function() {
      fpsLabel.textContent = (frameCount * 2).toFixed(1);
      frameCount = 0;
    }, 500);

    async function poll() {
      if (!sourceId) return;
      try {
        const res = await api("get_frame", {source_id: sourceId, format: "base64"});
        if (res.data) {
          liveImg.src = "data:image/jpeg;base64," + res.data;
          frameCount++;
          frameDot.className = "dot ok";
        }
      } catch(e) { frameDot.className = "dot err"; }
      if (sourceId) frameTimer = setTimeout(poll, 500);
    }
    poll();
  }

  function connectWebSocket() {
    if (!sourceId) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws/tags/" + sourceId);
    ws.onopen = function() { wsDot.className = "dot ok"; wsLabel.textContent = "connected"; };
    ws.onclose = function() { wsDot.className = "dot"; wsLabel.textContent = "disconnected"; };
    ws.onerror = function() { wsDot.className = "dot err"; wsLabel.textContent = "error"; };
    ws.onmessage = function(evt) {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.error) { tagBody.innerHTML = "<tr><td colspan='4'>" + msg.error + "</td></tr>"; return; }
        const tags = msg.tags || [];
        if (!tags.length) {
          tagBody.innerHTML = "<tr><td colspan='4'>No tags detected</td></tr>";
          return;
        }
        tagBody.innerHTML = tags.map(function(t) {
          const cx = t.center ? t.center[0].toFixed(1) : "--";
          const cy = t.center ? t.center[1].toFixed(1) : "--";
          const ori = t.orientation_deg != null ? t.orientation_deg.toFixed(1) + "\u00b0" : "--";
          return "<tr><td>" + t.tag_id + "</td><td>" + cx + "</td><td>" + cy + "</td><td>" + ori + "</td></tr>";
        }).join("");
      } catch(e) {}
    };
  }

  async function cleanup() {
    if (frameTimer) { clearTimeout(frameTimer); frameTimer = null; }
    if (fpsTimer) { clearInterval(fpsTimer); fpsTimer = null; }
    if (ws) { ws.close(); ws = null; }
    if (sourceId) {
      try { await api("stop_detection", {source_id: sourceId}); } catch(e) {}
    }
    if (cameraId) {
      try { await api("close_camera", {camera_id: cameraId}); } catch(e) {}
    }
    sourceId = null;
    cameraId = null;
  }

  function reset() {
    cameraSelect.disabled = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    frameDot.className = "dot";
    fpsLabel.textContent = "--";
    wsDot.className = "dot";
    wsLabel.textContent = "disconnected";
    srcLabel.textContent = "none";
    tagBody.innerHTML = "<tr><td colspan='4'>No data</td></tr>";
  }

  loadCameras();
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


async def _discovery(request: Request) -> Response:
    """GET / — return API discovery document or HTML UI.

    Content negotiation: if the ``Accept`` header explicitly contains
    ``text/html``, return the single-page web UI.  For everything else
    (``application/json``, ``*/*``, missing header) return the JSON
    discovery document to preserve backwards compatibility.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(_build_html_ui())

    try:
        ver = _pkg_version("aprilcam")
    except Exception:
        ver = "unknown"

    return JSONResponse({
        "server": "aprilcam",
        "version": ver,
        "usage": (
            "POST JSON to /api/<tool_name> with the documented parameters. "
            "Image endpoints return base64 JSON by default; set format='file' "
            "to receive binary JPEG with Content-Type: image/jpeg. "
            "Error responses have {\"error\": \"...\"}. "
            "MCP clients can connect via SSE at /mcp/sse (all 35+ tools available)."
        ),
        "endpoints": _TOOL_SPECS,
    })


async def _dispatch(request: Request) -> Response:
    """POST /api/<tool_name> — dispatch to the corresponding handler."""
    tool_name = request.path_params["tool_name"]

    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return _error_response(f"Unknown tool: '{tool_name}'", 404)

    # Parse JSON body (empty body is fine for no-param tools)
    try:
        body = await request.body()
        params: dict[str, Any] = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _error_response(f"Invalid JSON body: {exc}", 400)

    if not isinstance(params, dict):
        return _error_response("Request body must be a JSON object", 400)

    # Call the handler
    try:
        result = handler(**params)
    except TypeError as exc:
        # Bad parameter names or types
        return _error_response(f"Bad parameters: {exc}", 400)
    except Exception as exc:
        return _error_response(f"Internal error: {exc}", 500)

    # Check for error in result dict
    if isinstance(result, dict) and "error" in result and result.get("type") != "error":
        return _error_response(result["error"], 400)

    # Image-returning tools get special handling
    if tool_name in _IMAGE_TOOLS and isinstance(result, dict):
        return _image_or_json(result)

    # Standard JSON response
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


async def ws_tags(websocket: WebSocket) -> None:
    """Stream tag detections over WebSocket for a given source.

    Connect to ``/ws/tags/{source_id}`` to receive a JSON message per
    frame containing ``source_id``, ``timestamp``, ``frame_index``, and
    ``tags``.  Messages are only sent when a new frame is available
    (based on ``frame_index``).  If no detection loop is running on the
    requested source, the server sends an error message and closes the
    connection.
    """
    source_id: str = websocket.path_params["source_id"]
    await websocket.accept()

    entry = detection_registry.get(source_id)
    if entry is None:
        await websocket.send_json({
            "error": f"No detection loop running on source '{source_id}'"
        })
        await websocket.close(code=1008)
        return

    last_frame_index: int = -1
    try:
        while True:
            # Re-check registry in case detection was stopped while connected
            entry = detection_registry.get(source_id)
            if entry is None:
                await websocket.send_json({
                    "error": f"Detection loop on source '{source_id}' has stopped"
                })
                await websocket.close(code=1001)
                return

            frame = entry.ring_buffer.get_latest()
            if frame is not None and frame.frame_index != last_frame_index:
                last_frame_index = frame.frame_index
                msg = frame.to_dict()
                msg["source_id"] = source_id
                await websocket.send_json(msg)

            await asyncio.sleep(0.033)  # ~30 fps
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Starlette:
    """Create and return the Starlette application.

    Usage::

        app = create_app()
        # Run with: uvicorn aprilcam.web_server:app
    """
    # Build the MCP SSE sub-application.  The FastMCP.sse_app() method
    # returns a standalone Starlette app whose internal routes (``/sse``
    # for the event stream and ``/messages/`` for POSTs) are relative to
    # wherever we mount it.  Mounting at ``/mcp`` makes the full paths
    # ``/mcp/sse`` and ``/mcp/messages/``.
    mcp_sse = _mcp_server.sse_app(mount_path="/mcp")

    routes = [
        Route("/", _discovery, methods=["GET"]),
        Route("/api/{tool_name}", _dispatch, methods=["POST"]),
        WebSocketRoute("/ws/tags/{source_id}", ws_tags),
        Mount("/mcp", app=mcp_sse),
    ]
    return Starlette(routes=routes)
