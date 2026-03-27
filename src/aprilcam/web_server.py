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
from starlette.responses import JSONResponse, Response
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
# Endpoint handlers
# ---------------------------------------------------------------------------


async def _discovery(request: Request) -> JSONResponse:
    """GET / — return API discovery document."""
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
