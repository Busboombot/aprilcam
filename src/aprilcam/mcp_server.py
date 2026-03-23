"""MCP server exposing AprilCam camera tools."""

from __future__ import annotations

import base64
import json
import tempfile
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# ---------------------------------------------------------------------------
# Camera registry
# ---------------------------------------------------------------------------


class CameraRegistry:
    """Manages open camera/capture handles keyed by UUID strings."""

    def __init__(self) -> None:
        self._cameras: dict[str, Any] = {}

    def open(self, capture: Any) -> str:
        """Register *capture* and return a UUID4 handle string."""
        handle = str(uuid.uuid4())
        self._cameras[handle] = capture
        return handle

    def get(self, camera_id: str) -> Any:
        """Return the capture for *camera_id* or raise ``KeyError``."""
        return self._cameras[camera_id]

    def close(self, camera_id: str) -> None:
        """Release and remove the capture identified by *camera_id*.

        Raises ``KeyError`` if *camera_id* is not registered.
        """
        cap = self._cameras.pop(camera_id)  # KeyError if missing
        cap.release()

    def close_all(self) -> None:
        """Release every open capture and clear the registry.

        Individual release errors are swallowed so the rest still get closed.
        """
        for cap in self._cameras.values():
            try:
                cap.release()
            except Exception:
                pass
        self._cameras.clear()

    def list_open(self) -> list[str]:
        """Return a list of currently-active handle strings."""
        return list(self._cameras.keys())

    def __del__(self) -> None:
        self.close_all()


# ---------------------------------------------------------------------------
# Module-level instances
# ---------------------------------------------------------------------------

server = FastMCP("aprilcam")
registry = CameraRegistry()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.tool()
async def list_cameras() -> list[TextContent]:
    """List available cameras (indices 0-9)."""
    from aprilcam.camutil import list_cameras as _list_cameras

    try:
        cams = _list_cameras(max_index=10, quiet=True)
        result = [
            {"index": c.index, "name": c.name, "backend": c.backend}
            for c in cams
        ]
    except Exception:
        result = []  # empty array, not an error
    return [TextContent(type="text", text=json.dumps(result))]


@server.tool()
async def open_camera(
    index: int | None = None,
    pattern: str | None = None,
    source: str | None = None,
    backend: str | None = None,
) -> list[TextContent]:
    """Open a camera and return a handle for subsequent operations."""
    try:
        if source == "screen":
            from aprilcam.screencap import ScreenCaptureMSS

            cap = ScreenCaptureMSS()
        else:
            import cv2

            idx: int | None = None
            if pattern is not None:
                from aprilcam.camutil import (
                    list_cameras as _list_cameras,
                    select_camera_by_pattern,
                )

                cams = _list_cameras(max_index=10, quiet=True)
                idx = select_camera_by_pattern(pattern, cams)
                if idx is None:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"No camera matching pattern '{pattern}'"}
                            ),
                        )
                    ]
            elif index is not None:
                idx = index
            else:
                idx = 0

            if backend is not None:
                be = getattr(cv2, backend, None)
                if be is None:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"Unknown backend '{backend}'"}
                            ),
                        )
                    ]
                cap = cv2.VideoCapture(idx, be)
            else:
                cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                cap.release()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to open camera at index {idx}"}
                        ),
                    )
                ]

        camera_id = registry.open(cap)
        return [
            TextContent(
                type="text", text=json.dumps({"camera_id": camera_id})
            )
        ]
    except Exception as exc:
        return [
            TextContent(type="text", text=json.dumps({"error": str(exc)}))
        ]


@server.tool()
async def capture_frame(
    camera_id: str,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Capture a single frame from an open camera."""
    try:
        cap = registry.get(camera_id)
    except KeyError:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown camera_id '{camera_id}'"}),
            )
        ]

    try:
        import cv2

        ret, frame = cap.read()
        if not ret:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "Failed to read frame"}),
                )
            ]

        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "Failed to encode frame"}),
                )
            ]

        if format == "file":
            tmp = tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            )
            tmp.write(buf.tobytes())
            tmp.close()
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"path": tmp.name}),
                )
            ]

        # default: base64
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return [
            ImageContent(
                type="image",
                data=b64,
                mimeType="image/jpeg",
            )
        ]
    except Exception as exc:
        return [
            TextContent(type="text", text=json.dumps({"error": str(exc)}))
        ]


@server.tool()
async def close_camera(camera_id: str) -> list[TextContent]:
    """Close a previously-opened camera."""
    try:
        registry.close(camera_id)
    except KeyError:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown camera_id '{camera_id}'"}),
            )
        ]
    return [
        TextContent(type="text", text=json.dumps({"status": "closed"}))
    ]


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    try:
        server.run(transport="stdio")
    finally:
        registry.close_all()


if __name__ == "__main__":
    main()
