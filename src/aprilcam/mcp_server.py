"""MCP server exposing AprilCam camera tools."""

from __future__ import annotations

import base64
import json
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from aprilcam.homography import (
    CORNER_ID_MAP,
    FieldSpec,
    calibrate_from_corners,
    detect_aruco_4x4,
)
from aprilcam.playfield import Playfield

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
# Playfield registry
# ---------------------------------------------------------------------------


@dataclass
class PlayfieldEntry:
    """A registered playfield backed by a camera."""

    playfield_id: str
    camera_id: str
    playfield: Playfield
    field_spec: Optional[FieldSpec] = None
    homography: Optional[np.ndarray] = None


class PlayfieldRegistry:
    """Manages playfield entries keyed by playfield_id."""

    def __init__(self) -> None:
        self._playfields: dict[str, PlayfieldEntry] = {}

    def register(self, entry: PlayfieldEntry) -> None:
        self._playfields[entry.playfield_id] = entry

    def get(self, playfield_id: str) -> PlayfieldEntry:
        return self._playfields[playfield_id]  # raises KeyError

    def list(self) -> list[str]:
        return list(self._playfields.keys())

    def remove(self, playfield_id: str) -> None:
        del self._playfields[playfield_id]

    def find_by_camera(self, camera_id: str) -> Optional[str]:
        for pid, entry in self._playfields.items():
            if entry.camera_id == camera_id:
                return pid
        return None


# ---------------------------------------------------------------------------
# Module-level instances
# ---------------------------------------------------------------------------

server = FastMCP("aprilcam")
registry = CameraRegistry()
playfield_registry = PlayfieldRegistry()

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


@server.tool()
async def create_playfield(
    camera_id: str,
    max_frames: int = 30,
) -> list[TextContent]:
    """Create a playfield from a camera by detecting ArUco corner markers."""
    # Validate camera exists
    try:
        cap = registry.get(camera_id)
    except KeyError:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"Unknown camera_id '{camera_id}'"}
        ))]

    # Create playfield and try to detect corners
    pf = Playfield(detect_inverted=True)
    for _ in range(max(1, max_frames)):
        ret, frame = cap.read()
        if not ret:
            continue
        pf.update(frame)
        if pf.get_polygon() is not None:
            break

    poly = pf.get_polygon()
    if poly is None:
        # Detect which corners are missing
        import cv2

        ret, frame = cap.read()
        missing = [0, 1, 2, 3]
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detect_aruco_4x4(gray)
            found_ids = [tid for _, tid in dets if tid in (0, 1, 2, 3)]
            missing = [i for i in (0, 1, 2, 3) if i not in found_ids]
        return [TextContent(type="text", text=json.dumps({
            "error": "Failed to detect all 4 corner markers",
            "missing_corner_ids": missing,
        }))]

    # Register the playfield
    playfield_id = f"pf_{camera_id}"

    # Replace existing if same camera
    existing = playfield_registry.find_by_camera(camera_id)
    if existing:
        playfield_registry.remove(existing)

    entry = PlayfieldEntry(
        playfield_id=playfield_id,
        camera_id=camera_id,
        playfield=pf,
    )
    playfield_registry.register(entry)

    corners = poly.tolist()  # UL, UR, LR, LL
    return [TextContent(type="text", text=json.dumps({
        "playfield_id": playfield_id,
        "corners": corners,
        "calibrated": False,
    }))]


@server.tool()
async def calibrate_playfield(
    playfield_id: str,
    width: float,
    height: float,
    units: str = "inch",
) -> list[TextContent]:
    """Calibrate a playfield with real-world measurements."""
    try:
        entry = playfield_registry.get(playfield_id)
    except KeyError:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"Unknown playfield_id '{playfield_id}'"}
        ))]

    poly = entry.playfield.get_polygon()
    if poly is None:
        return [TextContent(type="text", text=json.dumps(
            {"error": "Playfield has no polygon (detection not complete)"}
        ))]

    # Build corner dict from polygon (UL, UR, LR, LL)
    pixel_corners = {
        "upper_left": (float(poly[0][0]), float(poly[0][1])),
        "upper_right": (float(poly[1][0]), float(poly[1][1])),
        "lower_right": (float(poly[2][0]), float(poly[2][1])),
        "lower_left": (float(poly[3][0]), float(poly[3][1])),
    }

    field_spec = FieldSpec(width_in=width, height_in=height, units=units)

    try:
        H, _, _ = calibrate_from_corners(pixel_corners, field_spec)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"Homography computation failed: {exc}"}
        ))]

    # Store calibration in the entry
    entry.field_spec = field_spec
    entry.homography = H

    return [TextContent(type="text", text=json.dumps({
        "playfield_id": playfield_id,
        "calibrated": True,
        "width_cm": field_spec.width_cm,
        "height_cm": field_spec.height_cm,
    }))]


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
