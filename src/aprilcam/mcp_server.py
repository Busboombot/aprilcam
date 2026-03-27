"""MCP server exposing AprilCam camera, playfield, and image-processing tools.

This module implements the FastMCP server that provides AI agents with
programmatic access to camera management, playfield homography, tag
detection loops, multi-camera compositing, and image processing
operations. It is the primary entry point for the ``aprilcam mcp``
subcommand and the ``aprilcam-mcp`` standalone script.

All ``@server.tool()`` functions follow a consistent error-handling
contract: on success they return structured JSON (or image data), and
on error they return ``{"error": "<message>"}``.
"""

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

from aprilcam.aprilcam import AprilCam
from aprilcam.composite import (
    CompositeManager,
    compute_cross_camera_homography,
    map_tags_to_primary,
)
from aprilcam.detection import DetectionLoop, RingBuffer
from aprilcam.frame import FrameEntry, FrameRegistry
from aprilcam.homography import (
    CORNER_ID_MAP,
    FieldSpec,
    calibrate_from_corners,
    detect_aruco_4x4,
)
from aprilcam.image_processing import (
    process_detect_circles,
    process_detect_contours,
    process_detect_lines,
    process_detect_qr_codes,
)
from aprilcam.models import AprilTag
from aprilcam.playfield import Playfield

# ---------------------------------------------------------------------------
# Camera registry
# ---------------------------------------------------------------------------


class CameraRegistry:
    """Manages open camera/capture handles keyed by deterministic strings."""

    def __init__(self) -> None:
        self._cameras: dict[str, Any] = {}

    def open(self, capture: Any, handle: str | None = None) -> str:
        """Register *capture* and return a handle string.

        If *handle* is provided it is used as-is (deterministic).
        Otherwise a UUID4 is generated.
        """
        if handle is None:
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
composite_manager = CompositeManager()
frame_registry = FrameRegistry()


@dataclass
class DetectionEntry:
    """A running detection loop bound to a source (camera or playfield)."""

    source_id: str
    loop: DetectionLoop
    ring_buffer: RingBuffer
    aprilcam: AprilCam
    operations: list[str] = field(default_factory=lambda: ["detect_tags"])


detection_registry: dict[str, DetectionEntry] = {}


@dataclass
class LiveViewEntry:
    """A running live-view subprocess with its ring buffer."""

    source_id: str
    process: Any  # LiveViewProcess
    ring_buffer: RingBuffer


live_view_registry: dict[str, LiveViewEntry] = {}


# ---------------------------------------------------------------------------
# Source resolution & image output helpers
# ---------------------------------------------------------------------------


def resolve_source(source_id: str) -> np.ndarray:
    """Resolve a source_id (playfield or camera) to a captured frame.

    If *source_id* names a playfield, the frame is deskewed automatically.

    Raises:
        KeyError: if *source_id* is not found in either registry.
        RuntimeError: if the underlying capture fails to read a frame.
    """
    # Try playfield first
    try:
        pf_entry = playfield_registry.get(source_id)
        cap = registry.get(pf_entry.camera_id)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")
        return pf_entry.playfield.deskew(frame)
    except KeyError:
        pass

    # Try camera
    try:
        cap = registry.get(source_id)
    except KeyError:
        raise KeyError(f"Unknown source_id '{source_id}'")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame")
    return frame


def format_image_output(
    frame: np.ndarray,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Encode *frame* as JPEG and return MCP content items.

    Args:
        frame: BGR image as a NumPy array.
        format: ``"base64"`` (default) returns an ``ImageContent`` with
            inline data; ``"file"`` writes a temp file and returns a
            ``TextContent`` with the path.
        quality: JPEG quality (0-100).
    """
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    if format == "file":
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(buf.tobytes())
        tmp.close()
        return [TextContent(type="text", text=json.dumps({"path": tmp.name}))]

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return [ImageContent(type="image", data=b64, mimeType="image/jpeg")]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.tool()
async def list_cameras() -> list[TextContent]:
    """List available cameras by probing indices 0 through 9.

    Returns:
        A JSON array of camera objects, each with ``index`` (int),
        ``name`` (str), and ``backend`` (str). Returns an empty
        array if no cameras are found or an error occurs.
    """
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
    """Open a camera by index, name pattern, or screen capture and return a UUID handle.

    Args:
        index: Camera device index (default 0 if nothing else is specified).
        pattern: Substring to match against camera names (e.g. ``"FaceTime"``).
        source: Set to ``"screen"`` to capture the desktop instead of a camera.
        backend: OpenCV backend constant name (e.g. ``"CAP_AVFOUNDATION"``).

    Returns:
        On success: ``{"camera_id": "<uuid>"}``.
        On error: ``{"error": "<message>"}``.
    """
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

            # USB cameras need several reads before producing valid frames
            import time

            for _ in range(10):
                ret, _ = cap.read()
                if ret:
                    break
                time.sleep(0.1)

        # Deterministic handle: cam_N for indexed cameras, screen for screen capture
        if source == "screen":
            handle = "screen"
        else:
            handle = f"cam_{idx}"
        # If this handle is already open, close the old one first
        if handle in registry._cameras:
            try:
                registry.close(handle)
            except Exception:
                pass
        camera_id = registry.open(cap, handle=handle)
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
    """Capture a single frame from an open camera or playfield.

    If *camera_id* refers to a playfield, the frame is automatically deskewed.

    Args:
        camera_id: UUID handle from ``open_camera`` or a playfield_id.
        format: ``"base64"`` (default) returns inline image data;
            ``"file"`` writes a JPEG to a temp file and returns its path.
        quality: JPEG encoding quality (0-100, default 85).

    Returns:
        On success (base64): an ``ImageContent`` with inline JPEG data.
        On success (file): ``{"path": "<temp_file_path>"}``.
        On error: ``{"error": "<message>"}``.
    """
    # Check if this is a playfield ID first
    pf_entry = None
    try:
        pf_entry = playfield_registry.get(camera_id)
        # Resolve to underlying camera
        try:
            cap = registry.get(pf_entry.camera_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Underlying camera '{pf_entry.camera_id}' is no longer open"}
            ))]
    except KeyError:
        # Not a playfield, try camera registry
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

        import time

        ret, frame = None, None
        for _attempt in range(5):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.1)
        if not ret:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "Failed to read frame"}),
                )
            ]

        # Apply deskew if this is a playfield capture
        if pf_entry is not None:
            frame = pf_entry.playfield.deskew(frame)

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
    """Close a previously-opened camera and release its resources.

    Args:
        camera_id: The UUID handle returned by ``open_camera``.

    Returns:
        On success: ``{"status": "closed"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        registry.close(camera_id)
    except KeyError:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown camera_id '{camera_id}'"}),
            )
        ]
    except Exception as exc:
        return [
            TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))
        ]
    return [
        TextContent(type="text", text=json.dumps({"status": "closed"}))
    ]


@server.tool()
async def create_playfield(
    camera_id: str,
    max_frames: int = 30,
) -> list[TextContent]:
    """Create a playfield from a camera by detecting ArUco corner markers.

    Reads up to *max_frames* frames from the camera, looking for four
    ArUco 4x4 corner markers (IDs 0-3). Once all four are found, the
    playfield polygon is established and a playfield_id is returned.

    Args:
        camera_id: UUID handle from ``open_camera``.
        max_frames: Maximum number of frames to read while searching
            for corner markers (default 30).

    Returns:
        On success: ``{"playfield_id": "<id>", "corners": [[x,y],...], "calibrated": false}``.
        On partial detection: ``{"error": "...", "missing_corner_ids": [...]}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        # Validate camera exists
        try:
            cap = registry.get(camera_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown camera_id '{camera_id}'"}
            ))]

        # Create playfield and try to detect corners (proc_width=0 disables downscale)
        pf = Playfield(detect_inverted=True, proc_width=0)
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
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def calibrate_playfield(
    playfield_id: str,
    width: float,
    height: float,
    units: str = "inch",
) -> list[TextContent]:
    """Calibrate a playfield with real-world measurements to enable pixel-to-world mapping.

    Uses the detected corner markers to compute a homography that maps
    pixel coordinates to real-world coordinates in the specified units.

    Args:
        playfield_id: The playfield handle from ``create_playfield``.
        width: Real-world width of the playfield between corner markers.
        height: Real-world height of the playfield between corner markers.
        units: Measurement units — ``"inch"`` (default) or ``"cm"``.

    Returns:
        On success: ``{"playfield_id": "<id>", "calibrated": true, "width_cm": ..., "height_cm": ...}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
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
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def create_playfield_from_image(
    image_path: str,
) -> list[TextContent]:
    """Create a playfield from a static image file by detecting ArUco corner markers.

    Reads the image from disk and attempts to detect four ArUco 4x4
    corner markers (IDs 0-3). Useful for testing or working with
    pre-captured images rather than live cameras.

    Args:
        image_path: Absolute path to an image file readable by OpenCV.

    Returns:
        On success: ``{"playfield_id": "<id>", "corners": [[x,y],...], "calibrated": false}``.
        On partial detection: ``{"error": "...", "missing_corner_ids": [...]}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Failed to read image file '{image_path}'"}
            ))]

        pf = Playfield(detect_inverted=True, proc_width=0)
        pf.update(img)

        poly = pf.get_polygon()
        if poly is None:
            # Detect which corners are missing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detect_aruco_4x4(gray)
            found_ids = [tid for _, tid in dets if tid in (0, 1, 2, 3)]
            missing = [i for i in (0, 1, 2, 3) if i not in found_ids]
            return [TextContent(type="text", text=json.dumps({
                "error": "Failed to detect all 4 corner markers",
                "missing_corner_ids": missing,
            }))]

        playfield_id = f"pf_{uuid.uuid4().hex[:8]}"
        camera_id = f"file:{image_path}"

        entry = PlayfieldEntry(
            playfield_id=playfield_id,
            camera_id=camera_id,
            playfield=pf,
        )
        playfield_registry.register(entry)

        corners = poly.tolist()
        return [TextContent(type="text", text=json.dumps({
            "playfield_id": playfield_id,
            "corners": corners,
            "calibrated": False,
        }))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def deskew_image(
    playfield_id: str,
    image_path: str,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Read a static image and apply a playfield's deskew (perspective warp) transform.

    Warps the image to a top-down view using the homography derived
    from the playfield's detected corner markers.

    Args:
        playfield_id: The playfield handle from ``create_playfield`` or
            ``create_playfield_from_image``.
        image_path: Absolute path to an image file readable by OpenCV.
        format: ``"base64"`` (default) or ``"file"``.
        quality: JPEG encoding quality (0-100, default 85).

    Returns:
        On success (base64): an ``ImageContent`` with inline JPEG data.
        On success (file): ``{"path": "<temp_file_path>"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            entry = playfield_registry.get(playfield_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown playfield_id '{playfield_id}'"}
            ))]

        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Failed to read image file '{image_path}'"}
            ))]

        deskewed = entry.playfield.deskew(img)

        ok, buf = cv2.imencode(
            ".jpg", deskewed, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            return [TextContent(type="text", text=json.dumps(
                {"error": "Failed to encode deskewed image"}
            ))]

        if format == "file":
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(buf.tobytes())
            tmp.close()
            return [TextContent(type="text", text=json.dumps({"path": tmp.name}))]

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return [ImageContent(type="image", data=b64, mimeType="image/jpeg")]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def get_playfield_info(
    playfield_id: str,
) -> list[TextContent]:
    """Return the current state of a registered playfield.

    Args:
        playfield_id: The playfield handle from ``create_playfield`` or
            ``create_playfield_from_image``.

    Returns:
        On success: ``{"playfield_id": ..., "camera_id": ..., "corners": ...,
        "calibrated": bool}``. If calibrated, also includes ``width_cm``,
        ``height_cm``, and ``homography`` (3x3 matrix as nested list).
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            entry = playfield_registry.get(playfield_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown playfield_id '{playfield_id}'"}
            ))]

        poly = entry.playfield.get_polygon()
        calibrated = entry.homography is not None

        result: dict = {
            "playfield_id": entry.playfield_id,
            "camera_id": entry.camera_id,
            "corners": poly.tolist() if poly is not None else None,
            "calibrated": calibrated,
        }

        if calibrated and entry.field_spec is not None:
            result["width_cm"] = entry.field_spec.width_cm
            result["height_cm"] = entry.field_spec.height_cm
            result["homography"] = entry.homography.tolist()

        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


# ---------------------------------------------------------------------------
# Composite tools
# ---------------------------------------------------------------------------


@server.tool()
async def create_composite(
    primary_camera_id: str,
    secondary_camera_id: str,
    playfield_id: str = "",
    correspondence_points: str = "",
) -> list[TextContent]:
    """Create a multi-camera composite by computing cross-camera homography.

    Maps tag detections from a secondary camera into the primary camera's
    coordinate system. If *correspondence_points* is empty, auto-detects
    shared ArUco markers between both cameras.

    Args:
        primary_camera_id: UUID handle of the primary (color) camera.
        secondary_camera_id: UUID handle of the secondary (e.g. B&W) camera.
        playfield_id: Optional playfield handle for world-coordinate mapping.
        correspondence_points: JSON string of point pairs
            ``[[px1,py1,sx1,sy1], ...]`` (primary x,y then secondary x,y).
            If empty, shared ArUco markers are auto-detected.

    Returns:
        On success: ``{"composite_id": "<id>", "reprojection_error": ...,
        "num_correspondences": ...}``.
        On error: ``{"error": "<message>"}``.
    """
    import cv2

    try:
        if correspondence_points and correspondence_points.strip():
            # Manual correspondence mode
            pairs = json.loads(correspondence_points)
            if not isinstance(pairs, list) or len(pairs) < 4:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "Need at least 4 correspondence point pairs"}
                ))]
            primary_pts = np.array([[p[0], p[1]] for p in pairs], dtype=np.float64)
            secondary_pts = np.array([[p[2], p[3]] for p in pairs], dtype=np.float64)
        else:
            # Auto-detect shared ArUco markers
            try:
                cap_pri = registry.get(primary_camera_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Unknown primary camera_id '{primary_camera_id}'"}
                ))]
            try:
                cap_sec = registry.get(secondary_camera_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Unknown secondary camera_id '{secondary_camera_id}'"}
                ))]

            ret1, frame1 = cap_pri.read()
            if not ret1:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "Failed to read frame from primary camera"}
                ))]
            ret2, frame2 = cap_sec.read()
            if not ret2:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "Failed to read frame from secondary camera"}
                ))]

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            dets1 = detect_aruco_4x4(gray1)
            dets2 = detect_aruco_4x4(gray2)

            # Build id->center maps
            map1 = {tid: pts.mean(axis=0) for pts, tid in dets1}
            map2 = {tid: pts.mean(axis=0) for pts, tid in dets2}

            shared_ids = sorted(set(map1.keys()) & set(map2.keys()))
            if len(shared_ids) < 4:
                return [TextContent(type="text", text=json.dumps({
                    "error": "Not enough shared markers for homography",
                    "shared_ids": shared_ids,
                    "primary_ids": sorted(map1.keys()),
                    "secondary_ids": sorted(map2.keys()),
                }))]

            primary_pts = np.array([map1[sid].tolist() for sid in shared_ids], dtype=np.float64)
            secondary_pts = np.array([map2[sid].tolist() for sid in shared_ids], dtype=np.float64)

        H, rms_error = compute_cross_camera_homography(primary_pts, secondary_pts)

        comp = composite_manager.create(
            primary_camera_id=primary_camera_id,
            secondary_camera_id=secondary_camera_id,
            homography=H,
            reprojection_error=rms_error,
            playfield_id=playfield_id if playfield_id else None,
        )

        return [TextContent(type="text", text=json.dumps({
            "composite_id": comp.composite_id,
            "reprojection_error": rms_error,
            "num_correspondences": len(primary_pts),
        }))]
    except (ValueError, json.JSONDecodeError) as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


def _detect_apriltags_on_frame(frame: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Detect AprilTag 36h11 markers on a BGR frame.

    Returns a list of (corners_4x2, raw_corners, tag_id) tuples suitable
    for passing to ``map_tags_to_primary``.
    """
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    results: list[tuple[np.ndarray, np.ndarray, int]] = []
    if ids is not None and len(ids) > 0:
        for c, tid in zip(corners, ids.flatten()):
            pts = np.array(c, dtype=np.float32).reshape(-1, 2)
            results.append((pts, c, int(tid)))
    return results


def render_tag_overlay(frame: np.ndarray, mapped_tags: list[dict]) -> np.ndarray:
    """Draw tag overlays (polygon + ID label) onto a frame copy.

    Args:
        frame: BGR image (will be copied, not modified in place).
        mapped_tags: list of dicts with ``corners_px`` and ``id`` keys.

    Returns:
        Annotated BGR image.
    """
    import cv2

    out = frame.copy()
    for tag in mapped_tags:
        corners = np.array(tag["corners_px"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = int(tag["center_px"][0]), int(tag["center_px"][1])
        cv2.putText(
            out, str(tag["id"]),
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return out


@server.tool()
async def get_composite_frame(
    composite_id: str,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Capture the primary camera frame with secondary-camera tag detections overlaid.

    Reads frames from both cameras, detects AprilTags on the secondary
    frame, maps their positions into the primary camera's coordinate
    system, and draws tag overlays on the primary frame.

    Args:
        composite_id: The composite handle from ``create_composite``.
        format: ``"base64"`` (default) or ``"file"``.
        quality: JPEG encoding quality (0-100, default 85).

    Returns:
        On success (base64): an ``ImageContent`` with the annotated JPEG.
        On success (file): ``{"path": "<temp_file_path>"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            comp = composite_manager.get(composite_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown composite_id '{composite_id}'"}
            ))]

        try:
            cap_pri = registry.get(comp.primary_camera_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Primary camera '{comp.primary_camera_id}' is no longer open"}
            ))]

        try:
            cap_sec = registry.get(comp.secondary_camera_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Secondary camera '{comp.secondary_camera_id}' is no longer open"}
            ))]

        ret1, frame_pri = cap_pri.read()
        if not ret1:
            return [TextContent(type="text", text=json.dumps(
                {"error": "Failed to read frame from primary camera"}
            ))]

        ret2, frame_sec = cap_sec.read()
        if not ret2:
            return [TextContent(type="text", text=json.dumps(
                {"error": "Failed to read frame from secondary camera"}
            ))]

        # Detect tags on secondary frame
        detections = _detect_apriltags_on_frame(frame_sec)
        mapped = map_tags_to_primary(detections, comp.homography)

        # Overlay on primary frame
        annotated = render_tag_overlay(frame_pri, mapped)

        return format_image_output(annotated, format, quality)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def get_composite_tags(
    composite_id: str,
) -> list[TextContent]:
    """Detect tags on the secondary camera and return positions in primary camera coordinates.

    If the composite has an associated calibrated playfield, each tag
    also includes ``world_xy`` coordinates.

    Args:
        composite_id: The composite handle from ``create_composite``.

    Returns:
        On success: ``{"composite_id": "<id>", "tags": [...]}``. Each tag
        has ``id``, ``center_px``, ``corners_px``, and optionally ``world_xy``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            comp = composite_manager.get(composite_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown composite_id '{composite_id}'"}
            ))]

        try:
            cap_sec = registry.get(comp.secondary_camera_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Secondary camera '{comp.secondary_camera_id}' is no longer open"}
            ))]

        ret, frame_sec = cap_sec.read()
        if not ret:
            return [TextContent(type="text", text=json.dumps(
                {"error": "Failed to read frame from secondary camera"}
            ))]

        detections = _detect_apriltags_on_frame(frame_sec)
        mapped = map_tags_to_primary(detections, comp.homography)

        # Add world_xy if composite has a calibrated playfield
        if comp.playfield_id:
            try:
                pf_entry = playfield_registry.get(comp.playfield_id)
                if pf_entry.homography is not None:
                    for tag in mapped:
                        cx, cy = tag["center_px"]
                        vec = np.array([cx, cy, 1.0], dtype=np.float64)
                        Xw = pf_entry.homography @ vec
                        if abs(Xw[2]) > 1e-9:
                            tag["world_xy"] = [float(Xw[0] / Xw[2]), float(Xw[1] / Xw[2])]
            except KeyError:
                pass  # playfield not found, skip world coords

        return [TextContent(type="text", text=json.dumps({
            "composite_id": composite_id,
            "tags": mapped,
        }))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


# ---------------------------------------------------------------------------
# Detection tools
# ---------------------------------------------------------------------------


@server.tool()
async def start_detection(
    source_id: str,
    family: str = "36h11",
    proc_width: int = 0,
    detect_interval: int = 1,
    use_clahe: bool = False,
    use_sharpen: bool = False,
) -> list[TextContent]:
    """Start a persistent tag detection loop on a camera or playfield.

    The loop captures frames continuously, detects AprilTag/ArUco markers
    on each frame, and stores results in a 300-frame ring buffer
    (~10 seconds at 30 fps).

    Args:
        source_id: A camera UUID or playfield_id to detect on.
        family: AprilTag family (default ``"36h11"``).
        proc_width: Processing width in pixels; 0 means no downscale.
        detect_interval: Run detection every N frames (default 1).
        use_clahe: Apply CLAHE contrast enhancement before detection.
        use_sharpen: Apply sharpening before detection.

    Returns:
        On success: ``{"source_id": "<id>", "status": "started"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        if source_id in detection_registry:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Detection already running on '{source_id}'"}
            ))]

        import cv2

        # Resolve source to a capture object and optional playfield data
        cap = None
        homography = None
        playfield_poly = None
        camera_id: str | None = None  # the registry handle to re-open on stop
        camera_index: int | None = None  # real device index (if applicable)
        exclusive_cap = None  # set when we open our own exclusive camera

        try:
            pf_entry = playfield_registry.get(source_id)
            camera_id = pf_entry.camera_id
            try:
                cap = registry.get(camera_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Underlying camera '{camera_id}' is no longer open"}
                ))]
            homography = pf_entry.homography
            poly = pf_entry.playfield.get_polygon()
            if poly is not None:
                playfield_poly = poly
        except KeyError:
            # Not a playfield — treat source_id as a camera handle
            camera_id = source_id
            try:
                cap = registry.get(source_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Unknown source_id '{source_id}'"}
                ))]

        # For real cameras (cam_N handles), open an exclusive capture to
        # avoid frame contention with other MCP tools reading the same handle.
        if camera_id and camera_id.startswith("cam_"):
            try:
                camera_index = int(camera_id.split("_", 1)[1])
            except (ValueError, IndexError):
                camera_index = None

            if camera_index is not None:
                # Release the shared camera so the detection loop gets exclusive access
                try:
                    registry.close(camera_id)
                except KeyError:
                    pass

                exclusive_cap = cv2.VideoCapture(camera_index)
                if exclusive_cap.isOpened():
                    cap = exclusive_cap
                else:
                    # Re-open shared camera on failure
                    exclusive_cap = None
                    try:
                        shared_cap = cv2.VideoCapture(camera_index)
                        if shared_cap.isOpened():
                            registry.open(shared_cap, handle=camera_id)
                            cap = registry.get(camera_id)
                    except Exception:
                        pass

        cam = AprilCam(
            index=camera_index if camera_index is not None else 0,
            backend=None,
            speed_alpha=0.3,
            family=family,
            proc_width=proc_width,
            detect_interval=detect_interval,
            use_clahe=use_clahe,
            use_sharpen=use_sharpen,
            headless=True,
            cap=cv2.VideoCapture(),
            homography=homography,
            playfield_poly_init=playfield_poly,
        )

        buf = RingBuffer(maxlen=300)
        loop = DetectionLoop(source=cap, aprilcam=cam, ring_buffer=buf)
        loop.start()

        detection_registry[source_id] = DetectionEntry(
            source_id=source_id,
            loop=loop,
            ring_buffer=buf,
            aprilcam=cam,
        )
        # Remember state so stop_detection can re-open the shared camera
        detection_registry[source_id]._camera_id = camera_id  # type: ignore[attr-defined]
        detection_registry[source_id]._camera_index = camera_index  # type: ignore[attr-defined]
        detection_registry[source_id]._exclusive_cap = exclusive_cap  # type: ignore[attr-defined]

        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "status": "started"}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def stop_detection(source_id: str) -> list[TextContent]:
    """Stop a running tag detection loop and discard its ring buffer.

    Args:
        source_id: The camera UUID or playfield_id passed to ``start_detection``.

    Returns:
        On success: ``{"source_id": "<id>", "status": "stopped"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        entry = detection_registry.pop(source_id, None)
        if entry is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"No detection running on '{source_id}'"}
            ))]

        entry.loop.stop()

        # Release the exclusive capture and re-open the shared camera
        exclusive_cap = getattr(entry, "_exclusive_cap", None)
        if exclusive_cap is not None:
            try:
                exclusive_cap.release()
            except Exception:
                pass
        camera_id = getattr(entry, "_camera_id", None)
        camera_index = getattr(entry, "_camera_index", 0)
        if camera_id is not None:
            try:
                import cv2
                shared_cap = cv2.VideoCapture(camera_index)
                if shared_cap.isOpened():
                    registry.open(shared_cap, handle=camera_id)
            except Exception:
                pass

        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "status": "stopped"}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def stream_tags(
    source_id: str,
    operations: list[str] | None = None,
    family: str = "36h11",
    proc_width: int = 0,
) -> list[TextContent]:
    """Start continuous tag detection on a camera or playfield with a fixed operation pipeline.

    This is the preferred entry point for starting a detection stream.  It
    wraps the same infrastructure as ``start_detection`` (AprilCam +
    DetectionLoop + RingBuffer) while recording an explicit *operations*
    pipeline for metadata.

    Args:
        source_id: A camera handle (``cam_N``) or playfield_id to stream from.
        operations: Operation pipeline names to apply each frame.  Stored as
            metadata; the underlying loop currently uses ``AprilCam.process_frame()``.
            Defaults to ``["detect_tags"]``.
        family: AprilTag family (default ``"36h11"``).
        proc_width: Processing width in pixels; 0 means no downscale.

    Returns:
        On success: ``{"stream_id": "<id>", "operations": [...], "status": "started"}``.
        On error: ``{"error": "<message>"}``.
    """
    if operations is None:
        operations = ["detect_tags"]

    try:
        if source_id in detection_registry:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Detection already running on '{source_id}'"}
            ))]

        import cv2

        # Resolve source to a capture object and optional playfield data
        cap = None
        homography = None
        playfield_poly = None
        camera_id: str | None = None
        camera_index: int | None = None
        exclusive_cap = None

        try:
            pf_entry = playfield_registry.get(source_id)
            camera_id = pf_entry.camera_id
            try:
                cap = registry.get(camera_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Underlying camera '{camera_id}' is no longer open"}
                ))]
            homography = pf_entry.homography
            poly = pf_entry.playfield.get_polygon()
            if poly is not None:
                playfield_poly = poly
        except KeyError:
            # Not a playfield — treat source_id as a camera handle
            camera_id = source_id
            try:
                cap = registry.get(source_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Unknown source_id '{source_id}'"}
                ))]

        # For real cameras (cam_N handles), open an exclusive capture
        if camera_id and camera_id.startswith("cam_"):
            try:
                camera_index = int(camera_id.split("_", 1)[1])
            except (ValueError, IndexError):
                camera_index = None

            if camera_index is not None:
                try:
                    registry.close(camera_id)
                except KeyError:
                    pass

                exclusive_cap = cv2.VideoCapture(camera_index)
                if exclusive_cap.isOpened():
                    cap = exclusive_cap
                else:
                    exclusive_cap = None
                    try:
                        shared_cap = cv2.VideoCapture(camera_index)
                        if shared_cap.isOpened():
                            registry.open(shared_cap, handle=camera_id)
                            cap = registry.get(camera_id)
                    except Exception:
                        pass

        cam = AprilCam(
            index=camera_index if camera_index is not None else 0,
            backend=None,
            speed_alpha=0.3,
            family=family,
            proc_width=proc_width,
            detect_interval=1,
            use_clahe=False,
            use_sharpen=False,
            headless=True,
            cap=cv2.VideoCapture(),
            homography=homography,
            playfield_poly_init=playfield_poly,
        )

        buf = RingBuffer(maxlen=300)
        loop = DetectionLoop(source=cap, aprilcam=cam, ring_buffer=buf)
        loop.start()

        detection_registry[source_id] = DetectionEntry(
            source_id=source_id,
            loop=loop,
            ring_buffer=buf,
            aprilcam=cam,
            operations=list(operations),
        )
        detection_registry[source_id]._camera_id = camera_id  # type: ignore[attr-defined]
        detection_registry[source_id]._camera_index = camera_index  # type: ignore[attr-defined]
        detection_registry[source_id]._exclusive_cap = exclusive_cap  # type: ignore[attr-defined]

        return [TextContent(type="text", text=json.dumps(
            {"stream_id": source_id, "operations": operations, "status": "started"}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def stop_stream(source_id: str) -> list[TextContent]:
    """Stop a running tag detection stream.

    This is the counterpart to ``stream_tags``.  It wraps the same teardown
    logic as ``stop_detection``: stops the loop, releases the exclusive
    capture, and re-opens the shared camera handle.

    Args:
        source_id: The source identifier passed to ``stream_tags``.

    Returns:
        On success: ``{"stream_id": "<id>", "status": "stopped"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        entry = detection_registry.pop(source_id, None)
        if entry is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"No stream running on '{source_id}'"}
            ))]

        entry.loop.stop()

        # Release the exclusive capture and re-open the shared camera
        exclusive_cap = getattr(entry, "_exclusive_cap", None)
        if exclusive_cap is not None:
            try:
                exclusive_cap.release()
            except Exception:
                pass
        camera_id = getattr(entry, "_camera_id", None)
        camera_index = getattr(entry, "_camera_index", 0)
        if camera_id is not None:
            try:
                import cv2
                shared_cap = cv2.VideoCapture(camera_index)
                if shared_cap.isOpened():
                    registry.open(shared_cap, handle=camera_id)
            except Exception:
                pass

        return [TextContent(type="text", text=json.dumps(
            {"stream_id": source_id, "status": "stopped"}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def get_tags(source_id: str) -> list[TextContent]:
    """Return the latest tag detections from a running detection loop.

    Args:
        source_id: The camera UUID or playfield_id passed to ``start_detection``.

    Returns:
        On success: ``{"source_id": "<id>", "frame": <int>, "tags": [...]}``.
        Each tag includes ``id``, pixel position, orientation, and velocity.
        Returns ``{"frame": null, "tags": []}`` if no frames have been
        processed yet.
        On error: ``{"error": "<message>"}``.
    """
    try:
        entry = detection_registry.get(source_id)
        if entry is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"No detection running on '{source_id}'"}
            ))]

        latest = entry.ring_buffer.get_latest()
        if latest is None:
            return [TextContent(type="text", text=json.dumps(
                {"source_id": source_id, "frame": None, "tags": []}
            ))]

        result = latest.to_dict()
        result["source_id"] = source_id
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def get_tag_history(
    source_id: str,
    num_frames: int = 30,
) -> list[TextContent]:
    """Return recent tag detection history from a running detection loop's ring buffer.

    Args:
        source_id: The camera UUID or playfield_id passed to ``start_detection``.
        num_frames: Number of most-recent frames to return (default 30,
            max 300 which is the ring buffer capacity).

    Returns:
        On success: ``{"source_id": "<id>", "frames": [...]}``. Each frame
        record includes a frame number, timestamp, and per-tag detections.
        On error: ``{"error": "<message>"}``.
    """
    try:
        entry = detection_registry.get(source_id)
        if entry is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"No detection running on '{source_id}'"}
            ))]

        records = entry.ring_buffer.get_last_n(num_frames)
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "frames": [r.to_dict() for r in records]}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


# ---------------------------------------------------------------------------
# Image processing tools
# ---------------------------------------------------------------------------

_motion_prev_frames: dict[str, Any] = {}


@server.tool()
async def get_frame(
    source_id: str,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Capture a raw frame from a camera or playfield (no processing applied).

    Args:
        source_id: A camera UUID or playfield_id. If a playfield, the
            frame is automatically deskewed.
        format: ``"base64"`` (default) or ``"file"``.
        quality: JPEG encoding quality (0-100, default 85).

    Returns:
        On success (base64): an ``ImageContent`` with inline JPEG data.
        On success (file): ``{"path": "<temp_file_path>"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        frame = resolve_source(source_id)
    except KeyError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    except RuntimeError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    try:
        return format_image_output(frame, format, quality)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def crop_region(
    source_id: str,
    x: int,
    y: int,
    w: int,
    h: int,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Crop a rectangular region from a camera or playfield frame.

    The crop rectangle is clipped to the frame boundaries. Returns an
    error if the clipped region has zero area.

    Args:
        source_id: A camera UUID or playfield_id.
        x: Left edge of the crop rectangle in pixels.
        y: Top edge of the crop rectangle in pixels.
        w: Width of the crop rectangle in pixels.
        h: Height of the crop rectangle in pixels.
        format: ``"base64"`` (default) or ``"file"``.
        quality: JPEG encoding quality (0-100, default 85).

    Returns:
        On success (base64): an ``ImageContent`` with inline JPEG data.
        On success (file): ``{"path": "<temp_file_path>"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        fh, fw = frame.shape[:2]
        # Clip to frame bounds
        x1 = max(0, min(x, fw))
        y1 = max(0, min(y, fh))
        x2 = max(0, min(x + w, fw))
        y2 = max(0, min(y + h, fh))
        if x2 <= x1 or y2 <= y1:
            return [TextContent(type="text", text=json.dumps(
                {"error": "Crop region is entirely outside frame bounds"}
            ))]
        cropped = frame[y1:y2, x1:x2]
        return format_image_output(cropped, format, quality)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def detect_lines(
    source_id: str,
    threshold: int = 50,
    min_length: int = 50,
    max_gap: int = 10,
) -> list[TextContent]:
    """Detect line segments in a frame using probabilistic Hough transform.

    Args:
        source_id: A camera UUID or playfield_id.
        threshold: Hough accumulator threshold (default 50).
        min_length: Minimum line length in pixels (default 50).
        max_gap: Maximum gap between line segments to merge (default 10).

    Returns:
        On success: ``{"source_id": "<id>", "lines": [[x1,y1,x2,y2],...]}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        # Track frame in registry for pipeline integration
        entry = frame_registry.add(frame, source_id)
        try:
            lines = process_detect_lines(entry.processed, threshold, min_length, max_gap)
            entry.results["detect_lines"] = lines
            entry.operations_applied.append("detect_lines")
            return [TextContent(type="text", text=json.dumps(
                {"source_id": source_id, "lines": lines}
            ))]
        finally:
            frame_registry.release(entry.frame_id)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def detect_circles(
    source_id: str,
    min_radius: int = 0,
    max_radius: int = 0,
    param1: float = 100.0,
    param2: float = 30.0,
) -> list[TextContent]:
    """Detect circles in a frame using Hough circle transform.

    Args:
        source_id: A camera UUID or playfield_id.
        min_radius: Minimum circle radius in pixels (0 = no minimum).
        max_radius: Maximum circle radius in pixels (0 = no maximum).
        param1: Canny edge detector upper threshold (default 100).
        param2: Accumulator threshold for circle centers (default 30).

    Returns:
        On success: ``{"source_id": "<id>", "circles": [{"x":..,"y":..,"radius":..},...]}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        # Track frame in registry for pipeline integration
        entry = frame_registry.add(frame, source_id)
        try:
            circles = process_detect_circles(entry.processed, min_radius, max_radius, param1, param2)
            entry.results["detect_circles"] = circles
            entry.operations_applied.append("detect_circles")
            return [TextContent(type="text", text=json.dumps(
                {"source_id": source_id, "circles": circles}
            ))]
        finally:
            frame_registry.release(entry.frame_id)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def detect_contours(
    source_id: str,
    min_area: float = 100.0,
) -> list[TextContent]:
    """Detect contours in a frame, filtered by minimum area.

    Args:
        source_id: A camera UUID or playfield_id.
        min_area: Minimum contour area in pixels squared (default 100).
            Contours smaller than this are discarded.

    Returns:
        On success: ``{"source_id": "<id>", "contours": [...]}``. Each
        contour is a list of ``[x, y]`` vertex points.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        # Track frame in registry for pipeline integration
        entry = frame_registry.add(frame, source_id)
        try:
            contours = process_detect_contours(entry.processed, min_area)
            entry.results["detect_contours"] = contours
            entry.operations_applied.append("detect_contours")
            return [TextContent(type="text", text=json.dumps(
                {"source_id": source_id, "contours": contours}
            ))]
        finally:
            frame_registry.release(entry.frame_id)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def detect_motion(source_id: str) -> list[TextContent]:
    """Detect motion between the current and previous frame using frame differencing.

    The first call for a given source establishes the baseline frame and
    returns ``is_baseline: true`` with no motion regions. Subsequent calls
    compare the current frame against the previous one.

    Args:
        source_id: A camera UUID or playfield_id.

    Returns:
        On success: ``{"source_id": "<id>", "motion_regions": [...],
        "is_baseline": <bool>}``. Each region is a bounding rectangle.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        import cv2

        from aprilcam.image_processing import process_detect_motion

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev = _motion_prev_frames.get(source_id)
        regions = process_detect_motion(frame, prev)
        _motion_prev_frames[source_id] = gray
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "motion_regions": regions, "is_baseline": prev is None}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def detect_qr_codes(source_id: str) -> list[TextContent]:
    """Detect and decode QR codes in a frame.

    Args:
        source_id: A camera UUID or playfield_id.

    Returns:
        On success: ``{"source_id": "<id>", "qr_codes": [...]}``. Each
        QR code entry includes the decoded ``data`` string and ``points``
        (corner coordinates).
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        # Track frame in registry for pipeline integration
        entry = frame_registry.add(frame, source_id)
        try:
            codes = process_detect_qr_codes(entry.processed)
            entry.results["detect_qr"] = codes
            entry.operations_applied.append("detect_qr")
            return [TextContent(type="text", text=json.dumps(
                {"source_id": source_id, "qr_codes": codes}
            ))]
        finally:
            frame_registry.release(entry.frame_id)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def apply_transform(
    source_id: str,
    operation: str,
    params: str = "{}",
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Apply an image transform to a live frame from a camera or playfield.

    Supported operations include ``rotate``, ``scale``, ``threshold``,
    ``edge``, ``blur``, ``grayscale``, and others defined in the
    image_processing module.

    Args:
        source_id: A camera UUID or playfield_id.
        operation: The transform operation name (e.g. ``"rotate"``,
            ``"threshold"``, ``"edge"``).
        params: JSON string with operation-specific parameters
            (e.g. ``'{"angle": 45}'`` for rotate). Defaults to ``"{}"``.
        format: ``"base64"`` (default) or ``"file"``.
        quality: JPEG encoding quality (0-100, default 85).

    Returns:
        On success (base64): an ``ImageContent`` with the transformed JPEG.
        On success (file): ``{"path": "<temp_file_path>"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        try:
            frame = resolve_source(source_id)
        except (KeyError, RuntimeError) as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        import json as _json

        try:
            p = _json.loads(params) if isinstance(params, str) else params
        except Exception:
            p = {}
        from aprilcam.image_processing import process_apply_transform

        try:
            result = process_apply_transform(frame, operation, p)
        except ValueError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        return format_image_output(result, format, quality)
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


# ---------------------------------------------------------------------------
# Live view tools
# ---------------------------------------------------------------------------


@server.tool()
async def start_live_view(
    camera_id: str,
    deskew: bool = True,
    family: str = "36h11",
    proc_width: int = 0,
    use_clahe: bool = False,
    use_sharpen: bool = False,
) -> list[TextContent]:
    """Open a live visualization window with tag detection overlays.

    Spawns a subprocess that opens an OpenCV window showing the camera
    feed with the playfield deskewed to a proportional rectangle.
    Detected tags are drawn with:
    - Green peaked "house" shape indicating the tag's front direction
    - Yellow arrow showing velocity vector
    - Red tag ID number centered on the tag
    - White playfield outline

    Detection data also feeds into a ring buffer accessible via
    ``get_tags`` and ``get_tag_history`` using the returned view_id.

    Args:
        camera_id: An open camera handle from ``open_camera``.
        deskew: Warp the playfield to a top-down rectangle (default True).
        family: AprilTag family (default ``"36h11"``).
        proc_width: Processing width for detection downscale (0 = full).
        use_clahe: Apply CLAHE contrast enhancement before detection.
        use_sharpen: Apply sharpening before detection.

    Returns:
        On success: ``{"view_id": "<id>", "status": "started"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        # Resolve camera_id to a camera index
        try:
            cap = registry.get(camera_id)
        except KeyError:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown camera_id '{camera_id}'"}
            ))]

        # Get the camera index from the handle (cam_0 -> 0, cam_1 -> 1, etc.)
        camera_index = 0
        if camera_id.startswith("cam_"):
            try:
                camera_index = int(camera_id.split("_", 1)[1])
            except (ValueError, IndexError):
                pass

        # Close the camera in the registry so the subprocess can open it
        try:
            registry.close(camera_id)
        except Exception:
            pass

        view_id = f"live_{camera_id}"
        if view_id in live_view_registry:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Live view already running for '{camera_id}'"}
            ))]

        from aprilcam.liveview import LiveViewProcess
        from aprilcam.detection import FrameRecord, TagRecord as _TR, RingBuffer

        buf = RingBuffer(maxlen=300)

        def on_frame(data: dict) -> None:
            """Feed detection data from the child process into the ring buffer."""
            try:
                tags = []
                for td in data.get("tags", []):
                    tags.append(_TR(
                        id=td["id"],
                        center_px=tuple(td["center_px"]),
                        corners_px=td["corners_px"],
                        orientation_yaw=td["orientation_yaw"],
                        world_xy=tuple(td["world_xy"]) if td.get("world_xy") else None,
                        in_playfield=td.get("in_playfield", True),
                        vel_px=tuple(td["vel_px"]) if td.get("vel_px") else None,
                        speed_px=td.get("speed_px"),
                        vel_world=tuple(td["vel_world"]) if td.get("vel_world") else None,
                        speed_world=td.get("speed_world"),
                        heading_rad=td.get("heading_rad"),
                        timestamp=td["timestamp"],
                        frame_index=td["frame_index"],
                    ))
                fr = FrameRecord(
                    timestamp=data["timestamp"],
                    frame_index=data["frame_index"],
                    tags=tags,
                )
                buf.append(fr)
            except Exception:
                pass

        proc = LiveViewProcess(
            camera_index=camera_index,
            deskew=deskew,
            family=family,
            proc_width=proc_width,
            use_clahe=use_clahe,
            use_sharpen=use_sharpen,
        )
        proc.start(on_frame=on_frame)

        live_view_registry[view_id] = LiveViewEntry(
            source_id=view_id,
            process=proc,
            ring_buffer=buf,
        )

        # Also register in detection_registry so get_tags/get_tag_history work
        detection_registry[view_id] = DetectionEntry(
            source_id=view_id,
            loop=_LiveViewLoopAdapter(proc),
            ring_buffer=buf,
            aprilcam=AprilCam(index=0, backend=None, speed_alpha=0.3,
                              family=family, proc_width=proc_width,
                              headless=True, cap=None),
        )

        return [TextContent(type="text", text=json.dumps(
            {"view_id": view_id, "camera_id": camera_id, "status": "started"}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


@server.tool()
async def stop_live_view(view_id: str) -> list[TextContent]:
    """Stop a running live visualization window.

    Args:
        view_id: The view_id returned by ``start_live_view``.

    Returns:
        On success: ``{"view_id": "<id>", "status": "stopped"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        entry = live_view_registry.pop(view_id, None)
        if entry is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"No live view running with id '{view_id}'"}
            ))]

        entry.process.stop()

        # Also remove from detection_registry
        detection_registry.pop(view_id, None)

        # Re-open the camera so it's available again
        camera_id = view_id.replace("live_", "", 1)
        camera_index = 0
        if camera_id.startswith("cam_"):
            try:
                camera_index = int(camera_id.split("_", 1)[1])
            except (ValueError, IndexError):
                pass
        try:
            import cv2
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                registry.open(cap, handle=camera_id)
        except Exception:
            pass

        return [TextContent(type="text", text=json.dumps(
            {"view_id": view_id, "status": "stopped"}
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": f"Unexpected error: {exc}"}))]


class _LiveViewLoopAdapter:
    """Adapts LiveViewProcess to the DetectionLoop interface for detection_registry."""

    def __init__(self, process: Any) -> None:
        self._process = process

    @property
    def is_running(self) -> bool:
        return self._process.is_running

    @property
    def frame_count(self) -> int:
        return 0

    @property
    def error(self) -> Exception | None:
        return None

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 5.0) -> None:
        self._process.stop(timeout=timeout)


# ---------------------------------------------------------------------------
# Operation pipeline
# ---------------------------------------------------------------------------

# Canonical set of operations the pipeline understands.
_KNOWN_OPERATIONS = frozenset({
    "deskew",
    "detect_tags",
    "detect_aruco",
    "detect_lines",
    "detect_circles",
    "detect_contours",
    "detect_qr",
})


def _detect_tags_on_frame(frame_bgr: np.ndarray, family: str = "36h11") -> list[dict]:
    """Detect AprilTags on a BGR frame and return JSON-serializable results.

    Uses ``cv2.aruco`` with the AprilTag dictionary corresponding to
    *family* (default ``"36h11"``).  Each result dict contains ``id``,
    ``family``, ``center_px``, ``corners_px``, and ``orientation_yaw``.
    """
    import cv2

    family_map = {
        "36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    }
    aruco_dict_id = family_map.get(family, cv2.aruco.DICT_APRILTAG_36h11)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    results: list[dict] = []
    if ids is None or len(ids) == 0:
        return results

    for c, tid in zip(corners, ids.flatten()):
        pts = np.array(c, dtype=np.float32).reshape(-1, 2)
        tag = AprilTag.from_corners(
            tag_id=int(tid),
            corners_px=pts,
            family=family,
        )
        results.append({
            "id": tag.id,
            "family": tag.family,
            "center_px": list(tag.center_px),
            "corners_px": tag.corners_px.tolist(),
            "orientation_yaw": tag.orientation_yaw,
        })
    return results


def _detect_aruco_on_frame(frame_bgr: np.ndarray) -> list[dict]:
    """Detect 4x4 ArUco markers and return JSON-serializable results."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detections = detect_aruco_4x4(gray)
    results: list[dict] = []
    for pts, tid in detections:
        center = pts.mean(axis=0)
        results.append({
            "id": int(tid),
            "center_px": [float(center[0]), float(center[1])],
            "corners_px": pts.tolist(),
        })
    return results


def run_operations(entry: FrameEntry, operations: list[str]) -> dict[str, Any]:
    """Execute a batch of operations on a :class:`FrameEntry` in order.

    Parameters
    ----------
    entry:
        The frame entry whose image slots and results will be mutated.
    operations:
        Ordered list of operation names to run.

    Returns
    -------
    dict
        Combined results keyed by operation name.

    Raises
    ------
    ValueError
        If any operation name is not recognised.
    """
    unknown = [op for op in operations if op not in _KNOWN_OPERATIONS]
    if unknown:
        raise ValueError(
            f"Unknown operation(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(_KNOWN_OPERATIONS))}"
        )

    combined: dict[str, Any] = {}

    for op in operations:
        if op == "deskew":
            combined[op] = _run_deskew(entry)
        elif op == "detect_tags":
            result = _detect_tags_on_frame(entry.processed)
            entry.results["detect_tags"] = result
            entry.apriltags = result
            combined[op] = result
        elif op == "detect_aruco":
            result = _detect_aruco_on_frame(entry.processed)
            entry.results["detect_aruco"] = result
            entry.aruco_corners = {d["id"]: d["corners_px"] for d in result}
            combined[op] = result
        elif op == "detect_lines":
            result = process_detect_lines(entry.processed)
            entry.results["detect_lines"] = result
            combined[op] = result
        elif op == "detect_circles":
            result = process_detect_circles(entry.processed)
            entry.results["detect_circles"] = result
            combined[op] = result
        elif op == "detect_contours":
            result = process_detect_contours(entry.processed)
            entry.results["detect_contours"] = result
            combined[op] = result
        elif op == "detect_qr":
            result = process_detect_qr_codes(entry.processed)
            entry.results["detect_qr"] = result
            combined[op] = result

        entry.operations_applied.append(op)

    return combined


def _run_deskew(entry: FrameEntry) -> dict[str, Any]:
    """Apply deskew to *entry* using its source's playfield, if available."""
    source_id = entry.source

    # Try to find a playfield for this source.
    # For file-based sources, the playfield camera_id is "file:<path>".
    pf_entry = None
    try:
        pf_entry = playfield_registry.get(source_id)
    except KeyError:
        pass

    # Also try find_by_camera in case source is a camera handle
    if pf_entry is None:
        pf_id = playfield_registry.find_by_camera(source_id)
        if pf_id is not None:
            pf_entry = playfield_registry.get(pf_id)

    if pf_entry is None:
        return {"applied": False, "reason": "no playfield for source"}

    warped = pf_entry.playfield.deskew(entry.original)
    h, w = warped.shape[:2]
    entry.deskewed = warped
    entry.processed = entry.deskewed
    entry.is_deskewed = True
    return {"applied": True, "width": w, "height": h}


# ---------------------------------------------------------------------------
# Frame lifecycle tools
# ---------------------------------------------------------------------------


@server.tool()
async def create_frame(
    source_id: str,
    operations: list[str] | None = None,
) -> list[TextContent]:
    """Capture a frame from a camera or playfield and store it in the frame registry.

    The frame is stored with three identical image slots (original, deskewed,
    processed).  If *operations* is provided, the operation pipeline runs
    immediately after capture and results are included in the response.

    Args:
        source_id: A camera or playfield handle (e.g. ``"cam_0"``).
        operations: Optional list of pipeline operations to run on the
            frame immediately after capture (e.g.
            ``["deskew", "detect_tags"]``).

    Returns:
        On success: ``{"frame_id": "<id>", "source": "<source_id>"}``
        (plus ``"results"`` when *operations* is provided).
        On error: ``{"error": "<message>"}``.
    """
    try:
        frame = resolve_source(source_id)
    except (KeyError, RuntimeError) as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    entry = frame_registry.add(raw=frame, source=source_id)
    response: dict[str, Any] = {
        "frame_id": entry.frame_id,
        "source": source_id,
    }

    if operations:
        try:
            results = run_operations(entry, operations)
            response["results"] = results
        except ValueError as exc:
            response["error"] = str(exc)

    return [
        TextContent(
            type="text",
            text=json.dumps(response),
        )
    ]


@server.tool()
async def create_frame_from_image(
    image_path: str,
    operations: list[str] | None = None,
) -> list[TextContent]:
    """Load an image file from disk and store it in the frame registry.

    If *operations* is provided, the operation pipeline runs immediately
    after loading and results are included in the response.

    Args:
        image_path: Absolute path to an image file (JPEG, PNG, etc.).
        operations: Optional list of pipeline operations to run on the
            frame immediately after loading (e.g.
            ``["detect_tags", "detect_lines"]``).

    Returns:
        On success: ``{"frame_id": "<id>", "source": "file:<path>"}``
        (plus ``"results"`` when *operations* is provided).
        On error: ``{"error": "<message>"}``.
    """
    import os

    import cv2

    if not os.path.isfile(image_path):
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"File not found: {image_path}"}),
            )
        ]

    img = cv2.imread(image_path)
    if img is None:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to load image: {image_path}"}),
            )
        ]

    source = f"file:{image_path}"
    entry = frame_registry.add(raw=img, source=source)
    response: dict[str, Any] = {
        "frame_id": entry.frame_id,
        "source": source,
    }

    if operations:
        try:
            results = run_operations(entry, operations)
            response["results"] = results
        except ValueError as exc:
            response["error"] = str(exc)

    return [
        TextContent(
            type="text",
            text=json.dumps(response),
        )
    ]


@server.tool()
async def process_frame(
    frame_id: str,
    operations: list[str],
) -> list[TextContent]:
    """Run one or more operations on an existing frame in the registry.

    Operations execute in order on the frame's ``processed`` image slot.
    Detection operations store structured results without modifying the
    image; the ``deskew`` operation replaces the ``deskewed`` and
    ``processed`` slots with a perspective-warped image.

    Supported operations: ``deskew``, ``detect_tags``, ``detect_aruco``,
    ``detect_lines``, ``detect_circles``, ``detect_contours``,
    ``detect_qr``.

    Args:
        frame_id: The frame handle from ``create_frame`` or
            ``create_frame_from_image``.
        operations: Ordered list of operation names to execute.

    Returns:
        On success: ``{"frame_id": "<id>", "results": {<op>: <data>, ...}}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        entry = frame_registry.get(frame_id)
    except KeyError:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Frame '{frame_id}' not found"}),
            )
        ]

    try:
        results = run_operations(entry, operations)
    except ValueError as exc:
        return [
            TextContent(type="text", text=json.dumps({"error": str(exc)}))
        ]

    return [
        TextContent(
            type="text",
            text=json.dumps({"frame_id": frame_id, "results": results}),
        )
    ]


@server.tool()
async def get_frame_image(
    frame_id: str,
    stage: str = "processed",
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Retrieve an image from a stored frame at the specified processing stage.

    Args:
        frame_id: The frame handle returned by ``create_frame`` or
            ``create_frame_from_image``.
        stage: Which image slot to return — ``"original"``, ``"deskewed"``,
            or ``"processed"`` (default).
        format: ``"base64"`` (inline image) or ``"file"`` (temp file path).
        quality: JPEG encoding quality (0–100).

    Returns:
        The encoded image, or ``{"error": "<message>"}`` on failure.
    """
    try:
        entry = frame_registry.get(frame_id)
    except KeyError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    stage_map = {
        "original": entry.original,
        "deskewed": entry.deskewed,
        "processed": entry.processed,
    }

    if stage not in stage_map:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Invalid stage '{stage}'. "
                        f"Must be one of: original, deskewed, processed"
                    }
                ),
            )
        ]

    image = stage_map[stage]
    try:
        return format_image_output(image, format=format, quality=quality)
    except RuntimeError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


@server.tool()
async def save_frame(
    frame_id: str,
    output_dir: str,
) -> list[TextContent]:
    """Save all image stages and metadata for a frame to a directory.

    Creates ``original.jpg``, ``deskewed.jpg``, ``processed.jpg``, and
    ``metadata.json`` in *output_dir*.

    Args:
        frame_id: The frame handle to save.
        output_dir: Directory path where files will be written (created if
            it does not exist).

    Returns:
        On success: ``{"path": "<dir>", "files": [...]}``.
        On error: ``{"error": "<message>"}``.
    """
    import os

    import cv2

    try:
        entry = frame_registry.get(frame_id)
    except KeyError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    os.makedirs(output_dir, exist_ok=True)

    files_written: list[str] = []
    for name, img in [
        ("original.jpg", entry.original),
        ("deskewed.jpg", entry.deskewed),
        ("processed.jpg", entry.processed),
    ]:
        path = os.path.join(output_dir, name)
        cv2.imwrite(path, img)
        files_written.append(name)

    metadata = {
        "frame_id": entry.frame_id,
        "source": entry.source,
        "timestamp": entry.timestamp,
        "operations_applied": list(entry.operations_applied),
        "is_deskewed": entry.is_deskewed,
        "results": entry.results,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    files_written.append("metadata.json")

    return [
        TextContent(
            type="text",
            text=json.dumps({"path": output_dir, "files": files_written}),
        )
    ]


@server.tool()
async def release_frame(frame_id: str) -> list[TextContent]:
    """Remove a frame from the registry, freeing its memory.

    Args:
        frame_id: The frame handle to release.

    Returns:
        On success: ``{"released": true, "frame_id": "<id>"}``.
        On error: ``{"error": "<message>"}``.
    """
    try:
        frame_registry.release(frame_id)
    except KeyError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return [
        TextContent(
            type="text",
            text=json.dumps({"released": True, "frame_id": frame_id}),
        )
    ]


@server.tool()
async def list_frames() -> list[TextContent]:
    """List all frames currently stored in the frame registry.

    Returns:
        A JSON array of frame summary objects, each containing
        ``frame_id``, ``source``, ``timestamp``, ``operations_applied``,
        and ``is_deskewed``.
    """
    summaries = frame_registry.list_frames()
    return [TextContent(type="text", text=json.dumps(summaries))]


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server on stdio transport.

    On shutdown, all detection loops are stopped and all open cameras
    are released, regardless of whether the server exits cleanly or
    due to an exception.

    Args:
        argv: Unused; accepted for CLI entry-point compatibility.
    """
    try:
        server.run(transport="stdio")
    finally:
        # Stop all live views first
        for entry in list(live_view_registry.values()):
            try:
                entry.process.stop()
            except Exception:
                pass
        live_view_registry.clear()
        # Stop all detection loops before closing cameras
        for entry in list(detection_registry.values()):
            try:
                entry.loop.stop()
            except Exception:
                pass
        detection_registry.clear()
        registry.close_all()


if __name__ == "__main__":
    main()
