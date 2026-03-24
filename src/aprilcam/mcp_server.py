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

from aprilcam.aprilcam import AprilCam
from aprilcam.composite import (
    CompositeManager,
    compute_cross_camera_homography,
    map_tags_to_primary,
)
from aprilcam.detection import DetectionLoop, RingBuffer
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
composite_manager = CompositeManager()


@dataclass
class DetectionEntry:
    """A running detection loop bound to a source (camera or playfield)."""

    source_id: str
    loop: DetectionLoop
    ring_buffer: RingBuffer
    aprilcam: AprilCam


detection_registry: dict[str, DetectionEntry] = {}


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

        ret, frame = cap.read()
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

        try:
            pf_entry = playfield_registry.get(source_id)
            try:
                cap = registry.get(pf_entry.camera_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Underlying camera '{pf_entry.camera_id}' is no longer open"}
                ))]
            homography = pf_entry.homography
            poly = pf_entry.playfield.get_polygon()
            if poly is not None:
                playfield_poly = poly
        except KeyError:
            # Not a playfield — try camera registry
            try:
                cap = registry.get(source_id)
            except KeyError:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Unknown source_id '{source_id}'"}
                ))]

        cam = AprilCam(
            index=0,
            backend=None,
            speed_alpha=0.1,
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
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "status": "stopped"}
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
        from aprilcam.image_processing import process_detect_lines

        lines = process_detect_lines(frame, threshold, min_length, max_gap)
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "lines": lines}
        ))]
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
        from aprilcam.image_processing import process_detect_circles

        circles = process_detect_circles(frame, min_radius, max_radius, param1, param2)
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "circles": circles}
        ))]
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
        from aprilcam.image_processing import process_detect_contours

        contours = process_detect_contours(frame, min_area)
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "contours": contours}
        ))]
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
        from aprilcam.image_processing import process_detect_qr_codes

        codes = process_detect_qr_codes(frame)
        return [TextContent(type="text", text=json.dumps(
            {"source_id": source_id, "qr_codes": codes}
        ))]
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
# Entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    try:
        server.run(transport="stdio")
    finally:
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
