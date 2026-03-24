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


@server.tool()
async def create_playfield_from_image(
    image_path: str,
) -> list[TextContent]:
    """Create a playfield from a static image file by detecting ArUco corner markers."""
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


@server.tool()
async def deskew_image(
    playfield_id: str,
    image_path: str,
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Read a static image and apply a playfield's deskew transform."""
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


@server.tool()
async def get_playfield_info(
    playfield_id: str,
) -> list[TextContent]:
    """Return the current state of a registered playfield."""
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

    If *correspondence_points* is empty, auto-detect shared ArUco markers
    between both cameras.  Otherwise provide a JSON string of point pairs:
    ``[[px1,py1,sx1,sy1], ...]`` where each entry has primary x,y then
    secondary x,y.
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
    """Start a tag detection loop on a camera or playfield."""
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


@server.tool()
async def stop_detection(source_id: str) -> list[TextContent]:
    """Stop a running tag detection loop."""
    entry = detection_registry.pop(source_id, None)
    if entry is None:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"No detection running on '{source_id}'"}
        ))]

    entry.loop.stop()
    return [TextContent(type="text", text=json.dumps(
        {"source_id": source_id, "status": "stopped"}
    ))]


@server.tool()
async def get_tags(source_id: str) -> list[TextContent]:
    """Return the latest tag detections from a running detection loop."""
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


@server.tool()
async def get_tag_history(
    source_id: str,
    num_frames: int = 30,
) -> list[TextContent]:
    """Return recent tag detection history from a running detection loop."""
    entry = detection_registry.get(source_id)
    if entry is None:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"No detection running on '{source_id}'"}
        ))]

    records = entry.ring_buffer.get_last_n(num_frames)
    return [TextContent(type="text", text=json.dumps(
        {"source_id": source_id, "frames": [r.to_dict() for r in records]}
    ))]


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
    """Capture a raw frame from a camera or playfield."""
    try:
        frame = resolve_source(source_id)
    except KeyError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    except RuntimeError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    return format_image_output(frame, format, quality)


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
    """Crop a rectangular region from a camera or playfield frame."""
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


@server.tool()
async def detect_lines(
    source_id: str,
    threshold: int = 50,
    min_length: int = 50,
    max_gap: int = 10,
) -> list[TextContent]:
    """Detect line segments in a frame using Hough transform."""
    try:
        frame = resolve_source(source_id)
    except (KeyError, RuntimeError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    from aprilcam.image_processing import process_detect_lines

    lines = process_detect_lines(frame, threshold, min_length, max_gap)
    return [TextContent(type="text", text=json.dumps(
        {"source_id": source_id, "lines": lines}
    ))]


@server.tool()
async def detect_circles(
    source_id: str,
    min_radius: int = 0,
    max_radius: int = 0,
    param1: float = 100.0,
    param2: float = 30.0,
) -> list[TextContent]:
    """Detect circles in a frame using Hough transform."""
    try:
        frame = resolve_source(source_id)
    except (KeyError, RuntimeError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    from aprilcam.image_processing import process_detect_circles

    circles = process_detect_circles(frame, min_radius, max_radius, param1, param2)
    return [TextContent(type="text", text=json.dumps(
        {"source_id": source_id, "circles": circles}
    ))]


@server.tool()
async def detect_contours(
    source_id: str,
    min_area: float = 100.0,
) -> list[TextContent]:
    """Detect contours in a frame."""
    try:
        frame = resolve_source(source_id)
    except (KeyError, RuntimeError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    from aprilcam.image_processing import process_detect_contours

    contours = process_detect_contours(frame, min_area)
    return [TextContent(type="text", text=json.dumps(
        {"source_id": source_id, "contours": contours}
    ))]


@server.tool()
async def detect_motion(source_id: str) -> list[TextContent]:
    """Detect motion between current and previous frame."""
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


@server.tool()
async def detect_qr_codes(source_id: str) -> list[TextContent]:
    """Detect and decode QR codes in a frame."""
    try:
        frame = resolve_source(source_id)
    except (KeyError, RuntimeError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    from aprilcam.image_processing import process_detect_qr_codes

    codes = process_detect_qr_codes(frame)
    return [TextContent(type="text", text=json.dumps(
        {"source_id": source_id, "qr_codes": codes}
    ))]


@server.tool()
async def apply_transform(
    source_id: str,
    operation: str,
    params: str = "{}",
    format: str = "base64",
    quality: int = 85,
) -> list[TextContent | ImageContent]:
    """Apply an image transform to a frame.

    The *params* argument is a JSON string with operation-specific
    parameters (e.g. ``{"angle": 45}`` for rotate).
    """
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
