"""aprilcam.client.control — DaemonControl: typed gRPC stub wrapper.

All RPC methods return Pydantic models from ``aprilcam.client.models``.
Proto-generated types are confined to this module.
"""

from __future__ import annotations

import numpy as np
import cv2
import grpc

from aprilcam.proto import aprilcam_pb2, aprilcam_pb2_grpc
from aprilcam.client.models import (
    CameraInfo,
    ImageFrame,
    StreamEndpoint,
    TagFrame,
)
from aprilcam.client.stream import ImageStreamConsumer, TagStreamConsumer


# ---------------------------------------------------------------------------
# DaemonControl
# ---------------------------------------------------------------------------


class DaemonControl:
    """Typed gRPC stub wrapper for the AprilCam daemon.

    Usage::

        with DaemonControl(unix_path="/tmp/aprilcam/control.sock") as dc:
            cameras = dc.list_cameras()

    Constructor keyword arguments:
      - ``unix_path`` — connect via Unix socket if provided (takes precedence).
      - ``host`` — TCP host (default ``"localhost"``).
      - ``port`` — TCP port (default ``5280``).
    """

    def __init__(
        self,
        unix_path: str | None = None,
        host: str = "localhost",
        port: int = 5280,
    ) -> None:
        self._unix_path = unix_path
        self._host = host
        self._port = port
        self._channel: grpc.Channel | None = None
        self._stub: aprilcam_pb2_grpc.AprilCamStub | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "DaemonControl":
        """Open the gRPC channel and create the stub.

        Idempotent — calling ``connect()`` on an already-connected instance
        is a no-op.
        """
        if self._channel is not None:
            return self
        if self._unix_path:
            target = f"unix:{self._unix_path}"
        else:
            target = f"{self._host}:{self._port}"
        self._channel = grpc.insecure_channel(target)
        self._stub = aprilcam_pb2_grpc.AprilCamStub(self._channel)
        return self

    def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "DaemonControl":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stub_or_raise(self) -> aprilcam_pb2_grpc.AprilCamStub:
        if self._stub is None:
            raise RuntimeError(
                "DaemonControl is not connected — call connect() first "
                "or use it as a context manager."
            )
        return self._stub

    # ------------------------------------------------------------------
    # RPC methods
    # ------------------------------------------------------------------

    def list_cameras(self) -> list[str]:
        """Return names of all currently open cameras."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.ListCamerasResponse = stub.ListCameras(
            aprilcam_pb2.Empty()
        )
        return list(resp.cameras)

    def open_camera(self, index: int) -> str:
        """Open camera by device index; return the assigned ``cam_name``."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.OpenCameraResponse = stub.OpenCamera(
            aprilcam_pb2.OpenCameraRequest(index=index)
        )
        return str(resp.cam_name)

    def close_camera(self, cam_name: str) -> None:
        """Close an open camera."""
        stub = self._stub_or_raise()
        stub.CloseCamera(aprilcam_pb2.CameraRequest(cam_name=cam_name))

    def reload_calibration(self, cam_name: str) -> None:
        """Reload calibration data for a camera from disk."""
        stub = self._stub_or_raise()
        stub.ReloadCalibration(aprilcam_pb2.CameraRequest(cam_name=cam_name))

    def get_camera_info(self, cam_name: str) -> CameraInfo:
        """Return metadata for an open camera."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.CameraInfoResponse = stub.GetCameraInfo(
            aprilcam_pb2.CameraRequest(cam_name=cam_name)
        )
        return CameraInfo.from_proto(resp)

    def capture_frame(self, cam_name: str) -> np.ndarray:
        """Capture a single frame; return a BGR ``np.ndarray``."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.CaptureFrameResponse = stub.CaptureFrame(
            aprilcam_pb2.CameraRequest(cam_name=cam_name)
        )
        buf = np.frombuffer(resp.jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(
                f"Failed to decode JPEG frame from camera '{cam_name}'"
            )
        return frame

    def get_tags(self, cam_name: str) -> TagFrame:
        """Return the most recent tag detections for an open camera."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.TagFrameResponse = stub.GetTags(
            aprilcam_pb2.CameraRequest(cam_name=cam_name)
        )
        return _tag_frame_response_to_pydantic(resp)

    def get_image_stream(
        self, cam_name: str, max_hz: int = 20
    ) -> "ImageStreamConsumer":
        """Request an image stream and return a connected ``ImageStreamConsumer``."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.StreamEndpoint = stub.GetImageStream(
            aprilcam_pb2.StreamRequest(cam_name=cam_name, max_hz=max_hz)
        )
        endpoint = StreamEndpoint.from_proto(resp)
        consumer = ImageStreamConsumer(endpoint, cam_name=cam_name)
        consumer.connect()
        return consumer

    def get_tag_stream(
        self, cam_name: str, max_hz: int = 20
    ) -> "TagStreamConsumer":
        """Request a tag stream and return a connected ``TagStreamConsumer``."""
        stub = self._stub_or_raise()
        resp: aprilcam_pb2.StreamEndpoint = stub.GetTagStream(
            aprilcam_pb2.StreamRequest(cam_name=cam_name, max_hz=max_hz)
        )
        endpoint = StreamEndpoint.from_proto(resp)
        consumer = TagStreamConsumer(endpoint)
        consumer.connect()
        return consumer

    def shutdown(self) -> None:
        """Send the Shutdown RPC; the daemon process will exit."""
        stub = self._stub_or_raise()
        stub.Shutdown(aprilcam_pb2.Empty())


# ---------------------------------------------------------------------------
# Private converters
# ---------------------------------------------------------------------------


def _tag_frame_response_to_pydantic(resp: "aprilcam_pb2.TagFrameResponse") -> TagFrame:
    """Convert a ``TagFrameResponse`` proto message to a ``TagFrame`` Pydantic model.

    ``TagFrameResponse`` is the one-shot GetTags variant; it lacks timestamp
    and fps fields so we default those to zero.
    """
    from aprilcam.client.models import TagRecord

    homo_flat: list[float] = list(resp.homography)
    homography: list[list[float]] | None = None
    if len(homo_flat) == 9:
        homography = [
            homo_flat[0:3],
            homo_flat[3:6],
            homo_flat[6:9],
        ]

    corners_flat: list[float] = list(resp.playfield_corners)
    playfield_corners: list[tuple[float, float]] = [
        (corners_flat[i], corners_flat[i + 1])
        for i in range(0, len(corners_flat), 2)
    ]

    return TagFrame(
        frame_id=int(resp.frame_id),
        ts_mono_ns=0,
        ts_wall_ms=0,
        tags=[TagRecord.from_proto(t) for t in resp.tags],
        homography=homography,
        playfield_corners=playfield_corners,
        fps=0.0,
    )
