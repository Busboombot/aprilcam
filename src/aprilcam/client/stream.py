"""aprilcam.client.stream — ImageStreamConsumer and TagStreamConsumer.

Each consumer owns a raw TCP or Unix socket, reads length-prefixed
protobuf messages, and converts them to Pydantic models.

Wire framing: 4-byte big-endian uint32 length prefix + protobuf payload.
"""

from __future__ import annotations

import socket
import struct
from typing import Iterator

import cv2
import numpy as np

from aprilcam.client.models import ImageFrame, StreamEndpoint, TagFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*, raising EOFError on short read."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError(
                f"Connection closed after {len(buf)} of {n} expected bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def _read_length_prefixed(sock: socket.socket) -> bytes:
    """Read one length-prefixed protobuf message from *sock*."""
    header = _recv_exactly(sock, 4)
    (length,) = struct.unpack(">I", header)
    return _recv_exactly(sock, length)


# ---------------------------------------------------------------------------
# ImageStreamConsumer
# ---------------------------------------------------------------------------


class ImageStreamConsumer:
    """Reads length-prefixed protobuf ``ImageFrame`` messages from a stream socket.

    Prefer Unix socket when ``endpoint.socket_path`` is set; fall back to TCP.

    Usage::

        consumer = ImageStreamConsumer(endpoint, cam_name="cam0")
        consumer.connect()
        for frame in consumer:          # numpy BGR array
            process(frame)
        consumer.close()
    """

    def __init__(self, endpoint: StreamEndpoint, *, cam_name: str = "") -> None:
        self._endpoint = endpoint
        self._cam_name = cam_name
        self._sock: socket.socket | None = None

    def connect(self) -> "ImageStreamConsumer":
        """Open the stream socket.  Idempotent."""
        if self._sock is not None:
            return self
        if self._endpoint.socket_path:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self._endpoint.socket_path)
        elif self._endpoint.tcp_port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("localhost", self._endpoint.tcp_port))
        else:
            raise ValueError(
                "StreamEndpoint has neither socket_path nor tcp_port"
            )
        self._sock = sock
        return self

    def close(self) -> None:
        """Close the stream socket."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_raw(self) -> tuple[int, bytes]:
        """Read one frame and return ``(frame_id, jpeg_bytes)``."""
        from aprilcam.proto import aprilcam_pb2

        if self._sock is None:
            raise RuntimeError("ImageStreamConsumer is not connected")

        data = _read_length_prefixed(self._sock)
        msg = aprilcam_pb2.ImageFrame()
        msg.ParseFromString(data)
        return int(msg.frame_id), bytes(msg.jpeg)

    def read(self) -> np.ndarray:
        """Read one frame and return a BGR ``np.ndarray``."""
        _, jpeg = self.read_raw()
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode JPEG frame from stream")
        return frame

    def read_image_frame(self) -> ImageFrame:
        """Read one frame and return a full ``ImageFrame`` Pydantic model."""
        from aprilcam.proto import aprilcam_pb2

        if self._sock is None:
            raise RuntimeError("ImageStreamConsumer is not connected")

        data = _read_length_prefixed(self._sock)
        msg = aprilcam_pb2.ImageFrame()
        msg.ParseFromString(data)
        return ImageFrame.from_proto(msg, cam_name=self._cam_name)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames until the connection closes."""
        try:
            while True:
                try:
                    yield self.read()
                except EOFError:
                    break
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ImageStreamConsumer":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# TagStreamConsumer
# ---------------------------------------------------------------------------


class TagStreamConsumer:
    """Reads length-prefixed protobuf ``TagFrame`` messages from a stream socket.

    Prefer Unix socket when ``endpoint.socket_path`` is set; fall back to TCP.

    Usage::

        consumer = TagStreamConsumer(endpoint)
        consumer.connect()
        for tag_frame in consumer:      # TagFrame Pydantic model
            process(tag_frame)
        consumer.close()
    """

    def __init__(self, endpoint: StreamEndpoint) -> None:
        self._endpoint = endpoint
        self._sock: socket.socket | None = None

    def connect(self) -> "TagStreamConsumer":
        """Open the stream socket.  Idempotent."""
        if self._sock is not None:
            return self
        if self._endpoint.socket_path:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self._endpoint.socket_path)
        elif self._endpoint.tcp_port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("localhost", self._endpoint.tcp_port))
        else:
            raise ValueError(
                "StreamEndpoint has neither socket_path nor tcp_port"
            )
        self._sock = sock
        return self

    def close(self) -> None:
        """Close the stream socket."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> TagFrame:
        """Read one message and return a ``TagFrame`` Pydantic model."""
        from aprilcam.proto import aprilcam_pb2

        if self._sock is None:
            raise RuntimeError("TagStreamConsumer is not connected")

        data = _read_length_prefixed(self._sock)
        msg = aprilcam_pb2.TagFrame()
        msg.ParseFromString(data)
        return TagFrame.from_proto(msg)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TagFrame]:
        """Yield TagFrame objects until the connection closes."""
        try:
            while True:
                try:
                    yield self.read()
                except EOFError:
                    break
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TagStreamConsumer":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()
