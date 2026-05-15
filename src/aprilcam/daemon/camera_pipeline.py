"""
aprilcam.daemon.camera_pipeline — per-camera capture, detection, and fan-out.

CameraPipeline owns one camera index.  It:
  - opens a cv.VideoCapture on start()
  - loads calibration for the camera (if available)
  - runs a background capture thread that:
      * reads frames from the camera
      * calls AprilCam.process_frame()
      * JPEG-encodes the result
      * builds a FrameMessage and encodes it
      * fans out the encoded bytes to all registered subscriber queues
  - writes info.json atomically to <data_dir>/<cam_name>/info.json

The data socket itself is managed by daemon.server.  This module only
manages subscriber queues (add_subscriber / remove_subscriber).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

from ..calibration.calibration import load_calibration_for_camera
from ..config import Config
from ..core.aprilcam import AprilCam
from ..core.detection import FrameRecord, RingBuffer, TagRecord
from .protocol import FrameMessage, encode_frame

log = logging.getLogger(__name__)

_JPEG_QUALITY = 85
_RING_BUFFER_SIZE = 300
_FPS_WINDOW = 30  # number of recent frame timestamps to keep for rolling FPS


class CameraPipeline:
    """Capture, detect, encode, and fan-out for a single camera.

    Lifecycle::

        pipeline = CameraPipeline("cam0", 0, config)
        pipeline.add_subscriber(q)
        pipeline.start()
        ...
        pipeline.stop()
    """

    def __init__(self, cam_name: str, index: int, config: Config) -> None:
        """Set up state only.  Does NOT open the camera.

        Args:
            cam_name: Human-readable camera name (used as directory key).
            index:    OpenCV camera index.
            config:   Daemon configuration (paths, etc.).
        """
        self.cam_name = cam_name
        self.index = index
        self.config = config

        # Camera and detection state (populated on start())
        self._cap: Optional[cv.VideoCapture] = None
        self._april_cam: Optional[AprilCam] = None
        self._calibration = None  # CameraCalibration | None

        # Ring buffer for tag history
        self._ring: RingBuffer = RingBuffer(maxlen=_RING_BUFFER_SIZE)

        # Latest raw frame (JPEG bytes) for capture_frame() RPC
        self._latest_raw_jpeg: Optional[bytes] = None
        self._raw_lock = threading.Lock()

        # Subscriber fan-out
        self._subscribers: List[queue.Queue] = []
        self._sub_lock = threading.Lock()

        # Frame counter
        self._frame_id: int = 0

        # Rolling FPS (deque of monotonic timestamps)
        self._ts_deque: deque[float] = deque(maxlen=_FPS_WINDOW)

        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def add_subscriber(self, q: queue.Queue) -> None:
        """Register a queue to receive encoded FrameMessage bytes."""
        with self._sub_lock:
            if q not in self._subscribers:
                self._subscribers.append(q)

    def remove_subscriber(self, q: queue.Queue) -> None:
        """Unregister a subscriber queue."""
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera, load calibration, write info.json, start thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("CameraPipeline(%s): already running", self.cam_name)
            return

        # Open camera
        cap = cv.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"CameraPipeline: failed to open camera index {self.index}"
            )
        self._cap = cap

        # Load calibration (may be None)
        cal_source = self.config.calibration_source
        self._calibration = load_calibration_for_camera(
            self.cam_name, data_dir=cal_source.parent
        )

        # Build AprilCam instance (headless, no display)
        homography: Optional[np.ndarray] = None
        if self._calibration is not None:
            homography = self._calibration.homography

        self._april_cam = AprilCam(
            index=self.index,
            backend=None,
            speed_alpha=0.1,
            family="36h11",
            proc_width=640,
            cap=self._cap,          # pass already-opened cap
            homography=homography,
            headless=True,
        )

        # Determine frame size
        frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Write info.json
        self._write_info_json(frame_w, frame_h, homography)

        # Start capture thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"aprilcam-{self.cam_name}",
            daemon=True,
        )
        self._thread.start()
        log.info("CameraPipeline(%s): started (index=%d)", self.cam_name, self.index)

    def stop(self) -> None:
        """Signal the capture thread to stop, then release the camera."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning(
                    "CameraPipeline(%s): thread did not stop cleanly", self.cam_name
                )
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        # Remove data socket file if present
        sock_path = self.config.socket_dir / self.cam_name / "data.sock"
        if sock_path.exists():
            try:
                sock_path.unlink()
            except OSError:
                pass

        log.info("CameraPipeline(%s): stopped", self.cam_name)

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------

    def capture_frame(self) -> Optional[bytes]:
        """Return the most recent raw camera frame as JPEG bytes.

        Returns ``None`` if no frame has been captured yet.
        """
        with self._raw_lock:
            return self._latest_raw_jpeg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_info_json(
        self,
        frame_w: int,
        frame_h: int,
        homography: Optional[np.ndarray],
    ) -> None:
        """Write <data_dir>/<cam_name>/info.json atomically."""
        cam_dir = self.config.data_dir / self.cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        info = {
            "data_socket": str(
                self.config.socket_dir / self.cam_name / "data.sock"
            ),
            "paths_file": str(self.config.data_dir / self.cam_name / "paths.json"),
            "device_name": self.cam_name,
            "homography": homography.tolist() if homography is not None else None,
            "calibrated": homography is not None,
            "frame_size": [frame_w, frame_h],
        }
        dest = cam_dir / "info.json"
        tmp = cam_dir / "info.json.tmp"
        tmp.write_text(json.dumps(info, indent=2))
        tmp.rename(dest)

    def _rolling_fps(self) -> float:
        """Compute FPS as frames / elapsed over the rolling window."""
        now = time.monotonic()
        self._ts_deque.append(now)
        if len(self._ts_deque) < 2:
            return 0.0
        elapsed = self._ts_deque[-1] - self._ts_deque[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._ts_deque) - 1) / elapsed

    def _capture_loop(self) -> None:
        """Background thread: read → detect → encode → fan-out."""
        assert self._cap is not None
        assert self._april_cam is not None

        paths_file = str(self.config.data_dir / self.cam_name / "paths.json")
        homography = self._april_cam.homography

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                log.warning(
                    "CameraPipeline(%s): camera read failed, stopping", self.cam_name
                )
                break

            now_mono = time.monotonic()
            ts_mono_ns = time.monotonic_ns()
            ts_wall_ms = int(time.time() * 1000)

            # JPEG-encode raw frame for capture_frame() RPC BEFORE processing
            ok_raw, raw_buf = cv.imencode(
                ".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if ok_raw:
                with self._raw_lock:
                    self._latest_raw_jpeg = raw_buf.tobytes()

            # Run detection / tracking
            try:
                tag_records: List[TagRecord] = self._april_cam.process_frame(
                    frame, now_mono
                )
            except Exception:
                log.exception(
                    "CameraPipeline(%s): process_frame error", self.cam_name
                )
                tag_records = []

            # Store in ring buffer
            frame_record = FrameRecord(
                timestamp=now_mono,
                frame_index=self._frame_id,
                tags=tag_records,
            )
            self._ring.append(frame_record)

            # Compute rolling FPS
            fps = self._rolling_fps()

            # Frame size
            frame_h, frame_w = frame.shape[:2]

            # JPEG-encode frame (same raw JPEG for wire — lightweight)
            if ok_raw:
                frame_jpeg = raw_buf.tobytes()
            else:
                frame_jpeg = b""

            # Build playfield corners
            poly = self._april_cam.playfield.get_polygon()
            if poly is not None:
                playfield_corners: list = poly.tolist()
            else:
                playfield_corners = []

            # Build homography list for wire format
            homography_list: Optional[list] = None
            if homography is not None:
                homography_list = homography.tolist()

            # Assemble FrameMessage
            msg = FrameMessage(
                schema=1,
                frame_id=self._frame_id,
                ts_mono_ns=ts_mono_ns,
                ts_wall_ms=ts_wall_ms,
                frame_jpeg=frame_jpeg,
                frame_w=frame_w,
                frame_h=frame_h,
                tags=[tr.to_dict() for tr in tag_records],
                homography=homography_list,
                playfield_corners=playfield_corners,
                paths_file=paths_file,
                fps=fps,
            )

            encoded = encode_frame(msg)
            self._frame_id += 1

            # Fan-out to subscribers
            with self._sub_lock:
                subs = list(self._subscribers)
            for sub_q in subs:
                try:
                    sub_q.put_nowait(encoded)
                except queue.Full:
                    pass  # silent drop
