"""
aprilcam.daemon.server — DaemonServer: control socket, pidfile/flock, RPC dispatch.

The daemon owns:
  - An exclusive flock on the pidfile (prevents duplicate daemons).
  - A UNIX control socket at <socket_dir>/control.sock.
  - One CameraPipeline per opened camera.
  - One data socket per opened camera at <socket_dir>/<cam_name>/data.sock.

Control protocol: newline-delimited JSON, one request → one response
per TCP-style UNIX connection. See _handle_rpc() for the full command
set.

Data sockets: each subscriber connection gets a queue registered with
the pipeline; a sender thread writes encoded FrameMessage bytes from
the queue to the socket until the connection closes.
"""

from __future__ import annotations

import base64
import errno
import fcntl
import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

from ..config import Config
from .camera_pipeline import CameraPipeline

log = logging.getLogger(__name__)


class DaemonServer:
    """Single-process AprilCam daemon.

    Lifecycle::

        server = DaemonServer(config)
        server.run()   # blocks until SIGTERM / SIGINT / "shutdown" RPC
    """

    def __init__(self, config: Config) -> None:
        self._config = config

        # Cameras: cam_name → CameraPipeline
        self._cameras: Dict[str, CameraPipeline] = {}
        self._cam_lock = threading.Lock()

        # Shutdown coordination
        self._shutdown_event = threading.Event()

        # Pidfile file descriptor (kept open for the lifetime of the process
        # so the flock is held continuously)
        self._pidfile_fd: Optional[int] = None

        # Control socket
        self._control_sock: Optional[socket.socket] = None

        # Data sockets: cam_name → socket.socket (the listening socket)
        self._data_socks: Dict[str, socket.socket] = {}
        self._data_sock_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block until shutdown.  Returns silently if already running."""
        # 1. Acquire exclusive pidfile lock --------------------------------
        pidfile = self._config.daemon_pidfile
        pidfile.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd = os.open(str(pidfile), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            print(f"aprilcamd: cannot open pidfile {pidfile}: {exc}", file=sys.stderr)
            return

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            print("aprilcamd already running", file=sys.stderr)
            return

        self._pidfile_fd = fd

        # Write our PID
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())

        # 2. Bind control socket ------------------------------------------
        control_path = self._config.socket_dir / "control.sock"
        self._control_sock = self._bind_unix_socket(control_path)
        if self._control_sock is None:
            self._release_pidfile()
            return

        # 3. Install signal handlers ---------------------------------------
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        log.info("aprilcamd started (pid=%d, ctrl=%s)", os.getpid(), control_path)

        # 4. Accept loop ---------------------------------------------------
        self._control_sock.settimeout(1.0)  # allow periodic shutdown checks
        while not self._shutdown_event.is_set():
            try:
                conn, _ = self._control_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                # Socket closed during shutdown
                break

            t = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
                name="aprilcam-ctrl-rpc",
            )
            t.start()

        # Clean shutdown
        self._shutdown()

    # ------------------------------------------------------------------
    # Internal: signal handling
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: object) -> None:  # noqa: ARG002
        """SIGTERM / SIGINT → request graceful shutdown."""
        log.info("aprilcamd: received signal %d, shutting down", signum)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Internal: shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Stop all pipelines, close sockets, release pidfile, exit."""
        log.info("aprilcamd: shutting down")

        # Stop camera pipelines
        with self._cam_lock:
            for pipeline in self._cameras.values():
                try:
                    pipeline.stop()
                except Exception:
                    log.exception("Error stopping pipeline %s", pipeline.cam_name)
            self._cameras.clear()

        # Close data listening sockets
        with self._data_sock_lock:
            for cam_name, sock in self._data_socks.items():
                try:
                    sock.close()
                except OSError:
                    pass
                sock_path = self._config.socket_dir / cam_name / "data.sock"
                try:
                    sock_path.unlink()
                except OSError:
                    pass
            self._data_socks.clear()

        # Close control socket
        if self._control_sock is not None:
            try:
                self._control_sock.close()
            except OSError:
                pass
            self._control_sock = None

        control_path = self._config.socket_dir / "control.sock"
        try:
            control_path.unlink()
        except OSError:
            pass

        # Release pidfile
        self._release_pidfile()

        log.info("aprilcamd: goodbye")

    def _release_pidfile(self) -> None:
        if self._pidfile_fd is not None:
            try:
                fcntl.flock(self._pidfile_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._pidfile_fd)
            except OSError:
                pass
            self._pidfile_fd = None

    # ------------------------------------------------------------------
    # Internal: socket helpers
    # ------------------------------------------------------------------

    def _bind_unix_socket(
        self, path: Path, *, backlog: int = 5
    ) -> Optional[socket.socket]:
        """Create and bind a UNIX stream socket at *path*.

        If EADDRINUSE, removes the stale socket file and retries once.
        Returns the listening socket or None on unrecoverable error.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(2):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.bind(str(path))
                sock.listen(backlog)
                return sock
            except OSError as exc:
                sock.close()
                if exc.errno == errno.EADDRINUSE and attempt == 0:
                    log.warning("Stale socket %s — removing and retrying", path)
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
                log.error("Cannot bind control socket %s: %s", path, exc)
                return None
        return None  # unreachable but satisfies type checker

    # ------------------------------------------------------------------
    # Internal: control connection handler
    # ------------------------------------------------------------------

    def _handle_connection(self, conn: socket.socket) -> None:
        """Read one JSON request, dispatch, write one JSON response."""
        try:
            conn.settimeout(10.0)
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk

            line = data.split(b"\n", 1)[0]
            try:
                request = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                response = {"ok": False, "error": f"invalid JSON: {exc}"}
            else:
                response = self._handle_rpc(request)

            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception:
            log.exception("Error handling control connection")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal: RPC dispatch
    # ------------------------------------------------------------------

    def _handle_rpc(self, request: dict) -> dict:
        """Dispatch a decoded RPC request to the appropriate handler."""
        cmd = request.get("cmd")

        if cmd == "list_cameras":
            return self._rpc_list_cameras()

        if cmd == "open_camera":
            index = request.get("index")
            if not isinstance(index, int):
                return {"ok": False, "error": "'index' must be an integer"}
            return self._rpc_open_camera(index)

        if cmd == "close_camera":
            cam_name = request.get("cam_name")
            if not isinstance(cam_name, str):
                return {"ok": False, "error": "'cam_name' must be a string"}
            return self._rpc_close_camera(cam_name)

        if cmd == "reload_calibration":
            cam_name = request.get("cam_name")
            if not isinstance(cam_name, str):
                return {"ok": False, "error": "'cam_name' must be a string"}
            return self._rpc_reload_calibration(cam_name)

        if cmd == "get_camera_info":
            cam_name = request.get("cam_name")
            if not isinstance(cam_name, str):
                return {"ok": False, "error": "'cam_name' must be a string"}
            return self._rpc_get_camera_info(cam_name)

        if cmd == "capture_frame":
            cam_name = request.get("cam_name")
            if not isinstance(cam_name, str):
                return {"ok": False, "error": "'cam_name' must be a string"}
            return self._rpc_capture_frame(cam_name)

        if cmd == "get_calibration_save_path":
            return self._rpc_get_calibration_save_path()

        if cmd == "shutdown":
            self._shutdown_event.set()
            return {"ok": True}

        return {"ok": False, "error": "unknown command"}

    # ------------------------------------------------------------------
    # Internal: individual RPC handlers
    # ------------------------------------------------------------------

    def _rpc_list_cameras(self) -> dict:
        with self._cam_lock:
            cameras = list(self._cameras.keys())
        return {"ok": True, "cameras": cameras}

    def _rpc_open_camera(self, index: int) -> dict:
        cam_name = f"cam_{index}"
        with self._cam_lock:
            if cam_name in self._cameras:
                # Already open — return existing info
                info_path = self._config.data_dir / cam_name / "info.json"
                return {
                    "ok": True,
                    "cam_name": cam_name,
                    "info_json_path": str(info_path),
                }

            pipeline = CameraPipeline(cam_name, index, self._config)
            try:
                pipeline.start()
            except RuntimeError as exc:
                return {"ok": False, "error": str(exc)}

            self._cameras[cam_name] = pipeline

        # Start the per-camera data socket (outside cam_lock to keep lock scope small)
        self._start_data_socket(cam_name, pipeline)

        info_path = self._config.data_dir / cam_name / "info.json"
        return {
            "ok": True,
            "cam_name": cam_name,
            "info_json_path": str(info_path),
        }

    def _rpc_close_camera(self, cam_name: str) -> dict:
        with self._cam_lock:
            pipeline = self._cameras.pop(cam_name, None)
        if pipeline is None:
            return {"ok": False, "error": f"camera '{cam_name}' not open"}

        # Stop the data socket listener first so no new subscribers attach
        self._stop_data_socket(cam_name)

        try:
            pipeline.stop()
        except Exception as exc:
            log.exception("Error stopping pipeline %s", cam_name)
            return {"ok": False, "error": str(exc)}

        return {"ok": True}

    def _rpc_reload_calibration(self, cam_name: str) -> dict:
        with self._cam_lock:
            pipeline = self._cameras.get(cam_name)
        if pipeline is None:
            return {"ok": False, "error": f"camera '{cam_name}' not open"}

        from ..calibration.calibration import load_calibration_for_camera

        cal_source = self._config.calibration_source
        try:
            calibration = load_calibration_for_camera(
                cam_name, data_dir=cal_source.parent
            )
        except Exception as exc:
            return {"ok": False, "error": f"calibration load failed: {exc}"}

        # Push the new calibration into the pipeline's april_cam instance
        if pipeline._april_cam is not None:
            if calibration is not None:
                pipeline._april_cam.homography = calibration.homography
                pipeline._calibration = calibration
            else:
                pipeline._april_cam.homography = None
                pipeline._calibration = None

        return {"ok": True}

    def _rpc_get_camera_info(self, cam_name: str) -> dict:
        info_path = self._config.data_dir / cam_name / "info.json"
        if not info_path.exists():
            return {"ok": False, "error": f"info.json not found for '{cam_name}'"}
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "error": f"cannot read info.json: {exc}"}
        return {"ok": True, "info": info}

    def _rpc_capture_frame(self, cam_name: str) -> dict:
        with self._cam_lock:
            pipeline = self._cameras.get(cam_name)
        if pipeline is None:
            return {"ok": False, "error": f"camera '{cam_name}' not open"}

        jpeg = pipeline.capture_frame()
        if jpeg is None:
            return {"ok": False, "error": "no frame captured yet"}

        frame_b64 = base64.b64encode(jpeg).decode("ascii")
        return {"ok": True, "frame_b64": frame_b64}

    def _rpc_get_calibration_save_path(self) -> dict:
        path = self._config.calibration_save_path
        return {"ok": True, "path": str(path)}

    # ------------------------------------------------------------------
    # Internal: per-camera data socket
    # ------------------------------------------------------------------

    def _start_data_socket(self, cam_name: str, pipeline: CameraPipeline) -> None:
        """Create the data socket for *cam_name* and start an acceptor thread."""
        sock_path = self._config.socket_dir / cam_name / "data.sock"
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        data_sock = self._bind_unix_socket(sock_path)
        if data_sock is None:
            log.error("Cannot bind data socket for %s", cam_name)
            return

        with self._data_sock_lock:
            self._data_socks[cam_name] = data_sock

        t = threading.Thread(
            target=self._data_accept_loop,
            args=(cam_name, data_sock, pipeline),
            daemon=True,
            name=f"aprilcam-data-accept-{cam_name}",
        )
        t.start()

    def _stop_data_socket(self, cam_name: str) -> None:
        """Close the data socket listener for *cam_name*."""
        with self._data_sock_lock:
            sock = self._data_socks.pop(cam_name, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        sock_path = self._config.socket_dir / cam_name / "data.sock"
        try:
            sock_path.unlink()
        except OSError:
            pass

    def _data_accept_loop(
        self,
        cam_name: str,
        server_sock: socket.socket,
        pipeline: CameraPipeline,
    ) -> None:
        """Accept subscriber connections on the data socket for *cam_name*."""
        server_sock.settimeout(1.0)
        while not self._shutdown_event.is_set():
            try:
                conn, _ = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            sub_q: queue.Queue[bytes] = queue.Queue(maxsize=2)
            pipeline.add_subscriber(sub_q)

            t = threading.Thread(
                target=self._data_sender,
                args=(conn, sub_q, pipeline),
                daemon=True,
                name=f"aprilcam-data-send-{cam_name}",
            )
            t.start()

    def _data_sender(
        self,
        conn: socket.socket,
        sub_q: queue.Queue,
        pipeline: CameraPipeline,
    ) -> None:
        """Dequeue encoded FrameMessage bytes and write them to *conn*."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    data: bytes = sub_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    conn.sendall(data)
                except OSError:
                    break
        finally:
            pipeline.remove_subscriber(sub_q)
            try:
                conn.close()
            except OSError:
                pass
