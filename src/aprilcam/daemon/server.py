"""
aprilcam.daemon.server — DaemonServer: gRPC server, pidfile/flock, camera registry.

The daemon owns:
  - An exclusive flock on the pidfile (prevents duplicate daemons).
  - A gRPC server bound to one or both of:
      * Unix domain socket at ``unix_path`` (default /tmp/aprilcam/control.sock)
      * TCP socket at ``0.0.0.0:tcp_port`` (default port 5280)
  - One CameraPipeline per opened camera.

The gRPC servicer (daemon.grpc_server.AprilCamServicer) implements all RPC
methods.  Stream endpoints are managed by ImageStreamProducer / TagStreamProducer
(see daemon.stream).
"""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Dict, Optional

from ..config import Config
from .camera_pipeline import CameraPipeline

log = logging.getLogger(__name__)

_DEFAULT_UNIX_PATH = "/tmp/aprilcam/control.sock"
_DEFAULT_TCP_PORT = 5280


class DaemonServer:
    """Single-process AprilCam daemon.

    Lifecycle::

        server = DaemonServer(config)
        server.run()   # blocks until SIGTERM / SIGINT / Shutdown RPC

    Transport configuration
    -----------------------
    ``unix_enabled``
        When True, the gRPC server binds ``unix://<unix_path>``.
    ``tcp_enabled``
        When True, the gRPC server binds ``[::]:<tcp_port>``.
    ``unix_path``
        Filesystem path for the Unix domain socket
        (default ``/tmp/aprilcam/control.sock``).
    ``tcp_port``
        TCP port number (default 5280).

    At least one of *unix_enabled* or *tcp_enabled* must be True; the
    constructor raises ``ValueError`` if both are False.
    """

    def __init__(
        self,
        config: Config,
        *,
        unix_enabled: bool = True,
        tcp_enabled: bool = True,
        unix_path: str = _DEFAULT_UNIX_PATH,
        tcp_port: int = _DEFAULT_TCP_PORT,
    ) -> None:
        if not unix_enabled and not tcp_enabled:
            raise ValueError(
                "DaemonServer: at least one transport must be enabled "
                "(--unix or --tcp).  Both are currently disabled."
            )

        self._config = config
        self._unix_enabled = unix_enabled
        self._tcp_enabled = tcp_enabled
        self._unix_path = unix_path
        self._tcp_port = tcp_port

        # Cameras: cam_name → CameraPipeline
        self._cameras: Dict[str, CameraPipeline] = {}
        self._cam_lock = threading.Lock()

        # Shutdown coordination
        self._shutdown_event = threading.Event()

        # Pidfile file descriptor (kept open for the lifetime of the process
        # so the flock is held continuously)
        self._pidfile_fd: Optional[int] = None

        # gRPC server (set in run())
        self._grpc_server = None

        # gRPC servicer (set in run(), kept for shutdown)
        self._servicer = None

        # Set once the gRPC server has started — useful for tests / readiness checks
        self.started_event = threading.Event()

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
            log.error("cannot open pidfile %s: %s", pidfile, exc)
            return

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            log.error("already running")
            return
        except OSError:
            # Stale pidfile with no live lock — remove and retry once
            os.close(fd)
            try:
                pidfile.unlink(missing_ok=True)
                fd = os.open(str(pidfile), os.O_RDWR | os.O_CREAT, 0o644)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception as exc:
                log.error("cannot acquire pidfile lock: %s", exc)
                return

        self._pidfile_fd = fd

        # Write our PID
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())

        # 2. Build gRPC server --------------------------------------------
        from .grpc_server import AprilCamServicer, make_grpc_server

        self._servicer = AprilCamServicer(
            cameras=self._cameras,
            cam_lock=self._cam_lock,
            config=self._config,
            shutdown_event=self._shutdown_event,
        )

        transports = []
        if self._unix_enabled:
            unix_sock_path = Path(self._unix_path)
            unix_sock_path.parent.mkdir(parents=True, exist_ok=True)
            transports.append(f"unix:{self._unix_path}")
        if self._tcp_enabled:
            transports.append(f"[::]:{self._tcp_port}")

        self._grpc_server = make_grpc_server(transports, self._servicer)

        # 3. Install signal handlers (only works from the main thread) ------
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except ValueError:
            # Called from a non-main thread (e.g. in tests) — skip signal setup.
            log.debug("aprilcamd: signal handlers not installed (non-main thread)")

        # 4. Start gRPC server --------------------------------------------
        self._grpc_server.start()

        addrs = []
        if self._unix_enabled:
            addrs.append(f"unix:{self._unix_path}")
        if self._tcp_enabled:
            addrs.append(f"tcp://[::]:{self._tcp_port}")
        log.info(
            "aprilcamd started (pid=%d, transports=%s)",
            os.getpid(),
            ", ".join(addrs),
        )

        # Signal that the server is ready (for tests and readiness checks)
        self.started_event.set()

        # 5. Wait for shutdown event --------------------------------------
        # Block until SIGTERM/SIGINT or the Shutdown RPC sets the event.
        self._shutdown_event.wait()

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
        """Stop all pipelines, stop gRPC server, release pidfile, exit."""
        log.info("aprilcamd: shutting down")

        # Stop gRPC server (grace=5 s for in-flight RPCs)
        if self._grpc_server is not None:
            try:
                self._grpc_server.stop(grace=5)
            except Exception:
                log.exception("Error stopping gRPC server")
            self._grpc_server = None

        # Stop stream producers (if servicer tracks them)
        if self._servicer is not None:
            try:
                self._servicer.stop_all_producers()
            except Exception:
                log.exception("Error stopping stream producers")
            self._servicer = None

        # Stop camera pipelines
        with self._cam_lock:
            for pipeline in self._cameras.values():
                try:
                    pipeline.stop()
                except Exception:
                    log.exception("Error stopping pipeline %s", pipeline.cam_name)
            self._cameras.clear()

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
