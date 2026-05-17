"""Integration tests for DaemonControl.

These tests start a real gRPC server in-process and exercise DaemonControl
against it without requiring camera hardware or a running daemon process.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import grpc
import pytest

from aprilcam.client.control import DaemonControl
from aprilcam.daemon.grpc_server import AprilCamServicer, make_grpc_server
from aprilcam.proto import aprilcam_pb2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(tmp_path: Path) -> tuple[grpc.Server, str]:
    """Start an in-process gRPC server on a random TCP port; return (server, target)."""
    from aprilcam.config import Config

    sock_dir = tmp_path / "s"
    data_dir = tmp_path / "d"
    sock_dir.mkdir()
    data_dir.mkdir()

    config = Config(
        data_dir=data_dir,
        socket_dir=sock_dir,
        daemon_pidfile=sock_dir / "aprilcamd.pid",
    )

    cameras: dict = {}
    cam_lock = threading.Lock()
    shutdown = threading.Event()

    servicer = AprilCamServicer(
        cameras=cameras,
        cam_lock=cam_lock,
        config=config,
        shutdown_event=shutdown,
    )
    server = make_grpc_server([], servicer)
    port = server.add_insecure_port("localhost:0")
    server.start()
    return server, f"localhost:{port}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDaemonControlConnect:
    """DaemonControl connection and context-manager behaviour."""

    def test_context_manager_cleans_up(self, tmp_path):
        server, target = _make_server(tmp_path)
        try:
            host, port_str = target.rsplit(":", 1)
            with DaemonControl(host=host, port=int(port_str)) as dc:
                assert dc._stub is not None
            # After __exit__, channel should be closed
            assert dc._channel is None
        finally:
            server.stop(grace=0)

    def test_connect_idempotent(self, tmp_path):
        server, target = _make_server(tmp_path)
        try:
            host, port_str = target.rsplit(":", 1)
            dc = DaemonControl(host=host, port=int(port_str))
            dc.connect()
            stub1 = dc._stub
            dc.connect()
            assert dc._stub is stub1  # same stub — no reconnect
            dc.close()
        finally:
            server.stop(grace=0)

    def test_stub_or_raise_before_connect(self):
        dc = DaemonControl()
        with pytest.raises(RuntimeError, match="not connected"):
            dc._stub_or_raise()


class TestDaemonControlListCameras:
    """list_cameras() returns an empty list when no cameras are open."""

    def test_list_cameras_empty(self, tmp_path):
        server, target = _make_server(tmp_path)
        try:
            host, port_str = target.rsplit(":", 1)
            with DaemonControl(host=host, port=int(port_str)) as dc:
                cameras = dc.list_cameras()
            assert cameras == []
        finally:
            server.stop(grace=0)

    def test_list_cameras_returns_list_of_str(self, tmp_path):
        server, target = _make_server(tmp_path)
        try:
            host, port_str = target.rsplit(":", 1)
            with DaemonControl(host=host, port=int(port_str)) as dc:
                result = dc.list_cameras()
            assert isinstance(result, list)
            for item in result:
                assert isinstance(item, str)
        finally:
            server.stop(grace=0)
