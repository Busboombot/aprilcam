"""Tests for aprilcam.daemon.server — DaemonServer control socket and RPC."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from aprilcam.config import Config
from aprilcam.daemon.server import DaemonServer


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_config(tmp_path: Path):  # noqa: ARG001 — tmp_path unused but keeps pytest happy
    """Return a Config that uses short paths to satisfy AF_UNIX length limits.

    macOS limits AF_UNIX socket paths to ~104 characters.  pytest's tmp_path
    under /private/var/folders/… is too long, so we create a short-named
    sub-directory under /tmp instead.
    """
    import tempfile, stat

    # Use a fresh dir under /tmp with a short name
    base = Path(tempfile.mkdtemp(prefix="act_", dir="/tmp"))
    # Make it world-accessible so socket permissions don't block tests
    base.chmod(base.stat().st_mode | stat.S_IRWXO)

    sock_dir = base / "s"
    data_dir = base / "d"
    sock_dir.mkdir()
    data_dir.mkdir()

    cfg = Config(
        data_dir=data_dir,
        socket_dir=sock_dir,
        calibration_dir=data_dir / "calibration",
        log_level="DEBUG",
        daemon_pidfile=sock_dir / "aprilcamd.pid",
    )
    yield cfg

    # Cleanup
    import shutil
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture()
def running_server(tmp_config: Config):
    """Start DaemonServer in a background thread; yield; then shut it down."""
    server = DaemonServer(tmp_config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for the control socket to appear (up to 2 s)
    ctrl_path = tmp_config.socket_dir / "control.sock"
    deadline = time.monotonic() + 2.0
    while not ctrl_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert ctrl_path.exists(), "DaemonServer did not bind control socket in time"

    yield server, tmp_config

    # Trigger shutdown via the shutdown RPC, then wait for thread to finish
    try:
        _rpc(ctrl_path, {"cmd": "shutdown"})
    except Exception:
        pass
    server._shutdown_event.set()
    t.join(timeout=5.0)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rpc(ctrl_path: Path, request: dict) -> dict:
    """Send one JSON RPC to the control socket and return the parsed response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(ctrl_path))
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()
    return json.loads(data.split(b"\n")[0])


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_server_binds_control_socket(running_server):
    """The daemon creates control.sock in the socket directory."""
    _, cfg = running_server
    assert (cfg.socket_dir / "control.sock").exists()


def test_list_cameras_empty(running_server):
    """list_cameras returns empty list when no cameras are open."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "list_cameras"})
    assert resp == {"ok": True, "cameras": []}


def test_unknown_command(running_server):
    """Unknown command returns ok=False with an error message."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "xyzzy"})
    assert resp["ok"] is False
    assert "unknown" in resp["error"].lower()


def test_get_calibration_save_path(running_server):
    """get_calibration_save_path returns the configured path."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "get_calibration_save_path"})
    assert resp["ok"] is True
    assert resp["path"] == str(cfg.calibration_dir)


def test_close_unknown_camera(running_server):
    """close_camera for a camera that was never opened returns ok=False."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "close_camera", "cam_name": "cam_99"})
    assert resp["ok"] is False
    assert "cam_99" in resp["error"]


def test_get_camera_info_missing(running_server):
    """get_camera_info for a camera with no info.json returns ok=False."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "get_camera_info", "cam_name": "cam_99"})
    assert resp["ok"] is False


def test_capture_frame_unknown_camera(running_server):
    """capture_frame for an unknown camera returns ok=False."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "capture_frame", "cam_name": "cam_99"})
    assert resp["ok"] is False
    assert "cam_99" in resp["error"]


def test_open_camera_invalid_index_type(running_server):
    """open_camera with a non-integer index returns ok=False."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "open_camera", "index": "not-an-int"})
    assert resp["ok"] is False


def test_reload_calibration_unknown_camera(running_server):
    """reload_calibration for an unknown camera returns ok=False."""
    _, cfg = running_server
    ctrl_path = cfg.socket_dir / "control.sock"
    resp = _rpc(ctrl_path, {"cmd": "reload_calibration", "cam_name": "cam_99"})
    assert resp["ok"] is False


def test_shutdown_rpc(tmp_config: Config):
    """The 'shutdown' RPC causes the server to exit cleanly."""
    server = DaemonServer(tmp_config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    ctrl_path = tmp_config.socket_dir / "control.sock"
    deadline = time.monotonic() + 2.0
    while not ctrl_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)

    resp = _rpc(ctrl_path, {"cmd": "shutdown"})
    assert resp["ok"] is True

    t.join(timeout=5.0)
    assert not t.is_alive(), "Server thread did not exit after shutdown RPC"


def test_pidfile_lock_prevents_duplicate(tmp_config: Config):
    """A second DaemonServer with the same config cannot acquire the pidfile."""
    results = []

    server1 = DaemonServer(tmp_config)
    t1 = threading.Thread(target=server1.run, daemon=True)
    t1.start()

    # Wait for server1 to hold the lock
    ctrl_path = tmp_config.socket_dir / "control.sock"
    deadline = time.monotonic() + 2.0
    while not ctrl_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)

    # server2 should detect "already running" and return quickly
    import sys
    from io import StringIO

    old_stderr = sys.stderr
    sys.stderr = buf = StringIO()
    try:
        server2 = DaemonServer(tmp_config)
        server2.run()  # should return without blocking
        results.append(buf.getvalue())
    finally:
        sys.stderr = old_stderr

    assert "already running" in results[0], f"Expected 'already running', got: {results[0]}"

    # Shut down server1
    _rpc(ctrl_path, {"cmd": "shutdown"})
    t1.join(timeout=5.0)


def test_stale_control_socket_removed(tmp_config: Config):
    """A stale control.sock left from a previous run is cleaned up on start."""
    # Create a stale socket file
    ctrl_path = tmp_config.socket_dir / "control.sock"
    ctrl_path.parent.mkdir(parents=True, exist_ok=True)
    ctrl_path.write_bytes(b"stale")

    server = DaemonServer(tmp_config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    deadline = time.monotonic() + 2.0
    while not ctrl_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)

    # Should be able to connect and get a valid response
    resp = _rpc(ctrl_path, {"cmd": "list_cameras"})
    assert resp["ok"] is True

    _rpc(ctrl_path, {"cmd": "shutdown"})
    t.join(timeout=5.0)
