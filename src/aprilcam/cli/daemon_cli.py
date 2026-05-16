"""CLI subcommand: aprilcam daemon — manage the AprilCam daemon process."""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def _cmd_start(config) -> int:
    """Ensure the daemon is running (auto-spawn if needed)."""
    from aprilcam.daemon.client import ensure_running, _try_connect

    already_up = _try_connect(config.socket_dir / "control.sock") is not None
    client = ensure_running(config)

    if already_up:
        print("daemon already running")
    else:
        print(f"daemon started  (control socket: {config.socket_dir / 'control.sock'})")

    try:
        resp = client.rpc("list_cameras")
        cameras = resp.get("cameras", [])
        if cameras:
            print(f"open cameras: {', '.join(cameras)}")
        else:
            print("no cameras open")
    except Exception:
        pass

    return 0


def _cmd_status(config) -> int:
    """Print daemon status: running/stopped, open cameras, data sockets."""
    from aprilcam.daemon.client import _try_connect, ControlClient

    control_path = config.socket_dir / "control.sock"
    sock = _try_connect(control_path)
    if sock is None:
        print("daemon: stopped")
        return 1

    sock.close()
    client = ControlClient(control_path)

    print("daemon: running")
    print(f"control socket: {control_path}")

    try:
        resp = client.rpc("list_cameras")
        cameras = resp.get("cameras", [])
        if not cameras:
            print("cameras: none open")
        else:
            for cam in cameras:
                print(f"  camera: {cam}")
                info_path = config.data_dir / cam / "info.json"
                try:
                    import json
                    info = json.loads(info_path.read_text())
                    print(f"    data socket : {info.get('data_socket', '?')}")
                    print(f"    paths file  : {info.get('paths_file', '?')}")
                    print(f"    calibrated  : {info.get('calibrated', False)}")
                    fw, fh = info.get("frame_size", [0, 0])
                    print(f"    frame size  : {fw}x{fh}")
                except Exception:
                    pass
    except Exception as exc:
        print(f"warning: could not query cameras: {exc}")

    return 0


def _cmd_stop(config) -> int:
    """Send a shutdown RPC to the running daemon."""
    from aprilcam.daemon.client import _try_connect, ControlClient

    control_path = config.socket_dir / "control.sock"
    if _try_connect(control_path) is None:
        print("daemon: not running")
        return 0

    client = ControlClient(control_path)
    try:
        client.rpc("shutdown")
        print("daemon: shutdown requested")
    except Exception:
        # The daemon closes the socket before we read the response sometimes
        print("daemon: shutdown requested")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam daemon",
        description="Manage the AprilCam daemon process",
    )
    sub = parser.add_subparsers(dest="subcmd", metavar="<subcommand>")
    sub.add_parser("start",  help="Start the daemon (no-op if already running)")
    sub.add_parser("status", help="Show daemon status and open cameras")
    sub.add_parser("stop",   help="Stop the running daemon")

    args = parser.parse_args(argv)

    if args.subcmd is None:
        parser.print_help()
        return 1

    from aprilcam.config import Config
    config = Config.load()

    if args.subcmd == "start":
        return _cmd_start(config)
    if args.subcmd == "status":
        return _cmd_status(config)
    if args.subcmd == "stop":
        return _cmd_stop(config)

    parser.print_help()
    return 1
