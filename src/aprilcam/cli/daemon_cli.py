"""CLI subcommand: aprilcam daemon — manage the AprilCam daemon process."""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def _read_pid(config) -> Optional[int]:
    """Return the daemon PID from the pidfile, or None if unreadable."""
    try:
        return int(config.daemon_pidfile.read_text().strip())
    except Exception:
        return None


def _cmd_start(config, verbosity: int = 0, detach: bool = False) -> int:
    """Ensure the daemon is running (auto-spawn if needed)."""
    foreground = verbosity > 0 and not detach

    if foreground:
        import logging
        level = logging.DEBUG if verbosity >= 2 else logging.INFO
        logging.basicConfig(
            level=level,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        from aprilcam.daemon.server import DaemonServer
        DaemonServer(config).run()
        return 0

    from aprilcam.daemon.client import ensure_running, _try_connect

    already_up = _try_connect(config.socket_dir / "control.sock") is not None
    log_level = "DEBUG" if verbosity >= 2 else "INFO" if verbosity == 1 else None
    client = ensure_running(config, log_level=log_level)
    pid = _read_pid(config)
    pid_str = f"  pid {pid}" if pid else ""

    if already_up:
        print(f"daemon already running{pid_str}")
    else:
        print(f"daemon started{pid_str}  (control socket: {config.socket_dir / 'control.sock'})")

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

    pid = _read_pid(config)
    pid_str = f"  (pid {pid})" if pid else ""
    print(f"daemon: running{pid_str}")
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
                    from pathlib import Path
                    info = json.loads(info_path.read_text())
                    print(f"    data socket : {info.get('data_socket', '?')}")
                    print(f"    calibrated  : {info.get('calibrated', False)}")
                    fw, fh = info.get("frame_size", [0, 0])
                    print(f"    frame size  : {fw}x{fh}")
                    pf = info.get("playfield")
                    if pf:
                        print(f"    playfield   : {pf.get('width_cm')}cm × {pf.get('height_cm')}cm")
                    paths_file = info.get("paths_file")
                    print(f"    paths file  : {paths_file or '?'}")
                    if paths_file:
                        try:
                            paths = json.loads(Path(paths_file).read_text())
                            print(f"    paths       : {len(paths)} path(s) queued to draw")
                        except Exception:
                            print(f"    paths       : (unreadable)")
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


def _cmd_restart(config, verbosity: int = 0, detach: bool = False) -> int:
    """Stop the daemon if running, then start it."""
    import time
    _cmd_stop(config)
    # Wait until the control socket disappears (daemon fully exited)
    control_path = config.socket_dir / "control.sock"
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        time.sleep(0.1)
        if not control_path.exists():
            break
    return _cmd_start(config, verbosity=verbosity, detach=detach)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam daemon",
        description="Manage the AprilCam daemon process",
    )
    sub = parser.add_subparsers(dest="subcmd", metavar="<subcommand>")

    for name, help_text in [
        ("start",   "Start the daemon (no-op if already running)"),
        ("restart", "Stop then start the daemon"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "-v",
            dest="verbosity",
            action="count",
            default=0,
            help="INFO logging, stay in foreground (-vv for DEBUG)",
        )
        p.add_argument(
            "-d", "--detach",
            action="store_true",
            help="Detach even when -v/-vv given; logs go to aprilcamd.log",
        )

    sub.add_parser("status",  help="Show daemon status and open cameras")
    sub.add_parser("stop",    help="Stop the running daemon")

    args = parser.parse_args(argv)

    if args.subcmd is None:
        parser.print_help()
        return 1

    from aprilcam.config import Config
    config = Config.load()

    if args.subcmd == "start":
        return _cmd_start(config, verbosity=args.verbosity, detach=args.detach)
    if args.subcmd == "status":
        return _cmd_status(config)
    if args.subcmd == "stop":
        return _cmd_stop(config)
    if args.subcmd == "restart":
        return _cmd_restart(config, verbosity=args.verbosity, detach=args.detach)

    parser.print_help()
    return 1
