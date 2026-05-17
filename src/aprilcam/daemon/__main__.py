"""Entry point for ``python -m aprilcam.daemon``.

Loads configuration from all sources, configures logging, and starts the DaemonServer.

Transport flags
---------------
--unix / --no-unix      Enable/disable the Unix domain socket transport (default: enabled).
--tcp  / --no-tcp       Enable/disable the TCP transport (default: enabled).
--tcp-port N            TCP port to bind (default: 5280).
--unix-path PATH        Filesystem path for the Unix socket
                        (default: /tmp/aprilcam/control.sock).
"""

import argparse
import logging
import sys

from aprilcam.config import Config
from aprilcam.daemon.server import DaemonServer, _DEFAULT_TCP_PORT, _DEFAULT_UNIX_PATH


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m aprilcam.daemon",
        description="AprilCam daemon",
    )
    parser.add_argument(
        "--unix",
        dest="unix_enabled",
        action="store_true",
        default=True,
        help="Enable Unix domain socket transport (default: enabled)",
    )
    parser.add_argument(
        "--no-unix",
        dest="unix_enabled",
        action="store_false",
        help="Disable Unix domain socket transport",
    )
    parser.add_argument(
        "--tcp",
        dest="tcp_enabled",
        action="store_true",
        default=True,
        help="Enable TCP transport (default: enabled)",
    )
    parser.add_argument(
        "--no-tcp",
        dest="tcp_enabled",
        action="store_false",
        help="Disable TCP transport",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=_DEFAULT_TCP_PORT,
        metavar="N",
        help=f"TCP port to bind (default: {_DEFAULT_TCP_PORT})",
    )
    parser.add_argument(
        "--unix-path",
        default=_DEFAULT_UNIX_PATH,
        metavar="PATH",
        help=f"Unix socket path (default: {_DEFAULT_UNIX_PATH})",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if not args.unix_enabled and not args.tcp_enabled:
        print(
            "error: at least one transport must be enabled (--unix or --tcp).",
            file=sys.stderr,
        )
        sys.exit(1)

    config = Config.load()
    logging.basicConfig(
        level=config.log_level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    DaemonServer(
        config,
        unix_enabled=args.unix_enabled,
        tcp_enabled=args.tcp_enabled,
        unix_path=args.unix_path,
        tcp_port=args.tcp_port,
    ).run()


if __name__ == "__main__":
    main()
