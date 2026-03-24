"""Unified CLI dispatcher for aprilcam."""

import argparse
import sys


SUBCOMMANDS = {
    "mcp": {
        "help": "Start the MCP server",
        "module": "aprilcam.mcp_server",
    },
    "taggen": {
        "help": "Generate AprilTag images",
        "module": "aprilcam.cli.taggen_cli",
    },
    "arucogen": {
        "help": "Generate ArUco marker images",
        "module": "aprilcam.cli.arucogen_cli",
    },
    "cameras": {
        "help": "List available cameras",
        "module": "aprilcam.cli.cameras_cli",
    },
}


def main(argv=None):
    """Entry point for the aprilcam CLI."""
    parser = argparse.ArgumentParser(
        prog="aprilcam",
        description="AprilCam — AprilTag detection and generation toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    for name, info in SUBCOMMANDS.items():
        sub.add_parser(name, help=info["help"])

    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Lazy import: only load the target module when actually dispatching.
    import importlib

    mod = importlib.import_module(SUBCOMMANDS[args.command]["module"])
    rc = mod.main(remaining) or 0
    sys.exit(rc)
