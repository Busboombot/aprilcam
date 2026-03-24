"""Unified CLI dispatcher for aprilcam."""

import sys


SUBCOMMANDS = {
    "mcp": {
        "help": "Start the MCP server",
        "module": "aprilcam.mcp_server",
    },
    "taggen": {
        "help": "Generate AprilTag or ArUco marker images (PDF or PNG)",
        "module": "aprilcam.cli.taggen_cli",
    },
    "cameras": {
        "help": "List available cameras",
        "module": "aprilcam.cli.cameras_cli",
    },
    "init": {
        "help": "Configure MCP server entries for Claude Code and VS Code",
        "module": "aprilcam.cli.init_cli",
    },
}


def _print_help():
    print("usage: aprilcam <command> [options]")
    print()
    print("AprilCam -- AprilTag detection and generation toolkit")
    print()
    print("commands:")
    for name, info in SUBCOMMANDS.items():
        print(f"  {name:<12} {info['help']}")
    print()
    print("Run 'aprilcam <command> --help' for command-specific options.")


def main(argv=None):
    """Entry point for the aprilcam CLI."""
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    command = args[0]
    remaining = args[1:]

    if command not in SUBCOMMANDS:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)

    # Lazy import: only load the target module when actually dispatching.
    import importlib

    mod = importlib.import_module(SUBCOMMANDS[command]["module"])
    rc = mod.main(remaining) or 0
    sys.exit(rc)
