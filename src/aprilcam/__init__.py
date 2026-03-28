from pathlib import Path as _Path

__all__ = ["__version__", "help"]
__version__ = "0.1.0"

_AGENT_GUIDE = _Path(__file__).parent / "AGENT_GUIDE.md"


def help() -> str:
    """Return the AprilCam agent guide as a markdown string.

    This guide explains how to use AprilCam as a library and via MCP,
    including available tools, common workflows, and tips for AI agents.
    """
    return _AGENT_GUIDE.read_text()
