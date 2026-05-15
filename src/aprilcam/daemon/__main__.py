"""Entry point for ``python -m aprilcam.daemon``.

Loads configuration from all sources and starts the DaemonServer.
"""

from aprilcam.config import Config
from aprilcam.daemon.server import DaemonServer

config = Config.load()
DaemonServer(config).run()
