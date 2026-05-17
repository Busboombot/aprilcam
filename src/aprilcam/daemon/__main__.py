"""Entry point for ``python -m aprilcam.daemon``.

Loads configuration from all sources, configures logging, and starts the DaemonServer.
"""

import logging
import sys

from aprilcam.config import Config
from aprilcam.daemon.server import DaemonServer

config = Config.load()
logging.basicConfig(
    level=config.log_level,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
DaemonServer(config).run()
