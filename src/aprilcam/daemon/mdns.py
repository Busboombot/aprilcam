"""
aprilcam.daemon.mdns — MDNSAdvertiser: mDNS/Bonjour service advertisement.

Registers a ``_aprilcam._tcp.local.`` service record via the ``zeroconf``
library when the daemon starts with TCP transport enabled.  All zeroconf
calls are wrapped in try/except so that failures never crash the daemon.
"""

from __future__ import annotations

import logging
import socket
from typing import Optional

log = logging.getLogger(__name__)

_SERVICE_TYPE = "_aprilcam._tcp.local."


class MDNSAdvertiser:
    """Advertise the AprilCam daemon on the local network via mDNS/Bonjour.

    Usage::

        advertiser = MDNSAdvertiser()
        advertiser.start(tcp_port=5280)
        # ... daemon runs ...
        advertiser.stop()

    All errors are caught and logged as warnings; they do not propagate.
    """

    def __init__(self) -> None:
        self._zeroconf: Optional[object] = None
        self._info: Optional[object] = None

    def start(self, tcp_port: int) -> None:
        """Register the mDNS service record.

        Parameters
        ----------
        tcp_port:
            The TCP port the gRPC server is listening on.
        """
        try:
            from zeroconf import ServiceInfo, Zeroconf  # type: ignore[import]

            hostname = socket.gethostname()
            service_name = f"aprilcam-{hostname}.{_SERVICE_TYPE}"

            # Resolve a local IP address for the service record.
            try:
                addr_str = socket.gethostbyname(hostname)
                addr_bytes = socket.inet_aton(addr_str)
            except OSError:
                addr_bytes = socket.inet_aton("127.0.0.1")

            self._info = ServiceInfo(
                type_=_SERVICE_TYPE,
                name=service_name,
                port=tcp_port,
                properties={b"version": b"1", b"host": hostname.encode()},
                addresses=[addr_bytes],
            )

            self._zeroconf = Zeroconf()
            self._zeroconf.register_service(self._info)

            log.info(
                "mDNS: registered %s on port %d",
                service_name,
                tcp_port,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mDNS: registration failed (daemon continues): %s", exc)
            self._zeroconf = None
            self._info = None

    def stop(self) -> None:
        """Unregister the mDNS service record and close the Zeroconf instance."""
        if self._zeroconf is None:
            return
        try:
            if self._info is not None:
                self._zeroconf.unregister_service(self._info)
            self._zeroconf.close()
            log.info("mDNS: service unregistered")
        except Exception as exc:  # noqa: BLE001
            log.warning("mDNS: error during stop: %s", exc)
        finally:
            self._zeroconf = None
            self._info = None
