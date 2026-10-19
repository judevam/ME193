"""Keeps one BLE device connected: reconnects with backoff and reports
"connected"/"disconnected" on status/<node>/ble."""
from __future__ import annotations

import logging
import sys
from typing import Any

from common import topics
from common.bus import Bus
from common.runtime import Backoff
from hardware.interfaces import Device

try:
    from legoeducation.basic_device import VersionMismatchError
except ImportError:  # pragma: no cover
    class VersionMismatchError(Exception):
        pass


class DeviceLink:
    def __init__(self, device: Device, bus: Bus, node: str, cfg: dict[str, Any]) -> None:
        self.device, self.bus, self.node = device, bus, node
        self.log = logging.getLogger(node)
        self.backoff = Backoff(cfg["lego"]["reconnect_min_s"], cfg["lego"]["reconnect_max_s"])
        self.up = False
        self._report("disconnected")

    def _report(self, state: str) -> None:
        self.bus.publish(topics.ble_status_topic(self.node), state)

    def ensure(self, now: float) -> bool:
        """True if the device is usable now; otherwise maybe try to reconnect."""
        if self.device.connected:
            if not self.up:
                self.up = True
                self.backoff.reset()
                self._report("connected")
            return True
        if self.up:
            self.lost("BLE connection dropped")
        if not self.backoff.ready(now):
            return False
        try:
            self.device.connect()
        except VersionMismatchError as e:
            self.log.error("FIRMWARE / PACKAGE VERSION MISMATCH:\n%s\n"
                           "Update the device in the LEGO Education Coding Canvas app "
                           "(code.legoeducation.com) or change the legoeducation version.", e)
            sys.exit(2)
        except Exception as e:
            self.backoff.failed(now)
            self.log.warning("connect failed (%s); retrying in %.0f s", e, self.backoff.next_try - now)
            return False
        self.up = True
        self.backoff.reset()
        self._report("connected")
        return True

    def lost(self, why: str) -> None:
        """Mark the device as down (e.g. a write failed)."""
        if self.up:
            self.log.warning("%s", why)
            self.up = False
            self._report("disconnected")
        self.device.close()
