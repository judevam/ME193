"""cmd/flipper -> Single Motor (BLE).

Refuses to fire while the brain reports defeated, and drops commands that waited
too long (e.g. queued during a reconnect).
Run: python -m bridges.flipper_bridge
"""
from __future__ import annotations

import logging
import queue
import time
from typing import Any

from bridges.link import DeviceLink
from common import schemas, topics
from common.bus import Bus, MqttBus
from common.config import load_config, setup_logging
from common.runtime import StopFlag
from hardware import factory
from hardware.interfaces import FlipperMotor

log = logging.getLogger("flipper_bridge")
MAX_WAIT_S = 0.5   # a fire older than this is no longer what the brain wants


class FlipperBridge:
    def __init__(self, bus: Bus, flipper: FlipperMotor) -> None:
        self.flipper = flipper
        self.defeated = False
        self.inbox: queue.SimpleQueue[tuple[str, float]] = queue.SimpleQueue()
        bus.subscribe(topics.CMD_FLIPPER, self._on_cmd)
        bus.subscribe(topics.STATE, self._on_state)

    def _on_cmd(self, _topic: str, payload: Any) -> None:
        action = schemas.parse_flipper(payload)
        if action:
            self.inbox.put((action, time.monotonic()))

    def _on_state(self, _topic: str, payload: Any) -> None:
        if isinstance(payload, dict):
            self.defeated = bool(payload.get("defeated"))

    def handle(self, action: str, queued_at: float) -> None:
        """Run one command (blocking while the arm moves)."""
        if time.monotonic() - queued_at > MAX_WAIT_S:
            log.info("dropping stale %s", action)
        elif action == "fire" and self.defeated:
            log.warning("refusing to fire while defeated")
        elif action == "fire":
            self.flipper.fire()
        else:
            self.flipper.home()


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "flipper_bridge")
    stop = StopFlag()
    bus = MqttBus(cfg, "flipper_bridge")
    flipper = factory.make_flipper(cfg)
    link = DeviceLink(flipper, bus, "flipper_bridge", cfg)
    bridge = FlipperBridge(bus, flipper)
    try:
        while not stop.is_set():
            if not link.ensure(time.monotonic()):
                stop.sleep(0.2)
                continue
            try:
                action, queued_at = bridge.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                bridge.handle(action, queued_at)
            except Exception as e:
                link.lost(f"flipper command failed: {e}")
    finally:
        flipper.close()
        bus.close()
        log.info("flipper bridge stopped")


if __name__ == "__main__":
    main()
