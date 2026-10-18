"""cmd/motors -> Double Motor (BLE).

Safety: stops the wheels if no valid command for cmd_timeout_s, if the brain
goes offline (Last Will), on BLE disconnect, and on exit.
Run: python -m bridges.drive_bridge
"""
from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Callable

from bridges.link import DeviceLink
from common import schemas, topics
from common.bus import Bus, MqttBus
from common.config import load_config, setup_logging
from common.runtime import Rate, StopFlag
from hardware import factory
from hardware.interfaces import DriveBase

log = logging.getLogger("drive_bridge")


class DriveBridge:
    """Decides what to send to the wheels. step() is called at drive_bridge.rate_hz."""

    def __init__(self, cfg: dict[str, Any], bus: Bus, drive: DriveBase,
                 clock: Callable[[], float] = time.monotonic) -> None:
        c = cfg["drive_bridge"]
        self.timeout, self.heartbeat = c["cmd_timeout_s"], c["heartbeat_s"]
        self.latency_log_s = c["latency_log_s"]
        self.drive, self.clock = drive, clock
        self.cmd: tuple[float, float] = (0.0, 0.0)
        self.src_t: float | None = None
        self.cmd_rx: float | None = None
        self.brain_online = False
        self.last_sent: tuple[float, float] | None = None
        self.last_send_t = float("-inf")
        self.latencies: list[float] = []
        self.last_latency_log = clock()
        bus.subscribe(topics.CMD_MOTORS, self._on_cmd)
        bus.subscribe(topics.status_topic("brain"), self._on_brain_status)

    def _on_cmd(self, _topic: str, payload: Any) -> None:
        parsed = schemas.parse_motors(payload)
        if parsed:
            self.cmd, self.src_t, self.cmd_rx = parsed[:2], parsed[2], self.clock()

    def _on_brain_status(self, _topic: str, payload: Any) -> None:
        self.brain_online = payload == "online"
        if not self.brain_online:
            log.warning("brain offline: stopping")

    def target(self, now: float) -> tuple[tuple[float, float], float | None]:
        """Command to apply now: the latest one, or stop if it's stale or the brain is gone."""
        if not self.brain_online or self.cmd_rx is None or now - self.cmd_rx > self.timeout:
            return (0.0, 0.0), None
        return self.cmd, self.src_t

    def step(self, now: float) -> None:
        """Send to the motors only on change or heartbeat. Raises if the write fails."""
        cmd, src_t = self.target(now)
        changed = cmd != self.last_sent
        if changed or now - self.last_send_t >= self.heartbeat:
            self.drive.set(*cmd)
            if changed:
                log.debug("wheels %+.2f %+.2f", *cmd)
                if src_t is not None:
                    self.latencies.append(time.time() - src_t)
            self.last_sent, self.last_send_t = cmd, now
        if now - self.last_latency_log >= self.latency_log_s:
            if self.latencies:
                ms = [x * 1000 for x in self.latencies]
                log.info("sensor->motor latency over %d changes: median %.0f ms, max %.0f ms",
                         len(ms), statistics.median(ms), max(ms))
            self.latencies.clear()
            self.last_latency_log = now

    def forget_sent(self) -> None:
        """After a reconnect, resend the current command."""
        self.last_sent = None


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "drive_bridge")
    stop = StopFlag()
    bus = MqttBus(cfg, "drive_bridge")
    drive = factory.make_drive(cfg)
    link = DeviceLink(drive, bus, "drive_bridge", cfg)
    bridge = DriveBridge(cfg, bus, drive)
    rate = Rate(cfg["drive_bridge"]["rate_hz"], stop)
    try:
        while not stop.is_set():
            now = time.monotonic()
            if link.ensure(now):
                try:
                    bridge.step(now)
                except Exception as e:
                    link.lost(f"motor write failed: {e}")
                    bridge.forget_sent()
            if rate.sleep():
                break
    finally:
        try:
            drive.stop()
        except Exception:
            pass
        drive.close()
        bus.close()
        log.info("drive bridge stopped")


if __name__ == "__main__":
    main()
