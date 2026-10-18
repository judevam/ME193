"""LEGO Controller levers (BLE) -> teleop/joystick. Teleoperated input.

Publishes {"left", "right"} for tank drive. On disconnect it publishes nothing,
so the brain sees stale data and stops driving.
Run: python -m teleop.joystick_bridge
"""
from __future__ import annotations

import time

from bridges.link import DeviceLink
from common import schemas, topics
from common.bus import MqttBus
from common.config import load_config, setup_logging
from common.runtime import Rate, StopFlag
from hardware import factory


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "joystick_bridge")
    stop = StopFlag()
    bus = MqttBus(cfg, "joystick_bridge")
    controller = factory.make_joystick(cfg)
    link = DeviceLink(controller, bus, "joystick_bridge", cfg)
    rate = Rate(cfg["joystick"]["rate_hz"], stop)
    try:
        while not stop.is_set():
            if link.ensure(time.monotonic()):
                levers = controller.read()
                if levers:
                    bus.publish(topics.TELEOP_JOYSTICK, schemas.joystick(*levers))
            if rate.sleep():
                break
    finally:
        controller.close()
        bus.close()


if __name__ == "__main__":
    main()
