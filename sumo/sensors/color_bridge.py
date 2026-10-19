"""LEGO color sensor (BLE) -> sensor/color. Autonomous sensor.
Run: python -m sensors.color_bridge
"""
from __future__ import annotations

import logging
import time

from bridges.link import DeviceLink
from common import schemas, topics
from common.bus import MqttBus
from common.config import load_config, setup_logging
from common.runtime import Rate, StopFlag
from hardware import factory

log = logging.getLogger("color_bridge")


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "color_bridge")
    stop = StopFlag()
    bus = MqttBus(cfg, "color_bridge")
    sensor = factory.make_color(cfg)
    link = DeviceLink(sensor, bus, "color_bridge", cfg)
    rate = Rate(cfg["color_bridge"]["rate_hz"], stop)
    last_name = None
    try:
        while not stop.is_set():
            # No reading -> publish nothing, so the brain sees the data go stale.
            if link.ensure(time.monotonic()):
                reading = sensor.read()
                if reading:
                    bus.publish(topics.SENSOR_COLOR, schemas.color(reading.name, reading.rgb))
                    if reading.name != last_name:
                        log.info("color %s rgb %s", reading.name, reading.rgb)
                        last_name = reading.name
            if rate.sleep():
                break
    finally:
        sensor.close()
        bus.close()


if __name__ == "__main__":
    main()
