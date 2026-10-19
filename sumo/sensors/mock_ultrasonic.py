"""Stand-in for the UNO Q ultrasonic publisher, for use before the sensor is wired.

  python -m sensors.mock_ultrasonic           # opponent sweeps 5..100 cm every 8 s
  python -m sensors.mock_ultrasonic --cm 10   # fixed distance
"""
from __future__ import annotations

import argparse

from common import schemas, topics
from common.bus import MqttBus
from common.config import load_config, setup_logging
from common.runtime import Rate, StopFlag
from hardware.mock import MockUltrasonic

RATE_HZ = 20  # same as the UNO Q app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cm", type=float, default=None, help="fixed distance in cm")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg, "mock_ultrasonic")
    stop = StopFlag()
    bus = MqttBus(cfg, "unoq")   # same node name as the real UNO Q publisher
    sensor = MockUltrasonic(args.cm)
    rate = Rate(RATE_HZ, stop)
    try:
        while not stop.is_set():
            cm = sensor.read_cm()
            if cm is not None:
                bus.publish(topics.SENSOR_ULTRASONIC, schemas.ultrasonic(cm))
            if rate.sleep():
                break
    finally:
        bus.close()


if __name__ == "__main__":
    main()
