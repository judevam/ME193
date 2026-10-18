"""Keyboard-driven stand-in for the color sensor, ultrasonic and controller,
so every tree branch can be demoed with no robot. Use in mock mode only.

  f  floor = ring (white)      b  floor = blue tape
  1..5  opponent at 5 / 10 / 14 / 30 / 80 cm            0  no opponent (no echo)
  j  toggle controller connected
  arrows  levers (up/down = both forward/back, left/right = spin)
  q  quit

Run: python -m sensors.fake_sensors
"""
from __future__ import annotations

import logging

from common import schemas, topics
from common.bus import Bus, MqttBus
from common.config import load_config, setup_logging
from common.runtime import Rate, StopFlag
from hardware.interfaces import KeyboardInput

log = logging.getLogger("fake_sensors")

FLOORS = {"f": "white", "b": "blue"}
DISTANCES = {"1": 5, "2": 10, "3": 14, "4": 30, "5": 80, "0": None}
KEYS = [*FLOORS, *DISTANCES, "j", "q"]
ARROWS = {"up": (1.0, 1.0), "down": (-1.0, -1.0), "left": (-0.6, 0.6), "right": (0.6, -0.6)}


class FakeSensors:
    def __init__(self, bus: Bus, keys: KeyboardInput) -> None:
        self.bus, self.keys = bus, keys
        self.floor = "white"
        self.cm: float | None = None
        self.controller_on = True

    def handle(self, key: str) -> bool:
        """Change the simulated world. Returns False on quit."""
        if key in FLOORS:
            self.floor = FLOORS[key]
            log.info("floor: %s", self.floor)
        elif key in DISTANCES:
            self.cm = DISTANCES[key]
            log.info("opponent: %s", f"{self.cm} cm" if self.cm else "none")
        elif key == "j":
            self.controller_on = not self.controller_on
            log.info("controller %s", "connected" if self.controller_on else "DISCONNECTED")
        elif key == "q":
            return False
        return True

    def levers(self) -> tuple[float, float]:
        for arrow, value in ARROWS.items():
            if self.keys.is_pressed(arrow):
                return value
        return (0.0, 0.0)

    def publish(self) -> None:
        """One round of sensor messages, like the real sensors at their rate."""
        self.bus.publish(topics.SENSOR_COLOR, schemas.color(self.floor, [0, 0, 0]))
        if self.cm is not None:
            self.bus.publish(topics.SENSOR_ULTRASONIC, schemas.ultrasonic(self.cm))
        if self.controller_on:
            self.bus.publish(topics.TELEOP_JOYSTICK, schemas.joystick(*self.levers()))


def main() -> None:
    from hardware.keys import GlobalKeyboard
    cfg = load_config()
    setup_logging(cfg, "fake_sensors")
    stop = StopFlag()
    bus = MqttBus(cfg, "fake_sensors")
    keys = GlobalKeyboard(KEYS)
    sim = FakeSensors(bus, keys)
    rate = Rate(cfg["fake_sensors"]["rate_hz"], stop)
    log.info("f/b floor  1-5,0 opponent  j controller  arrows drive  q quit")
    try:
        while not stop.is_set():
            key = keys.get_key(timeout=0)
            while key:
                if not sim.handle(key):
                    return
                key = keys.get_key(timeout=0)
            sim.publish()
            if rate.sleep():
                break
    finally:
        keys.close()
        bus.close()


if __name__ == "__main__":
    main()
