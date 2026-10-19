"""Laptop keyboard -> flipper and match controls. Teleoperated input.

  space  fire flipper          m  toggle flipper auto/manual (retained)
  d      manual defeat         r  reset (clear defeat)       q  quit

Keys are global hotkeys: they work whichever window has focus.
Run: python -m teleop.keyboard_client
"""
from __future__ import annotations

import logging
from typing import Any

from common import schemas, topics
from common.bus import Bus, MqttBus
from common.config import load_config, setup_logging
from common.runtime import StopFlag
from hardware.interfaces import KeyboardInput

log = logging.getLogger("keyboard")
KEYS = ["space", "m", "d", "r", "q"]


class KeyboardClient:
    def __init__(self, bus: Bus, keys: KeyboardInput) -> None:
        self.bus, self.keys = bus, keys
        self.mode = "auto"
        bus.subscribe(topics.TELEOP_FLIPPER_MODE, self._on_mode)   # pick up the retained mode

    def _on_mode(self, _topic: str, payload: Any) -> None:
        self.mode = schemas.parse_mode(payload) or self.mode

    def handle(self, key: str) -> bool:
        """Act on one key. Returns False on quit."""
        if key == "space":
            self.bus.publish(topics.TELEOP_FLIPPER_FIRE, schemas.event())
            log.info("fire")
        elif key == "m":
            self.mode = "manual" if self.mode == "auto" else "auto"
            self.bus.publish(topics.TELEOP_FLIPPER_MODE, self.mode)
            log.info("flipper mode: %s", self.mode)
        elif key == "d":
            self.bus.publish(topics.TELEOP_DEFEAT, schemas.event())
            log.info("manual defeat")
        elif key == "r":
            self.bus.publish(topics.CONTROL_RESET, schemas.event())
            log.info("reset")
        elif key == "q":
            return False
        return True


def main() -> None:
    from hardware.keys import GlobalKeyboard
    cfg = load_config()
    setup_logging(cfg, "keyboard")
    stop = StopFlag()
    bus = MqttBus(cfg, "keyboard")
    keys = GlobalKeyboard(KEYS)
    client = KeyboardClient(bus, keys)
    log.info("space=fire  m=auto/manual  d=defeat  r=reset  q=quit")
    try:
        while not stop.is_set():
            key = keys.get_key(timeout=0.2)
            if key and not client.handle(key):
                break
    finally:
        keys.close()
        bus.close()


if __name__ == "__main__":
    main()
