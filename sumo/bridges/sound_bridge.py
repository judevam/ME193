"""cmd/sound -> laptop speaker (fallback for the UNO Q buzzer).
Run: python -m bridges.sound_bridge
"""
from __future__ import annotations

import logging
import queue

from common.bus import MqttBus
from common.config import load_config, setup_logging
from common.runtime import StopFlag
from common import topics
from hardware import factory

log = logging.getLogger("sound_bridge")


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "sound_bridge")
    stop = StopFlag()
    bus = MqttBus(cfg, "sound_bridge")
    speaker = factory.make_sound(cfg)
    songs: queue.SimpleQueue[str] = queue.SimpleQueue()

    def on_sound(_topic, payload):
        if isinstance(payload, dict) and isinstance(payload.get("song"), str):
            songs.put(payload["song"])

    bus.subscribe(topics.CMD_SOUND, on_sound)
    try:
        while not stop.is_set():
            try:
                song = songs.get(timeout=0.2)
            except queue.Empty:
                continue
            log.info("playing %r", song)
            speaker.play(song)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
