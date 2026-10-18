"""Laptop speaker output (the fallback; the UNO Q buzzer is the main one)."""
from __future__ import annotations

import logging
import sys
import time

from common.sounds import SONGS
from hardware.interfaces import SoundOutput

log = logging.getLogger("sound")


class LaptopSpeaker(SoundOutput):
    """Plays tones with winsound.Beep on Windows; prints a warning elsewhere."""

    def play(self, song: str) -> None:
        notes = SONGS.get(song)
        if notes is None:
            log.warning("unknown song %r", song)
            return
        if sys.platform != "win32":
            log.warning("laptop tones only work on Windows; would play %r", song)
            return
        import winsound
        for freq, ms in notes:
            if freq:
                winsound.Beep(freq, ms)
            else:
                time.sleep(ms / 1000)
