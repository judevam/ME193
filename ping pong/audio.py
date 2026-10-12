"""Music and sound effects (pygame-ce mixer: effects play over the music and each other).

    python audio.py              # list which sounds are found, then play each one

Files go in sounds/, named after the event, as .wav, .ogg or .mp3:
    hit  miss  whiff  serve  start  milestone      (record these with record_sfx.py)
    easy  medium  hard                             (announcer, on the start screen)
    missed  too_late  too_early  wrong_spot  no_hand   (announcer, after a miss)
    music1  music2  music3                         (songs for easy / medium / hard; they loop)
    music                                          (optional: start screen, and any level
                                                    whose song is missing)
Anything missing is skipped, so the game runs before every sound exists.

Extra takes of any sound (hit2.wav, hit3.wav, ... up to 9; record_sfx.py --take 2) are picked at
random each time it plays, so the announcer doesn't say exactly the same thing every miss.
"""

import os
import random
import time
from pathlib import Path

import config

SOUNDS_DIR = Path(__file__).resolve().parent / "sounds"
SFX_NAMES = ("hit", "miss", "whiff", "serve", "start", "milestone", "easy", "medium", "hard",
             "missed", "too_late", "too_early", "wrong_spot", "no_hand")
LEVEL_ANNOUNCE = {1: "easy", 2: "medium", 3: "hard"}
MISS_CALLS = {"late": "too_late", "early": "too_early", "lane": "wrong_spot", "no_hand": "no_hand"}
MAX_TAKES = 9
MUSIC_NAMES = ("music", "music1", "music2", "music3")
MUSIC_FADE_MS = 600


def music_for(level, available):
    """Which song to play: the level's own (music1/2/3), else the general "music", else None.
    level None = the start screen."""
    if level is not None and f"music{level}" in available:
        return f"music{level}"
    return "music" if "music" in available else None
EXTENSIONS = (".wav", ".ogg", ".mp3")


def find_sound(name, folder=SOUNDS_DIR):
    """Path of sounds/<name>.wav/.ogg/.mp3 (first one found, in that order), or None."""
    for ext in EXTENSIONS:
        path = Path(folder) / f"{name}{ext}"
        if path.exists():
            return path
    return None


def find_takes(name, folder=SOUNDS_DIR):
    """All takes of a sound: <name>.ext, then <name>2.ext ... <name>9.ext (whichever exist)."""
    takes = [find_sound(name, folder)] + [find_sound(f"{name}{n}", folder) for n in range(2, MAX_TAKES + 1)]
    return [p for p in takes if p is not None]


def miss_call(reason, available):
    """Which announcer line to use for a miss: the one for its reason if it's been recorded,
    else the general "missed", else None."""
    specific = MISS_CALLS.get(reason)
    if specific in available:
        return specific
    return "missed" if "missed" in available else None


def milestone_reached(old_streak, new_streak, every=config.STREAK_MILESTONE):
    """True when the streak has just gone up onto a multiple of `every` (5, 10, 15...)."""
    return every > 0 and new_streak > old_streak and new_streak % every == 0


class HoverAnnouncer:
    """Pure logic: which level (if any) to announce as the start tag moves between columns.
    Announces each time the tag moves into a different column; a brief dropout (the tag flickering
    out of detection for a frame) doesn't repeat it, but after the tag has been gone for
    `gone_s` the next column it appears in is announced again."""

    def __init__(self, gone_s=config.ANNOUNCE_GONE_S):
        self.gone_s = gone_s
        self._said = None        # level last announced
        self._last_seen = None   # when the tag was last in a column

    def update(self, level, now):
        """level: the column the tag is in this frame (None = no tag). Returns a level to announce."""
        if level is None:
            if self._last_seen is not None and now - self._last_seen >= self.gone_s:
                self._said = None
            return None
        self._last_seen = now
        if level != self._said:
            self._said = level
            return level
        return None

    def reset(self):
        self._said = self._last_seen = None


class Audio:
    def __init__(self, folder=SOUNDS_DIR):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self._pg = pygame
        pygame.mixer.pre_init(44100, -16, 2, 512)   # small buffer: effects play with little delay
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)
        pygame.mixer.set_reserved(1)                    # channel 0: the announcer only
        self._announcer = pygame.mixer.Channel(0)
        self.sounds = {}   # name -> list of takes
        for name in SFX_NAMES:
            takes = [pygame.mixer.Sound(str(p)) for p in find_takes(name, folder)]
            for sound in takes:
                sound.set_volume(config.SFX_VOLUME)
            if takes:
                self.sounds[name] = takes
        self._pending = []   # (time, name) announcements waiting to be said
        self.music = {n: p for n in MUSIC_NAMES if (p := find_sound(n, folder)) is not None}
        self._playing = None   # which song is on
        missing = [n for n in SFX_NAMES + MUSIC_NAMES if n not in self.sounds and n not in self.music]
        if missing:
            print(f"Audio: no sound yet for {', '.join(missing)} (put files in {folder})")

    def play(self, name):
        if name in self.sounds:
            random.choice(self.sounds[name]).play()

    def announce(self, name, delay=0.0):
        """Say it on the announcer's own channel (cutting off whatever it was saying), now or after
        `delay` seconds (call update() every frame for delayed ones)."""
        if name not in self.sounds:
            return
        if delay > 0:
            self._pending.append((time.perf_counter() + delay, name))
            return
        self._announcer.stop()
        self._announcer.play(random.choice(self.sounds[name]))

    def update(self):
        """Say any delayed announcements that are due."""
        now = time.perf_counter()
        due = [name for t, name in self._pending if t <= now]
        self._pending = [(t, name) for t, name in self._pending if t > now]
        for name in due:
            self.announce(name)

    def play_music(self, level, volume):
        """Loop the song for `level` (None = start screen) at `volume`. Starting a different song
        fades it in; asking for the one already playing just sets the volume."""
        song = music_for(level, self.music)
        if song is None:
            self._pg.mixer.music.stop()
            self._playing = None
            return
        if song != self._playing:
            self._pg.mixer.music.load(str(self.music[song]))
            self._pg.mixer.music.play(loops=-1, fade_ms=MUSIC_FADE_MS)
            self._playing = song
        self._pg.mixer.music.set_volume(volume)

    def close(self):
        self._pg.mixer.quit()


class NoAudio:
    """Stand-in for --no-audio (or if the sound device can't be opened)."""

    sounds, music = {}, {}

    def play(self, name):
        pass

    def announce(self, name, delay=0.0):
        pass

    def update(self):
        pass

    def play_music(self, level, volume):
        pass

    def close(self):
        pass


def open_audio():
    try:
        return Audio()
    except Exception as e:   # no sound device, broken file...: play on in silence
        print(f"Audio off ({e})")
        return NoAudio()


def main():
    audio = open_audio()
    for name in SFX_NAMES:
        path = find_sound(name)
        print(f"{name:10s} {path.name if path else '-- missing (record it: python record_sfx.py ' + name + ')'}")
        if name in audio.sounds:
            audio.play(name)
            time.sleep(audio.sounds[name][0].get_length() + 0.3)
    for level, label in ((None, "music"), (1, "music1"), (2, "music2"), (3, "music3")):
        path = audio.music.get(label)
        print(f"{label:10s} {path.name if path else '-- missing (put ' + label + '.mp3 in sounds/)'}")
        if path:
            audio.play_music(level, config.MUSIC_VOLUME_MENU)
            time.sleep(4)
    audio.close()


if __name__ == "__main__":
    main()
