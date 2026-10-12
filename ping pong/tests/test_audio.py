"""Sound clean-up (record_sfx), file lookup and milestones (audio). No speakers or mic needed."""

import numpy as np
import pytest

from audio import (HoverAnnouncer, NoAudio, find_sound, find_takes, milestone_reached, miss_call,
                   music_for)
from record_sfx import KEEP_AFTER_MS, KEEP_BEFORE_MS, RATE, clean, fade, normalize, pick_mic, trim_silence


def blip(silence_before=0.5, sound=0.2, silence_after=0.5, noise=0.003):
    """Quiet room noise, a loud tone, quiet room noise."""
    rng = np.random.default_rng(0)
    n0, n1, n2 = (int(RATE * s) for s in (silence_before, sound, silence_after))
    x = rng.normal(0, noise, n0 + n1 + n2).astype(np.float32)
    t = np.arange(n1) / RATE
    x[n0:n0 + n1] += 0.5 * np.sin(2 * np.pi * 800 * t)
    return x


# ── record_sfx clean-up ──────────────────────────────────────────────────────

def test_trim_cuts_the_silence_but_keeps_the_sound():
    out = trim_silence(blip())
    margins = (KEEP_BEFORE_MS + KEEP_AFTER_MS) / 1000
    assert 0.2 <= len(out) / RATE <= 0.2 + margins + 0.02       # sound + the margins
    assert len(out) < len(blip())


def test_trim_keeps_a_little_before_the_sound():
    out = trim_silence(blip(), keep_before_ms=50)
    assert np.abs(out[: int(RATE * 0.04)]).max() < 0.05          # starts with a bit of quiet


def test_trim_keeps_a_fading_tail():
    """A sound that dies away slowly (like a 'pock' ringing out) keeps most of its decay."""
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.002, int(RATE * 0.5)).astype(np.float32)
    t = np.arange(int(RATE * 0.6)) / RATE
    ring = (0.6 * np.exp(-t / 0.12) * np.sin(2 * np.pi * 600 * t)).astype(np.float32)   # 600 ms decay
    x = np.concatenate([quiet, ring, quiet])
    out = trim_silence(x)
    assert len(out) / RATE > 0.5                                  # nearly all of the ring kept


def test_trim_keeps_a_soft_start():
    """A breathy 'fff' that swells up before the loud part isn't chopped."""
    rng = np.random.default_rng(2)
    quiet = rng.normal(0, 0.002, int(RATE * 0.5)).astype(np.float32)
    swell = (rng.normal(0, 1, int(RATE * 0.15)) * np.linspace(0.01, 0.1, int(RATE * 0.15))).astype(np.float32)
    loud = (0.5 * np.sin(2 * np.pi * 500 * np.arange(int(RATE * 0.1)) / RATE)).astype(np.float32)
    x = np.concatenate([quiet, swell, loud, quiet])
    out = trim_silence(x)
    assert len(out) / RATE > 0.25                                 # swell + loud part kept


def test_trim_of_silence_is_empty():
    assert len(trim_silence(np.zeros(1000, np.float32))) == 0


def test_fade_starts_and_ends_at_zero():
    x = np.ones(RATE // 5, np.float32)
    out = fade(x, in_ms=5, out_ms=40)
    assert out[0] == 0 and out[-1] == 0
    assert out[len(out) // 2] == 1                                 # middle untouched
    assert out[-int(RATE * 0.02)] > 0.3                            # fade-out is gradual, not a cut


def test_no_trim_keeps_everything():
    x = blip()
    assert len(clean(x, trim=False)) == len(x)


def test_pick_mic_by_name_or_number():
    devices = [{"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
               {"name": "AudioTechnica Mic (Realtek(R) A", "max_input_channels": 2},
               {"name": "Speakers (Realtek(R) Audio)", "max_input_channels": 0}]
    assert pick_mic(devices, "audiotechnica") == 1
    assert pick_mic(devices, "Speakers") is None                  # outputs don't count
    assert pick_mic(devices, "3") == 3
    assert pick_mic(devices, "nope") is None


def test_normalize_sets_the_peak():
    out = normalize(np.array([0.1, -0.2, 0.05], np.float32), peak=0.89)
    assert np.abs(out).max() == pytest.approx(0.89)


def test_clean_whole_chain():
    out = clean(blip())
    assert np.abs(out).max() == pytest.approx(0.89, abs=1e-3)
    assert abs(out[0]) < 1e-6 and abs(out[-1]) < 1e-6
    assert len(out) < len(blip())


# ── audio ────────────────────────────────────────────────────────────────────

def test_find_sound_any_extension(tmp_path):
    assert find_sound("hit", tmp_path) is None
    (tmp_path / "hit.ogg").write_bytes(b"")
    assert find_sound("hit", tmp_path).name == "hit.ogg"
    (tmp_path / "hit.wav").write_bytes(b"")
    assert find_sound("hit", tmp_path).name == "hit.wav"          # .wav preferred


@pytest.mark.parametrize("old, new, expected", [
    (4, 5, True), (9, 10, True), (5, 6, False), (3, 4, False),
    (5, 0, False),   # a miss back to 0 isn't a milestone
    (5, 5, False),   # no change
])
def test_milestones_every_five(old, new, expected):
    assert milestone_reached(old, new, every=5) is expected


def test_no_audio_does_nothing():
    a = NoAudio()
    a.play_music(2, 0.5)
    a.play("hit")
    a.close()


def test_real_mixer_plays_a_recorded_effect(tmp_path, monkeypatch):
    """Save a cleaned sound the way record_sfx does, then load it through Audio (silent driver)."""
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from scipy.io import wavfile

    import audio

    wavfile.write(tmp_path / "hit.wav", RATE, (clean(blip()) * 32767).astype(np.int16))
    a = audio.Audio(folder=tmp_path)
    try:
        assert set(a.sounds) == {"hit"} and a.music == {}
        a.play("hit")
        a.announce("hit")       # announcer channel
        a.announce("hit")       # cuts itself off and starts again
        a.announce("easy")      # missing: skipped quietly
        a.play("miss")          # missing: skipped quietly
        a.play_music(1, 0.5)    # no music file: skipped quietly
    finally:
        a.close()


# ── Start-screen announcer ───────────────────────────────────────────────────


def frames(announcer, levels, dt=1 / 30, t0=0.0):
    """Feed one level per camera frame; return what got announced, in order."""
    said = []
    for i, level in enumerate(levels):
        out = announcer.update(level, t0 + i * dt)
        if out:
            said.append(out)
    return said


def test_announces_the_column_the_tag_enters():
    assert frames(HoverAnnouncer(gone_s=0.6), [None, None, 1, 1, 1]) == [1]


def test_hovering_does_not_repeat():
    assert frames(HoverAnnouncer(gone_s=0.6), [2] * 60) == [2]


def test_flicker_does_not_repeat():
    # detection drops the tag for a frame or two now and then
    assert frames(HoverAnnouncer(gone_s=0.6), [3, 3, None, 3, None, None, 3, 3]) == [3]


def test_moving_between_columns_announces_each():
    assert frames(HoverAnnouncer(gone_s=0.6), [1, 1, 2, 2, 3, 3, 2]) == [1, 2, 3, 2]


def test_same_column_again_after_the_tag_was_put_away():
    levels = [1] * 5 + [None] * 25 + [1] * 5     # ~0.8 s gone
    assert frames(HoverAnnouncer(gone_s=0.6), levels) == [1, 1]


def test_reset_announces_again():
    a = HoverAnnouncer()
    assert a.update(2, 0.0) == 2
    a.reset()
    assert a.update(2, 0.1) == 2


# ── Miss calls and extra takes ───────────────────────────────────────────────

@pytest.mark.parametrize("reason, line", [
    ("late", "too_late"), ("early", "too_early"), ("lane", "wrong_spot"), ("no_hand", "no_hand"),
])
def test_miss_call_uses_the_line_for_the_reason(reason, line):
    everything = {"missed", "too_late", "too_early", "wrong_spot", "no_hand"}
    assert miss_call(reason, everything) == line


def test_miss_call_falls_back_to_missed():
    assert miss_call("lane", {"missed", "too_late"}) == "missed"


def test_miss_call_with_nothing_recorded():
    assert miss_call("late", {"hit", "miss"}) is None


def test_find_takes_in_order(tmp_path):
    for name in ("missed.wav", "missed3.wav", "missed2.ogg", "missedx.wav", "hit.wav"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_takes("missed", tmp_path)] == ["missed.wav", "missed2.ogg", "missed3.wav"]


def test_delayed_announcement_waits_then_plays(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import time
    from scipy.io import wavfile

    import audio

    for name in ("missed.wav", "missed2.wav"):
        wavfile.write(tmp_path / name, RATE, (clean(blip()) * 32767).astype(np.int16))
    a = audio.Audio(folder=tmp_path)
    try:
        assert len(a.sounds["missed"]) == 2                     # both takes loaded
        a.announce("missed", delay=0.2)
        a.update()
        assert len(a._pending) == 1                             # not yet
        time.sleep(0.25)
        a.update()
        assert a._pending == []                                 # said
    finally:
        a.close()


# ── A song per level ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("level, song", [(1, "music1"), (2, "music2"), (3, "music3"), (None, "music")])
def test_each_level_has_its_song(level, song):
    assert music_for(level, {"music", "music1", "music2", "music3"}) == song


def test_missing_level_song_falls_back_to_general_music():
    assert music_for(3, {"music", "music1"}) == "music"


def test_no_start_screen_song_is_silence():
    assert music_for(None, {"music1", "music2", "music3"}) is None


def test_switching_songs(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from scipy.io import wavfile

    import audio

    for name in ("music1.wav", "music2.wav"):
        wavfile.write(tmp_path / name, RATE, (clean(blip()) * 32767).astype(np.int16))
    a = audio.Audio(folder=tmp_path)
    try:
        a.play_music(None, 0.6)                      # start screen: no song for it
        assert a._playing is None
        a.play_music(1, 0.25)
        assert a._playing == "music1"
        a.play_music(1, 0.6)                         # same song: only the volume changes
        assert a._playing == "music1"
        a.play_music(2, 0.25)
        assert a._playing == "music2"
        a.play_music(3, 0.25)                        # no music3 and no general music: silence
        assert a._playing is None
    finally:
        a.close()
