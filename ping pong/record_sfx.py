"""Record your own sound effects for the game (headset mic: config.RECORD_MIC).

    python record_sfx.py --list          # which effects exist / are still missing
    python record_sfx.py hit             # record the hit sound
    python record_sfx.py miss --seconds 3
    python record_sfx.py hit --plot      # also show its waveform and spectrogram
    python record_sfx.py whiff --tail 500    # keep more of the fade-out (ms)
    python record_sfx.py start --no-trim     # keep the whole recording
    python record_sfx.py hit --device 3      # a different mic (number or part of its name)
    python record_sfx.py missed --take 2     # an extra take; the game picks one at random

It counts down, records, cuts the silence off both ends, fades the edges (no clicks), levels the
volume, plays it back and asks whether to keep it. Kept sounds are saved to sounds/<name>.wav,
which is where the game (audio.py) looks for them. Record again any time to replace one.

Trimming is gentle: it measures the room's background noise and only cuts what is down at that
level, keeping 50 ms before the sound starts and 300 ms after it fades (--tail), then fades out
over 40 ms so the natural decay isn't chopped off.
"""

import argparse
import time

import numpy as np

import config
from audio import MAX_TAKES, SFX_NAMES, SOUNDS_DIR, find_sound, find_takes

RATE = 44100
IDEAS = {
    "hit": "a short sharp 'pock': click your tongue, tap a pen on the desk, flick a cup",
    "miss": "something sad or deflating: a low 'bwomp', a groan, a slide whistle down",
    "whiff": "a quick 'whoosh' or 'fft' (air through your teeth)",
    "serve": "a light tap or 'tik' as the ball leaves the wall",
    "start": "a 'ready... go!' or a whistle",
    "milestone": "a cheer, 'nice!', or a little ta-da",
    "easy": "announcer voice: 'Easy!' (start screen, tag in the left column)",
    "medium": "announcer voice: 'Medium!' (tag in the middle column)",
    "hard": "announcer voice: 'Hard!' - make it dramatic (tag in the right column)",
    "missed": "announcer, any miss: 'Ohh, missed it!' (used when there's no line for the reason)",
    "too_late": "announcer, never swung in time: 'Too slow!'",
    "too_early": "announcer, swung too early: 'Wait for it!'",
    "wrong_spot": "announcer, paddle in the wrong lane: 'Wrong side!'",
    "no_hand": "announcer, camera lost your hand: 'Where'd you go?'",
}


# ── Clean-up (pure functions, tested) ────────────────────────────────────────

FRAME_MS = 10           # loudness is measured in 10 ms slices
KEEP_BEFORE_MS = 50     # kept before the sound starts (soft attacks: "f", breaths)
KEEP_AFTER_MS = 300     # kept after it fades below the threshold (the natural decay)
ABOVE_NOISE = 3.0       # a slice counts as sound once it's this many times louder than the room


def loudness(x, rate=RATE, frame_ms=FRAME_MS):
    """RMS loudness of each FRAME_MS slice."""
    n = max(1, int(rate * frame_ms / 1000))
    frames = len(x) // n
    if frames == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x[:frames * n].reshape(frames, n) ** 2, axis=1))


def trim_silence(x, rate=RATE, keep_before_ms=KEEP_BEFORE_MS, keep_after_ms=KEEP_AFTER_MS,
                 above_noise=ABOVE_NOISE):
    """Cut the silence off both ends, gently. The room's background noise is the loudness of the
    quietest slices; anything clearly above it (above_noise x) is sound and kept, plus a margin
    before and a longer one after, so soft starts and fade-outs survive."""
    level = loudness(x, rate)
    if len(level) == 0 or level.max() == 0:
        return x[:0]
    noise = np.percentile(level, 20)
    # well above the room noise, but never more than 5% of the loudest slice (so quiet tails count)
    threshold = min(max(noise * above_noise, level.max() * 0.005), level.max() * 0.05)
    loud = np.nonzero(level > threshold)[0]
    n = int(rate * FRAME_MS / 1000)
    start = max(0, loud[0] * n - int(rate * keep_before_ms / 1000))
    end = min(len(x), (loud[-1] + 1) * n + int(rate * keep_after_ms / 1000))
    return x[start:end]


def fade(x, rate=RATE, in_ms=5, out_ms=40):
    """Fade in/out so the sound doesn't start or end with a click; the longer fade-out lets the
    tail die away naturally."""
    x = x.copy()
    n_in = min(int(rate * in_ms / 1000), len(x) // 2)
    n_out = min(int(rate * out_ms / 1000), len(x) // 2)
    if n_in:
        x[:n_in] *= np.linspace(0.0, 1.0, n_in)
    if n_out:
        x[-n_out:] *= np.linspace(1.0, 0.0, n_out)
    return x


def normalize(x, peak=0.89):
    """Scale so the loudest sample is `peak` (0.89 = -1 dB: loud, no clipping)."""
    top = np.abs(x).max() if len(x) else 0.0
    return x if top == 0 else x * (peak / top)


def clean(x, rate=RATE, trim=True, keep_after_ms=KEEP_AFTER_MS):
    if trim:
        x = trim_silence(x, rate, keep_after_ms=keep_after_ms)
    return normalize(fade(x, rate))


def pick_mic(devices, wanted):
    """Index of the first input device whose name contains `wanted` (case-insensitive), else None.
    `wanted` may also be a device number."""
    if wanted is None:
        return None
    if str(wanted).isdigit():
        return int(wanted)
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and str(wanted).lower() in d["name"].lower():
            return i
    return None


# ── Recording ────────────────────────────────────────────────────────────────

def choose_mic(wanted):
    import sounddevice as sd

    devices = sd.query_devices()
    index = pick_mic(devices, wanted)
    if index is None:
        index = sd.default.device[0]
        print(f"  Mic '{wanted}' not found - using the default: {devices[index]['name']}")
    else:
        print(f"  Mic: {devices[index]['name']}")
    return index


def record(seconds, device=None):
    import sounddevice as sd

    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(0.7)
    print("  RECORDING - go!")
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    print("  done")
    return audio[:, 0]


def play(x):
    import sounddevice as sd

    sd.play(x, RATE)
    sd.wait()


def save(name, x, take=1):
    from scipy.io import wavfile

    SOUNDS_DIR.mkdir(exist_ok=True)
    path = SOUNDS_DIR / (f"{name}.wav" if take == 1 else f"{name}{take}.wav")
    wavfile.write(path, RATE, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return path


def show(name, raw, cleaned):
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(3, 1, figsize=(9, 7))
    a.plot(np.arange(len(raw)) / RATE, raw, lw=0.6)
    a.set_title(f"{name}: raw recording")
    b.plot(np.arange(len(cleaned)) / RATE, cleaned, lw=0.6, color="tab:orange")
    b.set_title("cleaned (trimmed, faded, levelled)")
    c.specgram(cleaned, Fs=RATE, NFFT=512, noverlap=384, cmap="magma")
    c.set_title("spectrogram")
    c.set_ylabel("Hz")
    c.set_xlabel("seconds")
    fig.tight_layout()
    plt.show()


def list_sounds():
    for name in SFX_NAMES:
        takes = find_takes(name)
        found = ", ".join(p.name for p in takes)
        print(f"  {name:10s} {found if takes else 'missing  - idea: ' + IDEAS[name]}")
    for name, what in (("music1", "easy"), ("music2", "medium"), ("music3", "hard"),
                       ("music", "start screen, optional")):
        song = find_sound(name)
        print(f"  {name:10s} {song.name if song else f'missing  - your {what} song as sounds/{name}.mp3'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", choices=SFX_NAMES, help="which effect to record")
    parser.add_argument("--list", action="store_true", help="show which sounds exist")
    parser.add_argument("--seconds", type=float, default=2.0, help="recording length (it gets trimmed)")
    parser.add_argument("--device", default=config.RECORD_MIC,
                        help="microphone: part of its name or its number (python -m sounddevice)")
    parser.add_argument("--plot", action="store_true", help="show waveform + spectrogram")
    parser.add_argument("--tail", type=int, default=KEEP_AFTER_MS, help="ms kept after the sound fades")
    parser.add_argument("--no-trim", action="store_true", help="keep the whole recording")
    parser.add_argument("--take", type=int, default=1, choices=range(1, MAX_TAKES + 1),
                        help="save as an extra take (2 -> <name>2.wav); takes are picked at random")
    args = parser.parse_args()

    if args.list or not args.name:
        list_sounds()
        if not args.name:
            print("\nRecord one with: python record_sfx.py <name>")
        return

    print(f"Recording '{args.name}' - idea: {IDEAS[args.name]}")
    mic = choose_mic(args.device)
    while True:
        raw = record(args.seconds, mic)
        if np.abs(raw).max() < 0.02:
            print("  That was nearly silent - is the mic on? Get closer and try again.")
            if input("  Enter = record again, q = quit: ").strip().lower() == "q":
                return
            continue
        cleaned = clean(raw, trim=not args.no_trim, keep_after_ms=args.tail)
        print(f"  {len(cleaned) / RATE:.2f} s after trimming. Playing it back...")
        play(cleaned)
        if args.plot:
            show(args.name, raw, cleaned)
        answer = input("  Keep it? [y]es / [n]o, record again / [p]lay again / [q]uit: ").strip().lower()
        while answer == "p":
            play(cleaned)
            answer = input("  Keep it? [y/n/p/q]: ").strip().lower()
        if answer == "y":
            print(f"  Saved {save(args.name, cleaned, args.take)}")
            return
        if answer == "q":
            return


if __name__ == "__main__":
    main()
