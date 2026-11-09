"""Synthesize a quiet, original heist-style ambient track (no samples, no licensing).

  python movie/make_music.py [seconds]      -> movie/out/music.wav
Layers: slow pad chords (Am F C G), a low drone, a soft heartbeat pulse and a gentle plucked arpeggio.
"""
import os
import sys
import wave

import numpy as np

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 450.0
N = int(dur * SR)
t_all = np.arange(N, dtype=np.float32) / SR
rng = np.random.RandomState(4)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# chord roots / tones (MIDI): Am, F, C, G
CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
ROOTS = [33, 29, 36, 31]
SEG = 16.0
XF = 3.0


def smooth_env(n, fade_in, fade_out):
    e = np.ones(n, np.float32)
    a, b = int(fade_in * SR), int(fade_out * SR)
    e[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    e[-b:] = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, b))
    return e


L = np.zeros(N, np.float32)
R = np.zeros(N, np.float32)

# ---- pad + drone, one segment per chord, overlapped
k = 0
start = 0.0
while start < dur:
    chord = CHORDS[k % 4]
    n = int((SEG + XF) * SR)
    i0 = int(start * SR)
    n = min(n, N - i0)
    if n <= 0:
        break
    t = t_all[:n]
    env = smooth_env(n, XF, XF)
    pad_l = np.zeros(n, np.float32)
    pad_r = np.zeros(n, np.float32)
    for m in chord:
        for det, side in ((-0.15, 0), (0.15, 1)):
            f = hz(m) * 2 ** (det / 12)
            tone = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t + 0.5)
                    + 0.12 * np.sin(2 * np.pi * 3 * f * t))
            (pad_l if side == 0 else pad_r)[:] += tone.astype(np.float32)
    swell = 0.75 + 0.25 * np.sin(2 * np.pi * t / 8.0 + k)
    drone = np.sin(2 * np.pi * hz(ROOTS[k % 4]) * t) * 1.3 + 0.4 * np.sin(2 * np.pi * 2 * hz(ROOTS[k % 4]) * t)
    L[i0:i0 + n] += (0.045 * pad_l * swell + 0.07 * drone.astype(np.float32)) * env
    R[i0:i0 + n] += (0.045 * pad_r * swell + 0.07 * drone.astype(np.float32)) * env
    k += 1
    start += SEG

# ---- soft heartbeat pulse (72 bpm)
beat = 60.0 / 72
tt = np.arange(int(0.35 * SR), dtype=np.float32) / SR
thump = np.sin(2 * np.pi * (52 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 14)
tick = rng.randn(int(0.05 * SR)).astype(np.float32) * np.exp(-np.arange(int(0.05 * SR)) / SR * 90)
tick = np.diff(tick, prepend=0)  # crude high-pass
b = 0.0
while b < dur - 0.4:
    i0 = int(b * SR)
    n = min(len(thump), N - i0)
    L[i0:i0 + n] += 0.16 * thump[:n]
    R[i0:i0 + n] += 0.16 * thump[:n]
    j0 = int((b + beat / 2) * SR)
    n2 = min(len(tick), N - j0)
    if n2 > 0:
        L[j0:j0 + n2] += 0.05 * tick[:n2]
        R[j0:j0 + n2] += 0.05 * tick[:n2]
    b += beat

# ---- plucked arpeggio (starts after 8 s, gentle)
step = beat / 2
pattern = [0, 2, 1, 2, 0, 1, 2, 1]
tp = np.arange(int(1.4 * SR), dtype=np.float32) / SR
q = 0
a = 8.0
while a < dur - 1.5:
    chord = CHORDS[int(a // SEG) % 4]
    m = chord[pattern[q % len(pattern)]] + 24
    f = hz(m)
    pluck = (np.sin(2 * np.pi * f * tp) + 0.3 * np.sin(2 * np.pi * 2 * f * tp)) * np.exp(-tp * 5.0)
    i0 = int(a * SR)
    n = min(len(pluck), N - i0)
    pan = 0.5 + 0.35 * np.sin(q * 0.7)
    vol = 0.05 * (0.7 + 0.3 * np.sin(a / 11.0))
    L[i0:i0 + n] += vol * (1 - pan) * 1.6 * pluck[:n]
    R[i0:i0 + n] += vol * pan * 1.6 * pluck[:n]
    q += 1
    a += step

# ---- master: fade in/out, gentle limit, quiet level
fade_in, fade_out = int(3 * SR), int(5 * SR)
env = np.ones(N, np.float32)
env[:fade_in] = np.linspace(0, 1, fade_in)
env[-fade_out:] = np.linspace(1, 0, fade_out)
stereo = np.stack([L * env, R * env], axis=1)
stereo = np.tanh(stereo * 1.6) / 1.6
stereo *= 0.30 / max(1e-6, np.abs(stereo).max())  # peak at 0.30: quiet under a voice
pcm = (stereo * 32767).astype(np.int16)
out = os.path.join(HERE, "out", "music.wav")
os.makedirs(os.path.dirname(out), exist_ok=True)
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("saved", out, f"{dur:.0f}s")
