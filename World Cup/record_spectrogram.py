"""Live spectrogram of the default microphone, with both frequency (Hz) and
musical note labels on the y-axis. Scrolls in real time instead of recording
a fixed clip first.

Notes are computed with standard equal-temperament tuning (A4 = 440 Hz):
    note_number = 12 * log2(f / 440) + 69   (MIDI number, A4 = 69)
which is inverted to place a labeled tick at each note's exact frequency.

Press 'q' (with the plot window focused) to quit.
"""

import numpy as np
import pyaudio
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

RATE = 44100          # sample rate, Hz
CHANNELS = 1
FORMAT = pyaudio.paInt16
WINDOW_SECONDS = 5      # how much history is shown at once, scrolling
UPDATE_INTERVAL_MS = 100  # how often the plot redraws

FREQ_MIN = 50          # lowest frequency shown, Hz -- below this is mostly rumble
FREQ_MAX = 5000         # highest frequency shown, Hz -- covers the musical range
                         # most instruments/voices sit in

# Anything quieter than this (dB) is treated as silence -- for the "current
# note" readout and for breaking up the pitch-track line, so both stop
# flickering between whatever bin has the least noise when nothing's playing.
SILENCE_FLOOR_DB = -60

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note_name(freq: float) -> str:
    """Nearest note name + octave for freq, e.g. 440.0 -> 'A4'."""
    midi = round(12 * np.log2(freq / 440.0) + 69)
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def c_note_frequencies(freq_min: float, freq_max: float) -> list[tuple[float, str]]:
    """(frequency, 'C<octave>') for every C note between freq_min and freq_max.

    Used as the y-axis reference grid instead of all 12 semitones -- one
    label per octave reads clearly, where all ~80 semitone labels in this
    range would overlap into an unreadable smear.
    """
    midi_min = int(np.ceil(12 * np.log2(freq_min / 440.0) + 69))
    midi_max = int(np.floor(12 * np.log2(freq_max / 440.0) + 69))
    notes = []
    for midi in range(midi_min, midi_max + 1):
        if midi % 12 != 0:  # 0 == C in MIDI-mod-12 space (C uses midi % 12 == 0)
            continue
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        octave = midi // 12 - 1
        notes.append((freq, f"C{octave}"))
    return notes


def main():
    plt.style.use("dark_background")

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    buffer = np.zeros(int(RATE * WINDOW_SECONDS), dtype=np.float32)
    samples_per_update = int(RATE * UPDATE_INTERVAL_MS / 1000)

    # Spectrogram geometry (freqs/times bin counts) stays fixed every frame
    # since buffer length never changes -- compute it once so the plot and
    # the mesh's flattened array shape line up.
    freqs, times, Sxx = signal.spectrogram(buffer, fs=RATE, nperseg=2048, noverlap=1536, scaling="spectrum")
    band = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    freqs = freqs[band]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    mesh = ax.pcolormesh(
        times, freqs, np.full((len(freqs), len(times)), SILENCE_FLOOR_DB),
        shading="gouraud", cmap="magma",
    )
    ax.set_yscale("log")
    ax.set_ylim(FREQ_MIN, FREQ_MAX)
    ax.set_xlabel(f"rolling {WINDOW_SECONDS:.0f}-second window")
    ax.set_ylabel("Frequency (Hz)")
    fig.suptitle("Live Spectrogram", fontsize=16, fontweight="bold")
    ax.set_title("press 'q' to quit", fontsize=9, color="gray")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")

    # Light reference gridlines at each octave (C notes) instead of labeling
    # every semitone -- easy to scan visually without cluttering the axis.
    c_notes = c_note_frequencies(FREQ_MIN, FREQ_MAX)
    for freq, _name in c_notes:
        ax.axhline(freq, color="white", alpha=0.15, linewidth=1, linestyle="--")
    note_ax = ax.secondary_yaxis("right")
    note_ax.set_yscale("log")
    note_ax.set_ylim(FREQ_MIN, FREQ_MAX)
    note_ax.set_yticks([f for f, _ in c_notes])
    note_ax.set_yticklabels([n for _, n in c_notes], fontsize=10)
    note_ax.set_ylabel("Note")

    # Rolling dominant-pitch overlay, plus a big current-note readout since
    # that's easier to read at a glance than the axis labels while something
    # is actually playing.
    (pitch_line,) = ax.plot([], [], color="cyan", linewidth=2, alpha=0.9)
    current_note_text = ax.text(
        0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top",
        color="cyan", fontsize=28, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5, edgecolor="none"),
    )
    fig.tight_layout()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal buffer
        data = stream.read(samples_per_update, exception_on_overflow=False)
        new_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        buffer = np.concatenate([buffer, new_samples])[-len(buffer):]

        _freqs, _times, Sxx = signal.spectrogram(buffer, fs=RATE, nperseg=2048, noverlap=1536, scaling="spectrum")
        Sxx_db = 10 * np.log10(Sxx[band, :] + 1e-12)
        mesh.set_array(Sxx_db.ravel())
        # Auto-scale contrast to whatever's actually coming in, instead of a
        # fixed dB range that looks washed-out on a quiet mic or blown-out on
        # a loud one.
        mesh.set_clim(np.percentile(Sxx_db, 20), max(np.percentile(Sxx_db, 99.5), SILENCE_FLOOR_DB + 1))

        dominant_idx = np.argmax(Sxx_db, axis=0)
        dominant_freq = freqs[dominant_idx]
        dominant_power = Sxx_db[dominant_idx, np.arange(len(dominant_idx))]
        # Break the line instead of drawing it through silence, where the
        # "loudest" bin is really just noise.
        pitch_line.set_data(times, np.where(dominant_power > SILENCE_FLOOR_DB, dominant_freq, np.nan))

        latest_power = dominant_power[-1]
        current_note_text.set_text(
            freq_to_note_name(dominant_freq[-1]) if latest_power > SILENCE_FLOOR_DB else ""
        )

        return mesh, pitch_line, current_note_text

    ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
