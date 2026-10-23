"""Recordings -> 2 s windows -> feature rows. Shared by training and (later) the board.

Pure numpy and pandas: no hardware, no MQTT.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]


def load_session(path: str | Path) -> pd.DataFrame:
    """One recording CSV -> DataFrame with columns t, ax..gz. Label comes from the file name."""
    return pd.read_csv(path, comment="#")


def label_of(path: str | Path) -> str:
    """'walking_20261022-141500.csv' -> 'walking'"""
    return Path(path).stem.split("_")[0]


def windows(df: pd.DataFrame, rate_hz: float = config.RATE_HZ, window_s: float = config.WINDOW_S,
            hop_s: float = config.HOP_S, trim_s: float = config.TRIM_S) -> list[np.ndarray]:
    """Cut a recording into overlapping windows, each an (n, 6) array of ax..gz.
    The first and last trim_s seconds are dropped (getting started / stopping)."""
    data = df[AXES].to_numpy(dtype=float)
    trim, size, hop = int(trim_s * rate_hz), int(window_s * rate_hz), int(hop_s * rate_hz)
    data = data[trim:len(data) - trim] if trim else data
    return [data[i:i + size] for i in range(0, len(data) - size + 1, hop)]


def peak(w: np.ndarray, axis: str = config.PEAK_AXIS, sign: int = config.PEAK_SIGN) -> float:
    """Model 1's only input: the peak acceleration along one axis, in g."""
    return float(np.max(sign * w[:, AXES.index(axis)]))


def dominant_freq(x: np.ndarray, rate_hz: float, lo: float = 0.3, hi: float = 5.0) -> tuple[float, float]:
    """(frequency in Hz, share of the signal's power at that frequency) within lo..hi.
    This is the step or pedal cadence."""
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate_hz)
    band = (freqs >= lo) & (freqs <= hi)
    total = power[1:].sum()
    if not band.any() or total <= 0:
        return 0.0, 0.0
    i = np.argmax(np.where(band, power, 0))
    return float(freqs[i]), float(power[i] / total)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def all_features(w: np.ndarray, rate_hz: float = config.RATE_HZ) -> dict[str, float]:
    """Model 2's inputs: ~50 numbers that summarize one window of all six IMU outputs."""
    ax, ay, az, gx, gy, gz = w.T
    signals = {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz,
               "amag": np.sqrt(ax**2 + ay**2 + az**2),
               "gmag": np.sqrt(gx**2 + gy**2 + gz**2)}
    f: dict[str, float] = {}
    for name, x in signals.items():
        f[f"{name}_mean"] = float(x.mean())
        f[f"{name}_std"] = float(x.std())
        f[f"{name}_min"] = float(x.min())
        f[f"{name}_max"] = float(x.max())
        f[f"{name}_range"] = float(np.ptp(x))
    for name in ("ax", "amag", "gmag"):
        f[f"{name}_freq"], f[f"{name}_freq_power"] = dominant_freq(signals[name], rate_hz)
    # Orientation from gravity (the window's mean acceleration). With X along the thigh:
    # standing -> pitch near +/-90 deg, biking or lying -> much flatter.
    mx, my, mz = ax.mean(), ay.mean(), az.mean()
    f["pitch_deg"] = float(np.degrees(np.arctan2(mx, np.hypot(my, mz))))
    f["roll_deg"] = float(np.degrees(np.arctan2(my, mz)))
    f["corr_xy"], f["corr_xz"], f["corr_yz"] = _corr(ax, ay), _corr(ax, az), _corr(ay, az)
    return f


def build_dataset(paths: list[Path]) -> pd.DataFrame:
    """Every window of every recording -> one row each, with label, session and peak."""
    rows = []
    for path in paths:
        label = label_of(path)
        for i, w in enumerate(windows(load_session(path))):
            rows.append({"session": Path(path).stem, "label": label, "window": i,
                         "peak": peak(w), **all_features(w)})
    return pd.DataFrame(rows)


FEATURE_COLUMNS = list(all_features(np.random.default_rng(0).normal(size=(200, 6))).keys())
