"""Recordings -> 2 s windows -> feature rows, for training.

The per-window math is in features_core.py, which also runs on the board.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import features_core
from features_core import AXES, dominant_freq  # noqa: F401  (re-exported)


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
    return features_core.peak(w, AXES.index(axis), sign)


def all_features(w: np.ndarray, rate_hz: float = config.RATE_HZ) -> dict[str, float]:
    """Model 2's inputs: ~50 numbers that summarize one window of all six IMU outputs."""
    return features_core.all_features(w, rate_hz)


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
