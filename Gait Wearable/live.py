"""Live classification of the IMU stream for dashboard.py: buffers samples for the graphs
and classifies with gait_models.py (the same code the board runs).

Pure logic (no MQTT, no plotting).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

import config
from gait_models import GaitModels, StreamClassifier


def load_models(models_dir: Path | None) -> GaitModels:
    path = Path(models_dir) / "gait_models.json" if models_dir is not None else None
    if path is not None and path.exists():
        return GaitModels.load(path)
    return GaitModels.peak_only(config.RATE_HZ, config.WINDOW_S, config.HOP_S,
                                config.PEAK_AXIS, config.PEAK_SIGN)


class LiveGait:
    def __init__(self, models_dir: Path | None = config.MODEL_DIR, rate_hz: int = config.RATE_HZ,
                 keep_s: float = 10, history_s: float = 120, smooth: int = 3):
        self.rate = rate_hz
        self.models = load_models(models_dir)
        self.stream = StreamClassifier(self.models, smooth)
        self.samples: deque[list[float]] = deque(maxlen=int(keep_s * rate_hz))  # for the graphs
        self.total = 0           # samples received (plus any lost), so total / rate = stream time
        self.lost = 0
        self._next_seq: int | None = None
        n_hist = int(history_s / config.HOP_S)
        self.peaks: deque[tuple[float, float]] = deque(maxlen=n_hist)        # (t, peak g)
        self.history: deque[tuple[float, str]] = deque(maxlen=n_hist)        # (t, smoothed gait)
        self.current: dict = {}  # latest result: gait, confidence, model1, model2, peak

    @property
    def has_models(self) -> bool:
        return self.models.has_models

    @property
    def now(self) -> float:
        return self.total / self.rate

    def add(self, samples: list[list[float]], seq: int | None = None) -> None:
        """Add a batch from TOPIC/imu. seq is the board's index of the first sample;
        a jump means batches were lost, which we count and skip over."""
        if seq is not None:
            if self._next_seq is not None and seq > self._next_seq:
                gap = seq - self._next_seq
                self.lost += gap
                self.total += gap
                self.samples.clear()  # don't glue windows across a gap
                self.stream.reset()
            self._next_seq = seq + len(samples)
        start = self.total
        self.samples.extend(samples)
        self.total += len(samples)
        for i, result in self.stream.add(samples):
            t = (start + i + 1) / self.rate
            self.peaks.append((t, result["peak"]))
            if "gait" in result:
                self.history.append((t, result["gait"]))
            self.current = result

    def recent(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """(times, samples) for the last `seconds`, samples as an (n, 6) array."""
        data = np.asarray(self.samples, dtype=float).reshape(-1, 6)
        data = data[-int(seconds * self.rate):]
        t = self.now - (len(data) - 1 - np.arange(len(data))) / self.rate
        return t, data


def parse_imu(payload: bytes) -> tuple[int | None, list[list[float]]]:
    msg = json.loads(payload)
    return msg.get("seq"), msg["s"]
