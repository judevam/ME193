"""Live classification of the IMU stream: buffer samples, classify every HOP_S.

Pure logic (no MQTT, no plotting), used by dashboard.py and later the board.
"""
from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np

import config
from features import all_features, peak
from model_peak import PeakThresholdModel


class LiveGait:
    def __init__(self, models_dir: Path | None = config.MODEL_DIR, rate_hz: int = config.RATE_HZ,
                 keep_s: float = 10, history_s: float = 120, smooth: int = 3):
        self.rate = rate_hz
        self.window_n = int(config.WINDOW_S * rate_hz)
        self.hop_n = int(config.HOP_S * rate_hz)
        self.samples: deque[list[float]] = deque(maxlen=int(keep_s * rate_hz))
        self.total = 0           # samples received (plus any lost), so total / rate = stream time
        self.lost = 0
        self._next_seq: int | None = None
        self._next_classify = self.window_n
        n_hist = int(history_s / config.HOP_S)
        self.peaks: deque[tuple[float, float]] = deque(maxlen=n_hist)        # (t, peak g)
        self.history: deque[tuple[float, str]] = deque(maxlen=n_hist)        # (t, smoothed gait)
        self._recent: deque[str] = deque(maxlen=smooth)
        self.current: dict = {}  # latest result: gait, confidence, model1, model2, peak
        self.model1, self.model2, self.features = self._load(models_dir)

    @staticmethod
    def _load(models_dir):
        m1 = m2 = feats = None
        if models_dir is not None:
            p1, p2 = Path(models_dir) / "peak_model.json", Path(models_dir) / "model2.joblib"
            if p1.exists():
                m1 = PeakThresholdModel.load(p1)
            if p2.exists():
                saved = joblib.load(p2)
                m2, feats = saved["model"], saved["features"]
        return m1, m2, feats

    @property
    def has_models(self) -> bool:
        return self.model1 is not None or self.model2 is not None

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
                self._next_classify = self.total + self.window_n
            self._next_seq = seq + len(samples)
        for s in samples:
            self.samples.append(s)
            self.total += 1
            if self.total >= self._next_classify:
                self._classify()
                self._next_classify += self.hop_n

    def _classify(self) -> None:
        if len(self.samples) < self.window_n:
            return
        w = np.asarray(list(self.samples)[-self.window_n:], dtype=float)
        p = peak(w)
        self.peaks.append((self.now, p))
        result = {"peak": p, "model1": None, "model2": None, "confidence": None}
        if self.model1 is not None:
            result["model1"] = str(self.model1.predict([p])[0])
        if self.model2 is not None:
            f = all_features(w, self.rate)
            x = np.array([[f[name] for name in self.features]])
            if hasattr(self.model2, "predict_proba"):  # tree, kNN: share of votes
                proba = self.model2.predict_proba(x)[0]
                best = int(np.argmax(proba))
                result["model2"] = str(self.model2.classes_[best])
                result["confidence"] = float(proba[best])
            else:                                      # nearest centroid, k-means: no score
                result["model2"] = str(self.model2.predict(x)[0])
        raw = result["model2"] or result["model1"]
        if raw:
            self._recent.append(raw)
            # Majority of the last few answers, so the display doesn't flicker
            result["gait"] = Counter(self._recent).most_common(1)[0][0]
            self.history.append((self.now, result["gait"]))
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
