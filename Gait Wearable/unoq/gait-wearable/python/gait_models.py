"""Run the trained gait models with numpy only (no scikit-learn).

train.py saves both models to models/gait_models.json; this file runs them. The same file
runs in dashboard.py on the laptop and on the UNO Q (push_models.py copies it there), so
both always give the same answer.
"""
import json
from collections import Counter, deque

import numpy as np

from features_core import AXES, all_features, peak


class GaitModels:
    def __init__(self, d):
        self.rate = d["rate_hz"]
        self.window_n = round(d["window_s"] * self.rate)
        self.hop_n = round(d["hop_s"] * self.rate)
        self.peak_axis = AXES.index(d["peak_axis"])
        self.peak_sign = d["peak_sign"]
        m1 = d.get("model1")
        self.order = m1["order"] if m1 else None            # model 1: classes, lowest peak first
        self.thresholds = m1["thresholds"] if m1 else None  # ... and the cut points between them
        m2 = d.get("model2")
        self.kind = m2["kind"] if m2 else None
        self.features = d.get("features", [])
        self._model2 = _MODEL2[self.kind](m2) if m2 else None

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def peak_only(cls, rate_hz, window_s, hop_s, peak_axis, peak_sign):
        """No trained models yet: still computes the peak for display."""
        return cls({"rate_hz": rate_hz, "window_s": window_s, "hop_s": hop_s,
                    "peak_axis": peak_axis, "peak_sign": peak_sign})

    @property
    def has_models(self):
        return self.order is not None or self._model2 is not None

    def classify(self, w):
        """One window ((window_n, 6) array) -> {"peak", "model1", "model2", "confidence"}."""
        p = peak(w, self.peak_axis, self.peak_sign)
        result = {"peak": p, "model1": None, "model2": None, "confidence": None}
        if self.order is not None:
            result["model1"] = self.order[int(np.searchsorted(self.thresholds, p, side="right"))]
        if self._model2 is not None:
            f = all_features(w, self.rate)
            x = np.array([f[name] for name in self.features])
            result["model2"], result["confidence"] = self._model2(x)
        return result


# ---- model 2, one small function per kind. Each returns (label, confidence or None) ----

def _tree(m):
    left, right = m["left"], m["right"]
    feature, threshold, value, classes = m["feature"], m["threshold"], m["value"], m["classes"]

    def predict(x):
        x = x.astype(np.float32)  # scikit-learn compares in float32; match it exactly
        node = 0
        while left[node] != -1:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        counts = np.asarray(value[node], float)
        best = int(np.argmax(counts))
        return classes[best], float(counts[best] / counts.sum())
    return predict


def _scaled(m):
    mean, scale = np.asarray(m["mean"]), np.asarray(m["scale"])
    return lambda x: (x - mean) / scale


def _knn(m):
    scale = _scaled(m)
    X, y, k, classes = np.asarray(m["X"]), np.asarray(m["y"]), m["k"], m["classes"]

    def predict(x):
        d = np.linalg.norm(X - scale(x), axis=1)
        nearest = y[np.argsort(d, kind="stable")[:k]]
        votes = [int(np.sum(nearest == c)) for c in classes]
        best = int(np.argmax(votes))
        return classes[best], votes[best] / k
    return predict


def _nearest(m):
    """Nearest centroid and k-means: closest center wins (no confidence)."""
    scale = _scaled(m)
    centers, labels = np.asarray(m["centers"]), m["labels"]

    def predict(x):
        return labels[int(np.argmin(np.linalg.norm(centers - scale(x), axis=1)))], None
    return predict


_MODEL2 = {"decision tree": _tree, "k-nearest neighbors": _knn,
           "nearest centroid": _nearest, "k-means": _nearest}


class StreamClassifier:
    """Feed samples as they arrive; classifies the last window every hop and smooths the
    answer (majority of the last `smooth` answers) so it doesn't flicker."""

    def __init__(self, models, smooth=3):
        self.models = models
        self.buf = deque(maxlen=models.window_n)
        self.since = 0
        self.recent = deque(maxlen=smooth)
        self.current = {}

    def reset(self):
        """After a gap in the data: start the window over."""
        self.buf.clear()
        self.since = 0

    def add(self, samples):
        """Returns [(index in `samples` where a window ended, result), ...]."""
        out = []
        for i, s in enumerate(samples):
            self.buf.append(s)
            self.since += 1
            if len(self.buf) == self.models.window_n and self.since >= self.models.hop_n:
                self.since = 0
                r = self.models.classify(np.asarray(self.buf, dtype=float))
                raw = r["model2"] or r["model1"]
                if raw:
                    self.recent.append(raw)
                    r["gait"] = Counter(self.recent).most_common(1)[0][0]
                self.current = r
                out.append((i, r))
        return out
