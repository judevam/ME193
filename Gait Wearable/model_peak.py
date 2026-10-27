"""Model 1: classify from a single number, the peak acceleration along one axis.

Classes are put in order of their typical peak (expected at the thigh: biking < standing
< walking < running), then one threshold is placed between each neighboring pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PeakThresholdModel:
    def __init__(self, order: list[str] | None = None, thresholds: list[float] | None = None):
        self.order = order or []            # classes from lowest to highest peak
        self.thresholds = thresholds or []  # len(order) - 1 cut points, increasing

    def fit(self, peaks, labels) -> "PeakThresholdModel":
        peaks, labels = np.asarray(peaks, float), np.asarray(labels)
        classes = sorted(set(labels), key=lambda c: np.median(peaks[labels == c]))
        self.order = classes
        self.thresholds = []
        for low, high in zip(classes, classes[1:]):
            self.thresholds.append(self._best_cut(peaks[labels == low], peaks[labels == high]))
        # Overlapping classes can produce cuts out of order; keep them increasing.
        self.thresholds = list(np.maximum.accumulate(self.thresholds))
        return self

    @staticmethod
    def _best_cut(low: np.ndarray, high: np.ndarray) -> float:
        """The cut between two classes that gets the most of both right (balanced
        accuracy), so a class with more windows doesn't pull the cut toward the other."""
        candidates = np.unique(np.concatenate([low, high]))
        cuts = (candidates[:-1] + candidates[1:]) / 2 if len(candidates) > 1 else candidates
        best, best_score = float(cuts[0]), -1.0
        for c in cuts:
            score = (low < c).mean() + (high >= c).mean()
            if score > best_score:
                best, best_score = float(c), score
        return best

    def predict(self, peaks) -> np.ndarray:
        idx = np.searchsorted(self.thresholds, np.asarray(peaks, float), side="right")
        return np.asarray(self.order)[idx]

    def to_dict(self) -> dict:
        return {"order": self.order, "thresholds": [round(t, 4) for t in self.thresholds]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PeakThresholdModel":
        d = json.loads(Path(path).read_text())
        return cls(d["order"], d["thresholds"])

    def describe(self) -> str:
        parts = [self.order[0]]
        for t, c in zip(self.thresholds, self.order[1:]):
            parts.append(f"< {t:.2f} g <= {c}")
        return " ".join(parts)
