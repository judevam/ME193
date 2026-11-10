"""Finger-count CNN inference in plain NumPy (no PyTorch on the UNO Q).

Weights come from finger_cnn.npz, written by Finger Lock/src/export_numpy.py.
Same network as src/model.py: 3 conv layers (BatchNorm already folded in),
a max-pool after each, the three stages average-pooled to 4x4 and concatenated,
then one linear layer.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _conv(x, w, b):
    """x: (C, H, W), w: (O, C, k, k), 'same' padding, stride 1."""
    o, c, k, _ = w.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    h, wd = x.shape[1:]
    cols = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))  # (C, H, W, k, k)
    cols = cols.transpose(1, 2, 0, 3, 4).reshape(h * wd, c * k * k)
    return (cols @ w.reshape(o, -1).T + b).T.reshape(o, h, wd)


def _maxpool2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _avgpool_to4(x):
    c, h, w = x.shape
    return x.reshape(c, 4, h // 4, 4, w // 4).mean(axis=(2, 4))


class FingerNetNumpy:
    def __init__(self, path=None):
        path = path or Path(__file__).with_name("finger_cnn.npz")
        z = np.load(path, allow_pickle=False)
        self.p = {k: z[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3", "fc_w", "fc_b")}
        self.classes = [str(c) for c in z["classes"]]
        self.roi = json.loads(str(z["roi"]))
        self.img = int(z["img"])

    def preprocess(self, frame_bgr):
        """Full camera frame -> (img, img) float32 in [0, 1]. Matches src/model.py."""
        r = self.roi
        crop = frame_bgr[r["y"]:r["y"] + r["size"], r["x"]:r["x"] + r["size"]]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.img, self.img), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0

    def logits(self, gray):
        p = self.p
        a1 = _maxpool2(np.maximum(_conv(gray[None], p["w1"], p["b1"]), 0))
        a2 = _maxpool2(np.maximum(_conv(a1, p["w2"], p["b2"]), 0))
        a3 = _maxpool2(np.maximum(_conv(a2, p["w3"], p["b3"]), 0))
        self.last = (gray, a1, a2, a3)  # kept for the feature-map view
        feats = np.concatenate([_avgpool_to4(a).ravel() for a in (a1, a2, a3)])
        return p["fc_w"] @ feats + p["fc_b"]

    def predict(self, frame_bgr):
        """-> (probabilities array, label, confidence)"""
        z = self.logits(self.preprocess(frame_bgr))
        e = np.exp(z - z.max())
        probs = e / e.sum()
        k = int(probs.argmax())
        return probs, self.classes[k], float(probs[k])
