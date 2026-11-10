"""Real data for the movie: photos, trained weights, activations, probabilities, curves.

Everything here comes from the actual project (nothing invented), except where a
scene says it is a simplified example.
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import FingerNet, preprocess  # noqa: E402

CKPT = torch.load(os.path.join(ROOT, "models", "finger_cnn_final.pt"), map_location="cpu")
CLASSES = CKPT["classes"]
ROI = CKPT["roi"]
NET = FingerNet(n_classes=len(CLASSES))
NET.load_state_dict(CKPT["state_dict"])
NET.eval()

NPZ = np.load(os.path.join(ROOT, "..", "ArduinoApps", "finger-lock-cam-test", "python", "finger_cnn.npz"))
W1, B1 = NPZ["w1"], NPZ["b1"]          # BatchNorm folded into the filters
N_PARAMS = sum(p.numel() for p in NET.parameters())


def files_for(label, person="jude"):
    return sorted(glob.glob(os.path.join(ROOT, "data", "raw", label, f"{person}_*.jpg")))


def load(path):
    """-> (color crop uint8 RGB, gray 96x96 float 0-1)."""
    bgr = cv2.imread(path)
    x, y, s = ROI["x"], ROI["y"], ROI["size"]
    crop = cv2.cvtColor(bgr[y:y + s, x:x + s], cv2.COLOR_BGR2RGB)
    return crop, preprocess(bgr, ROI)


def run(gray):
    """-> probs (7,), logits (7,), (a1, a2, a3) numpy, features (896,)."""
    x = torch.from_numpy(gray)[None, None]
    with torch.no_grad():
        out, (a1, a2, a3) = NET(x, return_maps=True)
        feats = torch.cat([torch.nn.functional.adaptive_avg_pool2d(a, 4).flatten(1) for a in (a1, a2, a3)], 1)
    return (torch.softmax(out, 1)[0].numpy(), out[0].numpy(),
            (a1[0].numpy(), a2[0].numpy(), a3[0].numpy()), feats[0].numpy())


def conv_same(gray, w, b):
    """Plain 'same' conv of a (96,96) image with filter w (5,5): returns (96,96)."""
    pad = np.pad(gray, 2)
    win = np.lib.stride_tricks.sliding_window_view(pad, (5, 5))
    return (win * w).sum(axis=(2, 3)) + b


def gradcam(gray):
    x = torch.from_numpy(gray)[None, None]
    out, (a1, a2, a3) = NET(x, return_maps=True)
    a3.retain_grad()
    k = int(out.argmax())
    NET.zero_grad()
    out[0, k].backward()
    wts = a3.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((wts * a3).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=gray.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), k


LOG = "log_block_v4.txt"          # training log of the model in models/finger_cnn_final.pt
HOLDOUT_LOG = "log_holdout_sophie_v4.txt"


def split_info(log=LOG):
    """(train, val, test) photo counts from a training log."""
    for ln in open(os.path.join(ROOT, "results", log)):
        m = re.match(r"split=\w+: train (\d+), val (\d+), test (\d+)", ln)
        if m:
            return tuple(int(v) for v in m.groups())


def test_accuracy(log):
    """(correct, total, fraction) from 'TEST accuracy: 0.895  (222/248)'."""
    for ln in open(os.path.join(ROOT, "results", log)):
        m = re.match(r"TEST accuracy: ([\d.]+)\s+\((\d+)/(\d+)\)", ln)
        if m:
            return int(m.group(2)), int(m.group(3)), float(m.group(1))


def training_curve():
    """Parse the real training log (every 5th epoch) -> dict of arrays."""
    rows = []
    for ln in open(os.path.join(ROOT, "results", LOG)):
        m = re.match(r"epoch\s+(\d+)\s+train loss ([\d.]+) acc ([\d.]+)\s+val loss ([\d.]+) acc ([\d.]+)", ln)
        if m:
            rows.append([float(v) for v in m.groups()])
    a = np.array(rows)
    return dict(epoch=a[:, 0], tl=a[:, 1], ta=a[:, 2], vl=a[:, 3], va=a[:, 4])


def counts():
    out = {}
    for c in CLASSES:
        out[c] = len(glob.glob(os.path.join(ROOT, "data", "raw", c, "*.jpg")))
    return out
