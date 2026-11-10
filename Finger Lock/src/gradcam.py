"""Grad-CAM: which part of the image did the CNN use for its answer?

For a chosen image we take the answer's score, backpropagate it to the last conv
layer, average the gradients over the map to get one weight per filter, and
make a heatmap = ReLU(sum_k weight_k * filter_k). Red = "this region pushed the
network toward its answer".

  python src/gradcam.py                  # grid: examples of each class, jude and mia
  python src/gradcam.py --person mia     # only one person's images
"""
import argparse
import glob
import os
import random
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FingerNet, preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "finger_cnn_final.pt"))
ap.add_argument("--person", default=None, help="only use images from this person")
ap.add_argument("--per-class", type=int, default=4)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "gradcam.png"))
args = ap.parse_args()

ck = torch.load(args.model, map_location="cpu")
classes, roi = ck["classes"], ck["roi"]
net = FingerNet(n_classes=len(classes))
net.load_state_dict(ck["state_dict"])
net.eval()
random.seed(args.seed)


def gradcam(gray):
    """gray (96, 96) float -> (heatmap in [0,1] at 96x96, predicted index, confidence)."""
    x = torch.from_numpy(gray)[None, None]
    out, (a1, a2, a3) = net(x, return_maps=True)
    a3.retain_grad()
    probs = torch.softmax(out, dim=1)[0]
    k = int(probs.argmax())
    net.zero_grad()
    out[0, k].backward()
    weights = a3.grad.mean(dim=(2, 3), keepdim=True)          # one weight per filter
    cam = F.relu((weights * a3).sum(dim=1, keepdim=True))      # (1, 1, 12, 12)
    cam = F.interpolate(cam, size=gray.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, k, float(probs[k])


rows = []
for c in classes:
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", c, "*.jpg")))
    if args.person:
        files = [f for f in files if os.path.basename(f).startswith(args.person + "_")]
    persons = sorted({os.path.basename(f).split("_")[0] for f in files})
    picks = []
    for who in persons:  # spread the picks over the people present
        fs = [f for f in files if os.path.basename(f).startswith(who + "_")]
        picks += random.sample(fs, min(len(fs), max(1, args.per_class // len(persons))))
    rows.append(picks[:args.per_class])

n_cols = max(len(r) for r in rows)
fig, axes = plt.subplots(len(classes), n_cols, figsize=(2.3 * n_cols, 2.3 * len(classes)))
axes = np.atleast_2d(axes)
for ri, (c, picks) in enumerate(zip(classes, rows)):
    for ci in range(n_cols):
        ax = axes[ri, ci]
        ax.axis("off")
        if ci >= len(picks):
            continue
        gray = preprocess(cv2.imread(picks[ci]), roi)
        cam, k, conf = gradcam(gray)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        who = os.path.basename(picks[ci]).split("_")[0]
        ok = classes[k] == c
        ax.set_title(f"true {c} / saw {classes[k]} {conf * 100:.0f}%\n{who}",
                     fontsize=8, color="green" if ok else "red")
fig.suptitle("Grad-CAM: where the network looks (red = most important)", fontsize=12)
plt.tight_layout()
plt.savefig(args.out, dpi=110)
print("saved", args.out)
