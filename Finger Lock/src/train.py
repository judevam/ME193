"""Train the finger CNN on data/raw (made by the UNO Q capture page).

  python src/train.py                 # split by time blocks inside each burst
  python src/train.py --split burst   # strict: train on first burst, test on second
  python src/train.py --split person  # train on jude, test on every other person
"""
import argparse
import glob
import json
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CLASSES, FingerNet, IMG, preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["block", "burst", "person"], default="block")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(OUT, exist_ok=True); os.makedirs(MODELS, exist_ok=True)

roi = json.load(open(os.path.join(RAW, "roi.json")))

# ---- load + split ---------------------------------------------------------
classes = [c for c in CLASSES if glob.glob(os.path.join(RAW, c, "*.jpg"))]
N = len(classes)
print("classes:", classes)
splits = {"train": [], "val": [], "test": []}
for ci, c in enumerate(classes):
    files = sorted(glob.glob(os.path.join(RAW, c, "*.jpg")))
    bursts = {}
    for f in files:  # name: person_date_time_index.jpg -> burst = person_date_time
        bursts.setdefault("_".join(os.path.basename(f).split("_")[:3]), []).append(f)
    keys = sorted(bursts)
    for bi, k in enumerate(keys):
        fs = bursts[k]
        n = len(fs)
        for i, f in enumerate(fs):
            img = preprocess(cv2.imread(f), roi)
            who = os.path.basename(f).split("_")[0]
            if args.split == "person":
                if who != "jude":
                    part = "test"
                else:
                    part = "train" if i / n < 0.85 else "val"
            elif args.split == "burst":
                part = "train" if bi < len(keys) - 1 else ("val" if i < n // 2 else "test")
            else:
                frac = i / n
                part = "train" if frac < 0.7 else ("val" if frac < 0.85 else "test")
            splits[part].append((img, ci))

def to_tensors(items):
    x = torch.from_numpy(np.stack([i for i, _ in items]))[:, None]
    y = torch.tensor([l for _, l in items])
    return x, y

xtr, ytr = to_tensors(splits["train"])
xva, yva = to_tensors(splits["val"])
xte, yte = to_tensors(splits["test"])
print(f"split={args.split}: train {len(ytr)}, val {len(yva)}, test {len(yte)}")

# ---- augmentation (training only) -------------------------------------------
aug = T.Compose([
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3),
])

def augment(batch):
    return torch.stack([aug(img) for img in batch]).clamp(0, 1)

# ---- train -------------------------------------------------------------------
net = FingerNet(n_classes=N)
print("parameters:", sum(p.numel() for p in net.parameters()))
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()
hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best, best_state = -1, None

def evaluate(x, y):
    net.eval()
    with torch.no_grad():
        out = net(x)
        return loss_fn(out, y).item(), (out.argmax(1) == y).float().mean().item(), out.argmax(1)

t0 = time.time()
for ep in range(args.epochs):
    net.train()
    perm = torch.randperm(len(ytr))
    tl, tc = 0.0, 0
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        xb, yb = augment(xtr[idx]), ytr[idx]
        opt.zero_grad()
        out = net(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        tl += loss.item() * len(idx); tc += (out.argmax(1) == yb).sum().item()
    vl, va, _ = evaluate(xva, yva)
    hist["train_loss"].append(tl / len(ytr)); hist["train_acc"].append(tc / len(ytr))
    hist["val_loss"].append(vl); hist["val_acc"].append(va)
    if va >= best:
        best, best_state = va, {k: v.clone() for k, v in net.state_dict().items()}
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep + 1:3d}  train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.2f}"
              f"  val loss {vl:.3f} acc {va:.2f}")
print(f"trained in {time.time() - t0:.0f}s")

# ---- test --------------------------------------------------------------------
net.load_state_dict(best_state)
tl_, ta, pred = evaluate(xte, yte)
cm = np.zeros((N, N), int)
for t, p in zip(yte.tolist(), pred.tolist()):
    cm[t, p] += 1
print(f"\nTEST accuracy: {ta:.3f}  ({int(round(ta * len(yte)))}/{len(yte)})")
print("per class:", {c: f"{cm[i, i]}/{cm[i].sum()}" for i, c in enumerate(classes)})
print("confusion matrix (rows = true, cols = predicted):\n", cm)

tag = args.tag or args.split
torch.save({"state_dict": best_state, "classes": classes, "img": IMG, "roi": roi},
           os.path.join(MODELS, f"finger_cnn_{tag}.pt"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
ax[0].set(title="Loss vs epoch", xlabel="epoch", ylabel="cross-entropy"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
ax[1].set(title="Accuracy vs epoch", xlabel="epoch"); ax[1].legend()
im = ax[2].imshow(cm, cmap="Blues")
ax[2].set(title=f"Test confusion (acc {ta:.2f})", xlabel="predicted", ylabel="true",
          xticks=range(N), yticks=range(N))
for i in range(N):
    for j in range(N):
        ax[2].text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout(); plt.savefig(os.path.join(OUT, f"train_{tag}.png"), dpi=120)
print("saved model and plot for tag:", tag)
