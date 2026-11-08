"""Export a trained FingerNet checkpoint to a NumPy .npz for the UNO Q.

PyTorch is only needed here, on the computer. BatchNorm is folded into the conv
weights, so the board needs just numpy (conv, relu, pool, linear).

  python src/export_numpy.py                       # models/finger_cnn_final.pt
  python src/export_numpy.py --model models/x.pt --out some.npz
"""
import argparse
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "finger_cnn_final.pt"))
ap.add_argument("--out", default=os.path.join(
    ROOT, "..", "ArduinoApps", "finger-lock-cam-test", "python", "finger_cnn.npz"))
args = ap.parse_args()

ck = torch.load(args.model, map_location="cpu")
sd = {k: v.numpy().astype(np.float64) for k, v in ck["state_dict"].items()}
out = {}
for i in (1, 2, 3):
    w, b = sd[f"c{i}.weight"], sd[f"c{i}.bias"]
    g, be = sd[f"b{i}.weight"], sd[f"b{i}.bias"]
    mean, var = sd[f"b{i}.running_mean"], sd[f"b{i}.running_var"]
    scale = g / np.sqrt(var + 1e-5)
    out[f"w{i}"] = (w * scale[:, None, None, None]).astype(np.float32)  # BN folded in
    out[f"b{i}"] = ((b - mean) * scale + be).astype(np.float32)
out["fc_w"] = sd["fc.weight"].astype(np.float32)
out["fc_b"] = sd["fc.bias"].astype(np.float32)
out["classes"] = np.array(ck["classes"])
out["roi"] = np.array(json.dumps(ck["roi"]))
out["img"] = np.array(ck["img"])
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, **out)
print("saved", os.path.abspath(args.out), f"({os.path.getsize(args.out) / 1024:.0f} KB)", ck["classes"])
