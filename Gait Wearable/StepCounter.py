"""Count steps (and estimate distance) from an ankle IMU with a small PyTorch neural net.

  python StepCounter.py template            # list the walking/running recordings in steps.csv
  python StepCounter.py train               # fit on data/, save models/step_counter.pt
  python StepCounter.py eval                # leave-one-recording-out accuracy -> models/step_scores.md
  python StepCounter.py count FILE.csv      # steps + distance for one recording
  python StepCounter.py count FILE.csv --height-cm 175
  python StepCounter.py serve               # live step count + distance on your phone (web page)
  python StepCounter.py serve --replay data/walking_20261025-174508.csv   # no board: replay a file

How it works
  Input   all six IMU outputs plus three derived signals (accel magnitude, gyro magnitude,
          accel with gravity filtered out), 100 Hz.
  Net     a fully-convolutional 1-D net (dilated residual blocks). For every sample it
          outputs  (a) how likely a step lands right there and  (b) which gait it looks
          like: none / walking / running.
  Count   steps = peaks of the step output.
  Gait    (b) says whether each step is a walking or a running step.
  Distance steps x step length. Step length comes from the gait and the cadence (steps/s),
          fitted to the recordings whose real distance you wrote in steps.csv; before that,
          a height-based guess.

Placement: ONE ankle. The sensor sees every step: its own foot as a hard impact (~3-4 g
walking) and the other foot as a weaker one (~1.5 g), alternating about 0.5 s apart. So the
net marks every foot landing and one event is one step (STEPS_PER_EVENT = 1). Record into a
separate folder (data_ankle/) with its own steps.csv, never mixed with the thigh recordings.

Labels. You only write down how many steps you took (and optionally how far) per recording
in data/steps.csv. Marking every single step is too slow, so we find the
candidate impacts with a peak detector, keep the N strongest where N is your count, and
use those as the training targets. Standing and biking recordings are all-zero targets.
Recordings with no count in steps.csv still train, but are "auto" labeled and are left out
of the accuracy scores.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn

import config

STEPS_NAME = "steps.csv"                         # in the data folder: file, steps, distance_m
MODEL_PATH = config.MODEL_DIR / "step_counter.pt"
SCORES_PATH = config.MODEL_DIR / "step_scores.md"

GAITS = ["none", "walking", "running"]          # the net's second output
STEPS_PER_EVENT = 1       # every foot landing is one event. The ankle sensor sees BOTH feet (its own as a hard
                          # impact, the other foot as a weaker shock), so one event = one step, not one stride
MIN_EVENT_GAP_S = 0.22    # fastest plausible running cadence is ~4.5 steps/s
BUMP_SIGMA = 4            # width (samples) of the target bump around each step
CROP = 400                # training crop: 4 s
PEAK_THRESHOLD = 0.35     # step output above this counts as a step

# Fallback step length as a share of body height (stride-length rules of thumb). Replaced
# by the fit when steps.csv has real distances.
DEFAULT_STEP_PER_HEIGHT = {"walking": 0.415, "running": 0.65}


# ---------------------------------------------------------------- data

def load_recording(path: Path) -> tuple[str, np.ndarray]:
    """(label, array of shape (n, 6)) from a CSV recorded on the board."""
    first = path.open().readline()
    label = next((p.split("=")[1] for p in first.split() if p.startswith("label=")), path.name.split("_")[0])
    df = pd.read_csv(path, comment="#")
    return label, df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(np.float32)


def read_truth(data_dir: Path) -> dict[str, dict]:
    """steps.csv -> {file name: {"steps": int or None, "distance_m": float or None}}."""
    if not (data_dir / STEPS_NAME).exists():
        return {}
    df = pd.read_csv(data_dir / STEPS_NAME)
    out = {}
    for row in df.itertuples():
        out[row.file] = {"steps": None if pd.isna(row.steps) else int(row.steps),
                         "distance_m": None if pd.isna(row.distance_m) else float(row.distance_m)}
    return out


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    return np.convolve(np.pad(x, n // 2, mode="edge"), k, mode="valid")[: len(x)]


def features(w: np.ndarray, rate_hz: int = config.RATE_HZ) -> np.ndarray:
    """(n, 6) raw IMU -> (9, n) net input. Fixed scales, so nothing to store with the model."""
    a, g = w[:, :3], w[:, 3:]
    amag = np.linalg.norm(a, axis=1)
    gmag = np.linalg.norm(g, axis=1)
    dyn = amag - moving_average(amag, rate_hz)          # impacts with gravity filtered out
    return np.concatenate([a / 2, g / 300, (amag - 1)[:, None], (gmag / 300)[:, None],
                           dyn[:, None]], axis=1).T.astype(np.float32)


def detect_peaks(w: np.ndarray, n_steps: int | None = None, rate_hz: int = config.RATE_HZ):
    """The classic way: peaks of the gravity-free accel magnitude. Used to make training
    targets, and as the baseline the neural net has to beat. Returns sample indices."""
    amag = np.linalg.norm(w[:, :3], axis=1)
    sig = moving_average(amag - moving_average(amag, rate_hz), 5)
    idx, props = find_peaks(sig, distance=int(MIN_EVENT_GAP_S * rate_hz), prominence=0.05)
    prom = props["prominences"]
    if len(idx) == 0:
        return idx
    if n_steps is not None:                              # keep the N strongest
        keep = np.sort(np.argsort(prom)[::-1][:n_steps])
    else:                                                # no count: keep clear peaks only
        keep = np.flatnonzero(prom >= 0.35 * np.percentile(prom, 90))
    return idx[keep]


def make_targets(w: np.ndarray, label: str, n_steps: int | None, rate_hz: int = config.RATE_HZ):
    """(step bump curve (n,), gait class per sample (n,)) for one recording."""
    n = len(w)
    bump = np.zeros(n, np.float32)
    gait = np.zeros(n, np.int64)
    if label in ("walking", "running"):
        steps = detect_peaks(w, n_steps, rate_hz)
        x = np.arange(n)
        for s in steps:
            bump = np.maximum(bump, np.exp(-0.5 * ((x - s) / BUMP_SIGMA) ** 2))
            gait[max(0, s - rate_hz): s + rate_hz] = GAITS.index(label)
    return bump, gait


# ---------------------------------------------------------------- the net

class Block(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 1), nn.BatchNorm1d(ch))

    def forward(self, x):
        return torch.relu(x + self.body(x))


class StepNet(nn.Module):
    """(batch, 9, n) -> (batch, 1 + 3, n): step logit, then none/walking/running logits.
    Dilations 1..16 give each output sample a view of about +-0.7 s around it."""
    def __init__(self, in_ch: int = 9, ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, ch, 5, padding=2), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(ch, d) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Conv1d(ch, 1 + len(GAITS), 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def train_net(recs: list[dict], iters: int = 600, seed: int = 0) -> StepNet:
    """recs: dicts with x (9, n), bump (n,), gait (n,). Random 4 s crops, light augmentation."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = StepNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=iters)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
    ce = nn.CrossEntropyLoss()
    lengths = np.array([r["x"].shape[1] for r in recs], float)
    net.train()
    for _ in range(iters):
        xs, bumps, gaits = [], [], []
        for i in rng.choice(len(recs), 16, p=lengths / lengths.sum()):
            r = recs[i]
            s = rng.integers(0, max(1, r["x"].shape[1] - CROP))
            sl = slice(s, s + CROP)
            x = r["x"][:, sl] * rng.uniform(0.9, 1.1)                 # sensor gain / strength of stride
            x = x + rng.normal(0, 0.01, x.shape).astype(np.float32)   # sensor noise
            xs.append(x), bumps.append(r["bump"][sl]), gaits.append(r["gait"][sl])
        x = torch.from_numpy(np.stack(xs))
        out = net(x)
        loss = bce(out[:, 0], torch.from_numpy(np.stack(bumps))) \
            + 0.5 * ce(out[:, 1:], torch.from_numpy(np.stack(gaits)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def count_peaks(p: np.ndarray, threshold: float, rate_hz: int = config.RATE_HZ) -> np.ndarray:
    """The net's step curve -> sample index of every step (the peaks above the threshold)."""
    return find_peaks(p, height=threshold, distance=int(MIN_EVENT_GAP_S * rate_hz))[0]


@torch.no_grad()
def predict(net: StepNet, w: np.ndarray, rate_hz: int = config.RATE_HZ, return_curve: bool = False):
    """Raw IMU (n, 6) -> (step sample indices, gait name per step[, the step curve])."""
    out = net(torch.from_numpy(features(w, rate_hz))[None])[0]
    p = torch.sigmoid(out[0]).numpy()
    g = out[1:].softmax(0).numpy()                                    # (3, n)
    idx = count_peaks(p, PEAK_THRESHOLD, rate_hz)
    half = int(0.3 * rate_hz)
    gaits = [GAITS[int(g[:, max(0, i - half): i + half].mean(axis=1).argmax())] for i in idx]
    return (idx, gaits, p) if return_curve else (idx, gaits)


# ---------------------------------------------------------------- distance

def step_lengths(idx: np.ndarray, gaits: list[str], params: dict, rate_hz: int = config.RATE_HZ):
    """Meters covered by each detected event (one stride = STEPS_PER_EVENT steps): the gait's step length times STEPS_PER_EVENT,
    nudged by cadence in steps/s (faster = longer)."""
    cadence = np.full(len(idx), np.nan)
    if len(idx) > 1:
        dt = np.diff(idx) / rate_hz
        cadence[1:] = STEPS_PER_EVENT / dt
        cadence[0] = cadence[1]
    out = np.zeros(len(idx))
    for k, (c, g) in enumerate(zip(cadence, gaits)):
        if g == "none" or g not in params:
            continue                                                  # not walking or running: no distance
        p = params[g]
        c = p["cadence0"] if not np.isfinite(c) else np.clip(c, 0.5, 5.0)
        out[k] = STEPS_PER_EVENT * max(0.0, p["length0"] + p["slope"] * (c - p["cadence0"]))
    return out


def default_length_params(height_cm: float) -> dict:
    h = height_cm / 100
    return {"walking": {"length0": DEFAULT_STEP_PER_HEIGHT["walking"] * h, "cadence0": 1.9, "slope": 0.0},
            "running": {"length0": DEFAULT_STEP_PER_HEIGHT["running"] * h, "cadence0": 2.8, "slope": 0.0}}


def fit_length_params(rows: list[dict], height_cm: float) -> dict:
    """rows: per walked recording {gait, steps, distance_m, cadence}. One average step length
    per gait (total distance / total steps). Slope vs. cadence only with 4+ recordings of the
    gait, and then only used if it is not absurd; otherwise 0."""
    params = default_length_params(height_cm)
    for g in ("walking", "running"):
        r = [x for x in rows if x["gait"] == g and x["steps"] > 0]
        if not r:
            continue
        L = np.array([x["distance_m"] / x["steps"] for x in r])
        c = np.array([x["cadence"] for x in r])
        params[g]["length0"], params[g]["cadence0"] = float(np.average(L, weights=[x["steps"] for x in r])), float(c.mean())
        if len(r) >= 4 and c.std() > 0.1:
            slope = float(np.polyfit(c, L, 1)[0])
            params[g]["slope"] = slope if 0 <= slope <= 0.6 else 0.0
    return params


# ---------------------------------------------------------------- commands

def collect(data_dir: Path):
    """Every recording with its labels. Returns a list of dicts."""
    truth = read_truth(data_dir)
    recs = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == STEPS_NAME:
            continue
        label, w = load_recording(path)
        t = truth.get(path.name, {})
        n_steps = t.get("steps")
        if label in ("standing", "biking"):
            n_steps = 0
        n_events = None if n_steps is None else round(n_steps / STEPS_PER_EVENT)   # the sensor leg's steps, one per stride
        bump, gait = make_targets(w, label, n_events)
        recs.append({"name": path.name, "label": label, "w": w, "x": features(w), "bump": bump, "gait": gait,
                     "steps": n_steps, "distance_m": t.get("distance_m"),
                     "verified": n_steps is not None and label in ("walking", "running")})
    return recs


def cmd_template(args):
    existing = read_truth(args.data)
    rows = [{"file": p.name, **existing.get(p.name, {"steps": None, "distance_m": None})}
            for p in sorted(args.data.glob("*.csv"))
            if p.name.split("_")[0] in ("walking", "running")]
    pd.DataFrame(rows, columns=["file", "steps", "distance_m"]).to_csv(args.data / STEPS_NAME, index=False)
    print(f"wrote {args.data / STEPS_NAME}: fill in steps (and distance_m if you measured it) for each file")


def cmd_train(args):
    recs = collect(args.data)
    print(f"{len(recs)} recordings, {sum(r['verified'] for r in recs)} with a step count")
    net = train_net(recs, args.iters)
    rows = [{"gait": r["label"], "steps": r["steps"], "distance_m": r["distance_m"],
             "cadence": STEPS_PER_EVENT * len(predict(net, r["w"])[0]) / (len(r["w"]) / config.RATE_HZ) }
            for r in recs if r["verified"] and r["distance_m"]]
    params = fit_length_params(rows, args.height_cm)
    config.MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state": net.state_dict(), "length_params": params}, MODEL_PATH)
    print(f"saved {MODEL_PATH}; step lengths: {json.dumps(params)}")


def load_model():
    ck = torch.load(MODEL_PATH, weights_only=True)
    net = StepNet()
    net.load_state_dict(ck["state"])
    return net.eval(), ck["length_params"]


def cmd_count(args):
    net, params = load_model()
    if args.height_cm:
        params = default_length_params(args.height_cm)
    _, w = load_recording(args.file)
    idx, gaits = predict(net, w)
    d = step_lengths(idx, gaits, params).sum()
    secs = len(w) / config.RATE_HZ
    print(f"{STEPS_PER_EVENT * len(idx)} steps ({STEPS_PER_EVENT * gaits.count('walking')} walking, {STEPS_PER_EVENT * gaits.count('running')} running) in {secs:.0f} s, "
          f"about {d:.1f} m")


def cmd_eval(args):
    """Leave-one-recording-out: each walking/running recording that has a step count is
    counted by a net trained on every OTHER recording. The peak detector is the baseline."""
    recs = collect(args.data)
    verified = [r for r in recs if r["verified"]]
    if not verified:
        raise SystemExit("no recordings with a step count: run `template`, fill in steps.csv")
    rows = []
    for held in verified:
        net = train_net([r for r in recs if r is not held], args.iters)
        idx, gaits, curve = predict(net, held["w"], return_curve=True)
        base = STEPS_PER_EVENT * len(detect_peaks(held["w"]))
        rows.append({"file": held["name"], "gait": held["label"], "steps": held["steps"], "net": STEPS_PER_EVENT * len(idx),
                     "curve": curve,
                     "baseline": base, "distance_m": held["distance_m"], "w": held["w"],
                     "idx": idx, "gaits": gaits, "cadence": STEPS_PER_EVENT * len(idx) / (len(held["w"]) / config.RATE_HZ)})
        print(f"{held['name']}: true {held['steps']}, net {STEPS_PER_EVENT * len(idx)}, peak detector {base}")
    # Recordings with no steps in them (standing, biking): any step counted is a false step.
    neg_rows = []
    for held in [r for r in recs if r["label"] in ("standing", "biking")]:
        net = train_net([r for r in recs if r is not held], args.iters)
        idx0, _, curve0 = predict(net, held["w"], return_curve=True)
        n_net = STEPS_PER_EVENT * len(idx0)
        n_base = STEPS_PER_EVENT * len(detect_peaks(held["w"]))
        neg_rows.append({"file": held["name"], "label": held["label"], "secs": len(held["w"]) / config.RATE_HZ,
                         "net": n_net, "baseline": n_base, "curve": curve0})
        print(f"{held['name']}: true 0, net {n_net}, peak detector {n_base}")
    # Distance: lengths fitted on the other walked recordings (again leave-one-out).
    for r in rows:
        if not r["distance_m"]:
            r["dist_est"] = None
            continue
        others = [x for x in rows if x is not r and x["distance_m"]]
        params = fit_length_params([{**x, "steps": x["steps"]} for x in others], args.height_cm)
        r["dist_est"] = step_lengths(r["idx"], r["gaits"], params).sum()
    lines = ["# Step counter scores", "", "Leave-one-recording-out: each recording is counted by a net trained on the others.", "",
             "| recording | true steps | neural net | peak detector |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['steps']} | {r['net']} ({100 * (r['net'] - r['steps']) / r['steps']:+.1f}%) "
                     f"| {r['baseline']} ({100 * (r['baseline'] - r['steps']) / r['steps']:+.1f}%) |")
    err = lambda k: np.mean([abs(r[k] - r["steps"]) / r["steps"] for r in rows]) * 100
    lines += ["", f"Mean absolute count error: neural net **{err('net'):.1f}%**, peak detector {err('baseline'):.1f}%."]
    if neg_rows:
        lines += ["", "Recordings with no steps (every step counted is false):", "",
                  "| recording | seconds | neural net | peak detector |", "|---|---|---|---|"]
        lines += [f"| {r['file']} ({r['label']}) | {r['secs']:.0f} | {r['net']} | {r['baseline']} |" for r in neg_rows]
    # Does a stricter threshold on the net's step curve fix the over-counting? Same held-out
    # curves, different cut-off. Picking the best one on these same recordings flatters it, so
    # report the 0.35 row (the untuned setting) next to it.
    lines += ["", "Step-curve threshold sweep (same held-out predictions; 0.35 is the untuned default):", "",
              "| threshold | mean abs. count error | mean signed error | false steps on non-step recordings |", "|---|---|---|---|"]
    for th in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        cnt = [STEPS_PER_EVENT * len(count_peaks(r["curve"], th)) for r in rows]
        rel = np.array([(c - r["steps"]) / r["steps"] for c, r in zip(cnt, rows)]) * 100
        fal = sum(STEPS_PER_EVENT * len(count_peaks(r["curve"], th)) for r in neg_rows)
        lines.append(f"| {th} | {np.abs(rel).mean():.1f}% | {rel.mean():+.1f}% | {fal} |")
    dist = [r for r in rows if r["dist_est"] is not None]
    if dist:
        derr = np.mean([abs(r["dist_est"] - r["distance_m"]) / r["distance_m"] for r in dist]) * 100
        lines += ["", f"Distance (fitted on the other recordings): mean absolute error **{derr:.1f}%** over {len(dist)} recordings.", "",
                  "| recording | true m | estimated m |", "|---|---|---|"]
        lines += [f"| {r['file']} | {r['distance_m']:.0f} | {r['dist_est']:.0f} |" for r in dist]
    config.MODEL_DIR.mkdir(exist_ok=True)
    SCORES_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- live: web page for the phone

class LiveCounter:
    """Counts steps on the live IMU stream. Every half second the net looks at the last 10 s;
    a step is only counted once it is 1 s old, because the net needs to see about 0.7 s on
    each side of a step to be sure of it."""
    KEEP_S, LOOKAHEAD_S, EVERY_S = 10, 1.0, 0.5

    def __init__(self, net, params, rate_hz=config.RATE_HZ):
        self.net, self.params, self.rate = net, params, rate_hz
        self.lock = threading.Lock()
        self.buf: deque = deque(maxlen=self.KEEP_S * rate_hz)
        self.reset()

    def reset(self):
        with self.lock:
            self.total = 0                     # samples seen since the reset
            self.steps = 0
            self.distance_m = 0.0
            self.gait = "none"
            self.cadence = 0.0
            self.walk_steps = self.run_steps = 0
            self.last_step = None              # sample index of the last counted step
            self.last_run = 0
            self.buf.clear()

    def add(self, samples):
        with self.lock:
            self.buf.extend(samples)
            self.total += len(samples)
            if self.total - self.last_run < self.EVERY_S * self.rate or len(self.buf) < 3 * self.rate:
                return
            self.last_run = self.total
            w = np.asarray(self.buf, np.float32)
            idx, gaits = predict(self.net, w, self.rate)
            base = self.total - len(w)
            for i, g in zip(idx, gaits):
                a = base + int(i)
                if a > self.total - self.LOOKAHEAD_S * self.rate:
                    break                                                   # too new to be sure
                if self.last_step is not None and a - self.last_step < MIN_EVENT_GAP_S * self.rate:
                    continue                                                # already counted
                gap = None if self.last_step is None else a - self.last_step
                close = gap is not None and gap < 2 * self.rate
                pair = np.array([self.last_step, a]) if close else np.array([a])
                m = step_lengths(pair, [g] * len(pair), self.params, self.rate)[-1]
                if close:
                    self.cadence = STEPS_PER_EVENT * self.rate / gap
                self.last_step = a
                self.gait = g
                if g in ("walking", "running"):
                    self.steps += STEPS_PER_EVENT
                    self.distance_m += m
                    self.walk_steps += STEPS_PER_EVENT * (g == "walking")
                    self.run_steps += STEPS_PER_EVENT * (g == "running")

    def state(self):
        with self.lock:
            idle = self.last_step is None or self.total - self.last_step > 3 * self.rate
            return {"steps": self.steps, "distance_m": round(self.distance_m, 1), "walk_steps": int(self.walk_steps),
                    "run_steps": int(self.run_steps), "cadence": round(self.cadence, 2),
                    "gait": "stopped" if idle else self.gait, "seconds": round(self.total / self.rate)}


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Step counter</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
.big{font-size:22vw;font-weight:700;line-height:1}.sub{color:#9aa;margin:4px 0 18px}
.row{display:flex;gap:10px;justify-content:center;margin:8px 0}
.card{background:#1c1f24;border-radius:12px;padding:12px 16px;flex:1}.card b{display:block;font-size:7vw}
button{font-size:5vw;padding:12px;border:0;border-radius:10px;background:#2d333b;color:#eee;flex:1}
button.rec{background:#b3261e}#gait{font-size:6vw;text-transform:uppercase}
</style>
<div class=big id=steps>0</div><div class=sub>steps</div>
<div id=gait>-</div>
<div class=row><div class=card><b id=dist>0</b>meters</div><div class=card><b id=cad>0</b>steps/s</div></div>
<div class=row><div class=card><b id=ws>0</b>walking steps</div><div class=card><b id=rs>0</b>running steps</div></div>
<div class=sub id=rec>-</div>
<div class=row><button onclick="cmd('reset')">Reset count</button></div>
<div class=row><button class=rec onclick="cmd('start','walking')">Record walk</button>
<button class=rec onclick="cmd('start','running')">Record run</button></div>
<div class=row><button onclick="cmd('stop')">Stop recording</button></div>
<script>
const $=id=>document.getElementById(id);
function cmd(c,label){fetch('/cmd',{method:'POST',body:JSON.stringify({cmd:c,label:label})})}
async function tick(){try{const s=await (await fetch('/state')).json();
$('steps').textContent=s.steps;$('dist').textContent=s.distance_m;$('cad').textContent=s.cadence;
$('ws').textContent=s.walk_steps;$('rs').textContent=s.run_steps;$('gait').textContent=s.gait;
$('rec').textContent=s.recording?('RECORDING '+s.recording):(s.online?'wearable online':'wearable offline')}catch(e){}}
setInterval(tick,500);tick();
</script>"""


def lan_addresses() -> list[str]:
    addrs = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    return sorted(a for a in addrs if not a.startswith("127."))


def cmd_serve(args):
    net, params = load_model()
    if args.height_cm_given:
        params = default_length_params(args.height_cm)
    counter = LiveCounter(net, params)
    status = {"online": False, "recording": None}
    client = None

    if args.replay:                                    # no board: play a recording in real time
        _, w = load_recording(args.replay)

        def feed():
            for i in range(0, len(w), 10):
                counter.add(w[i:i + 10].tolist())
                time.sleep(0.1)

        threading.Thread(target=feed, daemon=True).start()
        status["online"] = True
    else:
        import gait
        from live import parse_imu

        def on_message(c, userdata, msg):
            if msg.topic.endswith("/imu"):
                counter.add(parse_imu(msg.payload)[1])
            elif msg.topic.endswith("/status"):
                status["online"] = msg.payload.decode() == "online"
            else:
                try:
                    s = json.loads(msg.payload)
                    status["recording"] = s.get("label") if s.get("mode") == "record" else None
                except ValueError:
                    pass
        client = gait.connect(on_message, imu=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self._send(json.dumps({**counter.state(), **status}).encode(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if cmd.get("cmd") == "reset":
                counter.reset()
            elif client and cmd.get("cmd") in ("start", "stop"):
                gait.send(client, {k: v for k, v in cmd.items() if v})
            self._send(b"{}", "application/json")

    print("Open on your phone (same Wi-Fi as this computer): " +
          ", ".join(f"http://{a}:{args.port}" for a in lan_addresses()))
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATA_DIR, help="folder of recordings (default data/)")
    p.add_argument("--iters", type=int, default=600, help="training steps")
    p.add_argument("--height-cm", type=float, default=167, help="for the starting step-length guess")
    p.add_argument("--threshold", type=float, help="step curve cut-off (default %s): higher = fewer steps counted" % PEAK_THRESHOLD)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template").set_defaults(fn=cmd_template)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    c = sub.add_parser("count")
    c.add_argument("file", type=Path)
    c.set_defaults(fn=cmd_count)
    sv = sub.add_parser("serve", help="live step count and distance as a web page for your phone")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--replay", type=Path, help="play a recorded CSV instead of listening to the board")
    sv.set_defaults(fn=cmd_serve)
    args = p.parse_args()
    args.height_cm_given = "--height-cm" in sys.argv
    if args.threshold is not None:
        globals()["PEAK_THRESHOLD"] = args.threshold
    args.fn(args)


if __name__ == "__main__":
    main()
