"""Portfolio figure: how the two models decide, drawn on the real dataset.
Run from "Gait Wearable/" after train.py; writes models/model_overview.png."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import config
from features import build_dataset, FEATURE_COLUMNS

INK, MUTED, GRID, BOXFILL, EDGE = "#1f2321", "#5e6661", "#e3e6e4", "#f3f5f4", "#c4cbc7"
COLORS = {"standing": "#2a78d6", "walking": "#eb6834", "running": "#1baf7a", "biking": "#eda100"}
ORDER = ["biking", "standing", "walking", "running"]          # by peak, low to high

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": EDGE})

df = build_dataset(sorted(config.DATA_DIR.glob("*.csv")))
models = json.load(open(config.MODEL_DIR / "gait_models.json"))
m2 = models["model2"]
cut = [0.88, 1.16, 2.34]

fig = plt.figure(figsize=(14, 10.2), dpi=200, facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.18,
                      left=0.095, right=0.97, top=0.9, bottom=0.095)
fig.suptitle("Gait Wearable: how the classifier decides", x=0.06, ha="left",
             fontsize=17, fontweight="bold")
fig.text(0.06, 0.925, "Every second, the last 2 s of thigh IMU data goes through two models. "
         "The LED shows Model 2's answer; Model 1's is reported alongside it.",
         fontsize=11, color=MUTED)

# ---- Top: pipeline schematic -------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")


def box(x0, x1, y0, y1, text, accent=False, bold_first=True):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#e6f0fb" if accent else BOXFILL,
                                ec=COLORS["standing"] if accent else EDGE, lw=1.3))
    lines = text.split("\n")
    n = len(lines)
    for i, ln in enumerate(lines):
        y = (y0 + y1) / 2 + (n - 1) * 1.55 - i * 3.1
        ax.text((x0 + x1) / 2, y, ln, ha="center", va="center",
                fontsize=10.5 if i == 0 else 9.2,
                fontweight="bold" if (i == 0 and bold_first) else "normal",
                color=INK if i == 0 else MUTED)


def arrow(p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=INK,
                                 linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0))


box(1, 17, 13, 27, "2 s window\n200 samples × 6 channels\nax ay az · gx gy gz\nnew window every 1 s")

ax.text(22, 37.3, "MODEL 1 · THRESHOLDS", fontsize=9, fontweight="bold", color=MUTED)
box(22, 40, 24, 35, "Peak on thigh axis (X)\none number per window")
box(45, 64, 24, 35, "Compare to 3 cutoffs\n0.88 g · 1.16 g · 2.34 g")
box(69, 84, 24, 35, "Model 1 label\nsent to dashboard")
arrow((17, 22), (22, 29.5)); arrow((40, 29.5), (45, 29.5)); arrow((64, 29.5), (69, 29.5))

ax.text(22, 15.3, "MODEL 2 · K-NEAREST NEIGHBORS", fontsize=9, fontweight="bold", color=MUTED)
box(22, 40, 1, 13, f"{len(FEATURE_COLUMNS)} features\nmean, std, min, max, range\nrhythm, tilt, correlations", accent=True)
box(45, 64, 1, 13, "5 nearest neighbors\nstandardize, then find the\n5 closest of 533 labeled windows", accent=True)
box(69, 84, 1, 13, "Majority vote\nof the 5 neighbors\n= Model 2 label", accent=True)
box(88, 99, 1, 13, "LED matrix\nmajority of last 3\nanswers (no flicker)")
arrow((17, 18), (22, 7)); arrow((40, 7), (45, 7)); arrow((64, 7), (69, 7)); arrow((84, 7), (88, 7))

# ---- Bottom left: Model 1 on the real data ------------------------------------------------
a1 = fig.add_subplot(gs[1, 0])
rng = np.random.default_rng(1)
for i, lab in enumerate(ORDER):
    p = df.loc[df.label == lab, "peak"].to_numpy()
    a1.scatter(p, i + rng.uniform(-0.28, 0.28, len(p)), s=14, color=COLORS[lab], alpha=0.75,
               edgecolors="white", linewidths=0.4, zorder=3)
for c in cut:
    a1.axvline(c, color=INK, ls=(0, (4, 3)), lw=1.2, zorder=2)
    a1.text(c + (-0.04 if c == 0.88 else 0.04), len(ORDER) - 0.35, f"{c:.2f} g",
            ha="right" if c == 0.88 else "left", va="bottom", fontsize=9, color=INK,
            bbox=dict(fc="white", ec="none", pad=1))
a1.set_yticks(range(len(ORDER)), [f"{l} ({(df.label == l).sum()})" for l in ORDER], color=INK)
a1.set_ylim(-0.6, len(ORDER) - 0.05)
a1.set_xlabel("peak acceleration along the thigh in the window (g)")
a1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    a1.spines[s].set_visible(False)
a1.tick_params(axis="y", length=0)
a1.set_title("Model 1: each activity lands in its own range", loc="left", fontsize=12,
             fontweight="bold", pad=22)
a1.text(0, 1.015, "One dot per 2 s window. Dashed lines are the learned cutoffs.",
        transform=a1.transAxes, fontsize=9.5, color=MUTED)

# ---- Bottom right: Model 2's feature space ------------------------------------------------
a2 = fig.add_subplot(gs[1, 1])
feats = models["features"]
X = (df[feats].to_numpy(float) - np.array(m2["mean"])) / np.array(m2["scale"])
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Xc @ Vt[:2].T
var = S**2 / np.sum(S**2)

labels = df.label.to_numpy()
for lab in ["standing", "walking", "running", "biking"]:
    m = labels == lab
    a2.scatter(P[m, 0], P[m, 1], s=14, color=COLORS[lab], alpha=0.75, edgecolors="white",
               linewidths=0.4, label=lab, zorder=3)

# one example: a walking window and its 5 nearest neighbors in the full feature space
walk = np.where(labels == "walking")[0]
q = walk[len(walk) // 2]
d = np.linalg.norm(X - X[q], axis=1); d[q] = np.inf
nn = np.argsort(d)[:5]
for j in nn:
    a2.plot([P[q, 0], P[j, 0]], [P[q, 1], P[j, 1]], color=INK, lw=1, zorder=4)
a2.scatter(P[nn, 0], P[nn, 1], s=40, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=5)
a2.scatter(P[q, 0], P[q, 1], s=150, marker="*", color=INK, edgecolors="white", linewidths=0.8, zorder=6)
votes = ", ".join(f"{(labels[nn] == l).sum()} {l}" for l in COLORS if (labels[nn] == l).any())
a2.annotate(f"new window\n5 neighbors: {votes}", (P[q, 0], P[q, 1]), xytext=(40, -60),
            textcoords="offset points", fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
            bbox=dict(fc="white", ec=EDGE, pad=3))

a2.set_xlabel(f"feature-space direction 1 ({var[0]:.0%} of spread)")
a2.set_ylabel(f"direction 2 ({var[1]:.0%} of spread)")
a2.grid(color=GRID, lw=0.8, zorder=0)
for s in ("top", "right"):
    a2.spines[s].set_visible(False)
a2.legend(loc="best", frameon=True, framealpha=0.95, edgecolor=EDGE, fontsize=9)
a2.set_title(f"Model 2: windows cluster by activity in {len(feats)}-D feature space", loc="left",
             fontsize=12, fontweight="bold", pad=22)
a2.text(0, 1.015, "Flattened to 2-D (PCA) for display; neighbors are found in all dimensions.",
        transform=a2.transAxes, fontsize=9.5, color=MUTED)

fig.text(0.095, 0.012, "Leave-one-recording-out accuracy on 14 recordings / 533 windows:  "
         "Model 1  98.7 %   ·   Model 2 (k-NN)  98.3 %", fontsize=10, color=MUTED)

out = config.MODEL_DIR / "model_overview.png"
fig.savefig(out, facecolor="white")
print(out)
