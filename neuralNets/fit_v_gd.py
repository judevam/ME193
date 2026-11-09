"""Fit Anna's V data by gradient descent, with every gradient written out by hand (NumPy only).

Three models, from simplest to richest:

  1. single neuron      y_hat = m*x + b
  2. ReLU neuron        y_hat = max(0, w*x + b)
  3. ReLU layer (J units) y_hat = sum_j v_j * max(0, w_j*x + b_j) + c

Each one is trained the same way, for a number of epochs (one epoch = one full pass over the
data = one gradient descent step):

  loss  L = mean( (y_hat - y)^2 )                      (mean squared error)
  grad  dL/dparam  from the chain rule (see the *_grads functions)
  step  param = param - learning_rate * dL/dparam

x and y are divided by 100 before training so one learning rate works for every parameter
(the same trick as in the spreadsheet). Plots and printed errors are in the original units.

    python fit_v_gd.py                       # uses annacode-v2.tsv next to this file
    python fit_v_gd.py other.tsv --units 2   # number of ReLU units in the layer
    python fit_v_gd.py --check               # compare the hand-written gradients to numerical ones
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SCALE = 100.0


def load(path):
    """TSV with a header row: time, x, output. Returns x and y as 1-D arrays."""
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    return data[:, 1], data[:, 2]


# ---------------------------------------------------------------------------------------
# 1. single neuron:  y_hat = m x + b
# ---------------------------------------------------------------------------------------
def line_forward(p, x):
    return p["m"] * x + p["b"]


def line_grads(p, x, y):
    e = line_forward(p, x) - y                      # error of every point
    return {"m": 2 * np.mean(e * x),                # dL/dm = (2/N) sum e_i x_i
            "b": 2 * np.mean(e)}                    # dL/db = (2/N) sum e_i


# ---------------------------------------------------------------------------------------
# 2. ReLU neuron:  z = w x + b,  y_hat = max(0, z)
# ---------------------------------------------------------------------------------------
def relu_neuron_forward(p, x):
    return np.maximum(0, p["w"] * x + p["b"])


def relu_neuron_grads(p, x, y):
    z = p["w"] * x + p["b"]
    e = np.maximum(0, z) - y
    on = (z > 0)                                    # ReLU'(z): 1 where z > 0, else 0
    return {"w": 2 * np.mean(e * on * x),           # dL/dw = (2/N) sum e_i ReLU'(z_i) x_i
            "b": 2 * np.mean(e * on)}               # dL/db = (2/N) sum e_i ReLU'(z_i)


# ---------------------------------------------------------------------------------------
# 3. ReLU layer:  z_ij = w_j x_i + b_j,  a_ij = max(0, z_ij),  y_hat_i = sum_j v_j a_ij + c
# ---------------------------------------------------------------------------------------
def layer_forward(p, x):
    a = np.maximum(0, np.outer(x, p["w"]) + p["b"])   # (N, J)
    return a @ p["v"] + p["c"]


def layer_grads(p, x, y):
    z = np.outer(x, p["w"]) + p["b"]
    a = np.maximum(0, z)
    e = a @ p["v"] + p["c"] - y
    on = (z > 0)
    e_col = e[:, None]
    return {"c": 2 * np.mean(e),                                       # dL/dc
            "v": 2 * np.mean(e_col * a, axis=0),                       # dL/dv_j  = (2/N) sum e a_ij
            "w": 2 * np.mean(e_col * p["v"] * on * x[:, None], axis=0),  # dL/dw_j = (2/N) sum e v_j ReLU' x
            "b": 2 * np.mean(e_col * p["v"] * on, axis=0)}             # dL/db_j  = (2/N) sum e v_j ReLU'


def layer_init(units, xs, seed):
    """Start each unit's kink at a random data point (a kink outside the data is a dead or
    always-on unit that cannot learn)."""
    rng = np.random.RandomState(seed)
    w = rng.choice([-1.0, 1.0], units) * (0.5 + rng.rand(units))
    kinks = xs[rng.randint(len(xs), size=units)]
    return {"w": w, "b": -w * kinks, "v": 0.5 * rng.randn(units), "c": np.zeros(())}


# ---------------------------------------------------------------------------------------
# gradient descent
# ---------------------------------------------------------------------------------------
def train(forward, grads, params, x, y, lr, epochs):
    """Plain full-batch gradient descent. Returns the trained params and the loss every epoch."""
    params = {k: np.array(v, dtype=float) for k, v in params.items()}
    losses = np.empty(epochs)
    for epoch in range(epochs):
        losses[epoch] = np.mean((forward(params, x) - y) ** 2)
        g = grads(params, x, y)
        for k in params:
            params[k] = params[k] - lr * g[k]            # the update rule
    return params, losses


def gradient_check():
    """Numerical check: dL/dparam by finite differences should match the hand-written gradients."""
    rng = np.random.RandomState(0)
    x = rng.uniform(0.2, 1, 40)
    y = 2 * np.abs(x - 0.6) + 0.02 * rng.randn(40)
    cases = [("single neuron", line_forward, line_grads, {"m": 0.3, "b": 0.1}),
             ("ReLU neuron", relu_neuron_forward, relu_neuron_grads, {"w": 1.0, "b": -0.5}),
             ("ReLU layer", layer_forward, layer_grads,
              {"w": np.array([1.0, -1.0]), "b": np.array([-0.5, 0.7]), "v": np.array([1.0, 1.0]), "c": np.array(0.0)})]
    ok = True
    for name, fwd, grd, p in cases:
        g = grd(p, x, y)
        worst = 0.0
        for k in p:
            flat = np.atleast_1d(p[k]).astype(float)
            for i in range(flat.size):
                q = {kk: np.array(vv, dtype=float).copy() for kk, vv in p.items()}
                qf = np.atleast_1d(q[k]); qf[i] += 1e-6
                q[k] = qf.reshape(np.shape(p[k]))
                num = (np.mean((fwd(q, x) - y) ** 2) - np.mean((fwd(p, x) - y) ** 2)) / 1e-6
                worst = max(worst, abs(num - np.atleast_1d(g[k])[i]))
        print(f"{name:14s} biggest difference, hand-written vs numerical gradient: {worst:.2e}")
        ok &= worst < 1e-4
    print("gradient check", "PASSED" if ok else "FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=HERE / "annacode-v2.tsv")
    ap.add_argument("--units", type=int, default=2, help="ReLU units in the layer (2 is the minimum for a V)")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.1, help="learning rate for the ReLU models")
    ap.add_argument("--restarts", type=int, default=8, help="random starts for the ReLU layer; best is kept")
    ap.add_argument("--check", action="store_true", help="only run the gradient check")
    ap.add_argument("--show", action="store_true", help="also open the plot windows")
    args = ap.parse_args()

    if args.check:
        gradient_check()
        return

    x, y = load(args.data)
    xs, ys = x / SCALE, y / SCALE
    print(f"{len(x)} points, x {x.min():.1f} to {x.max():.1f}, y {y.min():.1f} to {y.max():.1f}")

    results = {}

    # 1. single neuron (a line can't bend, so expect a poor fit to a V)
    p, L = train(line_forward, line_grads, {"m": 0.0, "b": 0.0}, xs, ys, lr=0.5, epochs=args.epochs)
    results["single neuron  y = m x + b"] = (p, L, line_forward)
    print(f"single neuron:  m = {float(p['m']):.4f}, b = {float(p['b']) * SCALE:.2f}  (original units)")

    # 2. ReLU neuron (one hinge: it can follow one arm of the V)
    p, L = train(relu_neuron_forward, relu_neuron_grads, {"w": 1.0, "b": -0.5}, xs, ys, lr=args.lr, epochs=args.epochs)
    results["ReLU neuron  y = max(0, w x + b)"] = (p, L, relu_neuron_forward)
    print(f"ReLU neuron:    w = {float(p['w']):.4f}, b = {float(p['b']):.4f}  (scaled units), kink at x = {-float(p['b']) / float(p['w']) * SCALE:.1f}")

    # 3. ReLU layer: several starts, keep the best (gradient descent can get stuck)
    runs = [train(layer_forward, layer_grads, layer_init(args.units, xs, s), xs, ys, args.lr, args.epochs)
            for s in range(args.restarts)]
    print("ReLU layer, final loss of each random start (scaled units):",
          " ".join(f"{r[1][-1]:.5f}" for r in runs))
    p, L = min(runs, key=lambda r: r[1][-1])
    results[f"ReLU layer ({args.units} units)"] = (p, L, layer_forward)
    kinks = -p["b"] / p["w"] * SCALE
    print(f"ReLU layer:     kinks at x = {', '.join(f'{k:.1f}' for k in kinks)}")

    grid = np.linspace(x.min(), x.max(), 400)
    print("\nmodel                                   final MSE   RMS error (original units)")
    fits = {}
    for name, (p, L, fwd) in results.items():
        fit = fwd(p, grid / SCALE) * SCALE
        fits[name] = fit
        mse = np.mean((fwd(p, xs) * SCALE - y) ** 2)
        print(f"{name:38s}  {mse:9.2f}   {np.sqrt(mse):.2f}")

    colors = ["#eb6834", "#7a4fd1", "#1a9e5c"]
    # plot 1: data and the three fits
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=18, color="#2a78d6", label="data", zorder=2)
    for (name, fit), c in zip(fits.items(), colors):
        rms = np.sqrt(np.mean((results[name][2](results[name][0], xs) * SCALE - y) ** 2))
        ax.plot(grid, fit, color=c, lw=2.5, label=f"{name}  (RMS {rms:.1f})", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("output")
    ax.set_title("Fitting the V with gradient descent")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = HERE / "fit_v_data_fit.png"
    fig.savefig(out1, dpi=150)

    # plot 2: loss as a function of epoch
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for (name, (p, L, fwd)), c in zip(results.items(), colors):
        ax2.plot(np.arange(1, len(L) + 1), L * SCALE ** 2, color=c, lw=2.2, label=name)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("epoch (one gradient descent step each)")
    ax2.set_ylabel("loss (mean squared error)")
    ax2.set_title("Loss as a function of epoch")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    out2 = HERE / "fit_v_loss.png"
    fig2.savefig(out2, dpi=150)
    print(f"\nplots: {out1.name}, {out2.name}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
