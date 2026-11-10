"""Fit a "V" shaped data set with a tiny PyTorch neural net: x -> ReLU layer -> straight-line sum.

    y_hat = sum_j  w2[j] * relu(w1[j] * x + b1[j])  +  b2

Each ReLU unit is a "hinge": flat on one side, a straight line on the other. Adding two
hinges that face opposite ways makes a V. The weights are found by minimizing the mean
squared error (the loss) with gradient descent; PyTorch works out the gradients (autograd).

    python fit_v.py                       # uses annacode-v.tsv next to this file
    python fit_v.py other.tsv --hidden 2 --steps 5000
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load(path):
    """TSV with a header row: time, x, output. Returns x and y as 1-D arrays."""
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    return data[:, 1], data[:, 2]


def make_model(hidden, x, seed):
    """Linear(1 -> hidden) -> ReLU -> Linear(hidden -> 1)."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    # Start each unit's kink (where w1*x + b1 = 0) at a random data point. A kink outside
    # the data leaves the unit always off (no gradient, it never learns) or always on.
    with torch.no_grad():
        layer = model[0]
        sign = torch.where(torch.rand(hidden) < 0.5, -1.0, 1.0)
        layer.weight.copy_((sign * (0.5 + torch.rand(hidden))).unsqueeze(1))
        kinks = x[torch.randint(len(x), (hidden,))].squeeze(1)
        layer.bias.copy_(-layer.weight.squeeze(1) * kinks)
    return model


def train(x, y, hidden=4, steps=4000, lr=0.05, seed=0):
    """Gradient descent on mean squared error. x and y are (n, 1) tensors, roughly unit-scaled."""
    model = make_model(hidden, x, seed)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)   # forward pass
        loss.backward()               # backward pass: d(loss)/d(each weight)
        opt.step()                    # step downhill
        losses.append(loss.item())
    return model, np.array(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="?", default=HERE / "annacode-v.tsv")
    p.add_argument("--hidden", type=int, default=4, help="number of ReLU units (2 is the minimum for a V)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.05, help="learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restarts", type=int, default=5, help="random starts to try; the best is kept")
    args = p.parse_args()

    x, y = load(args.data)

    # Scale to mean 0, std 1 so one learning rate works for every weight.
    x_mu, x_sd, y_mu, y_sd = x.mean(), x.std(), y.mean(), y.std()
    xs = torch.tensor((x - x_mu) / x_sd, dtype=torch.float32).unsqueeze(1)
    ys = torch.tensor((y - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)

    # Gradient descent can get stuck (e.g. both hinges facing the same way), so train from
    # a few random starts and keep the one with the lowest final loss.
    runs = [train(xs, ys, args.hidden, args.steps, args.lr, args.seed + r) for r in range(args.restarts)]
    net, losses = min(runs, key=lambda run: run[1][-1])
    print("final loss of each start (scaled):", " ".join(f"{run[1][-1]:.4f}" for run in runs))

    def model(x_raw):
        with torch.no_grad():
            xt = torch.tensor((x_raw - x_mu) / x_sd, dtype=torch.float32).unsqueeze(1)
            return net(xt).squeeze(1).numpy() * y_sd + y_mu

    y_fit = model(x)
    mse = np.mean((y_fit - y) ** 2)
    print(f"{len(x)} points, {args.hidden} ReLU units, {args.steps} steps")
    print(f"final loss (MSE, original units): {mse:.2f}   RMS error: {np.sqrt(mse):.2f}")

    # Read the V off the fitted curve: the bottom, and the slope on each side.
    grid = np.linspace(x.min(), x.max(), 2001)
    g = model(grid)
    i = int(np.argmin(g))
    left = np.polyfit(grid[: i + 1], g[: i + 1], 1)[0] if i > 10 else float("nan")
    right = np.polyfit(grid[i:], g[i:], 1)[0] if i < len(grid) - 10 else float("nan")
    print(f"bottom of the V: x = {grid[i]:.2f}, output = {g[i]:.2f}")
    print(f"slopes: left {left:.3f}, right {right:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.scatter(x, y, s=16, color="#2a78d6", label="data")
    a1.plot(grid, g, color="#eb6834", lw=2, label=f"ReLU net ({args.hidden} units)")
    a1.set_xlabel("x"); a1.set_ylabel("output"); a1.legend()
    a1.set_title(f"Fit: bottom at x = {grid[i]:.1f}, RMS error {np.sqrt(mse):.2f}")
    a2.plot(losses * y_sd**2, color="#2a78d6")
    a2.set_yscale("log"); a2.set_xlabel("gradient descent step"); a2.set_ylabel("loss (MSE)")
    a2.set_title("Loss going down")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE / "fit_v.png"
    fig.savefig(out, dpi=150)
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
