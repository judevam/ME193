"""Fake recordings with roughly the right shape for each activity, for tests and for trying
train.py before real data exists. Not a substitute for real data!

  python synthetic.py                 # writes data_synthetic/, 5 sessions per activity
  python train.py --data data_synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _impulses(t: np.ndarray, freq: float, height: float, width_s: float = 0.04) -> np.ndarray:
    """A sharp bump once per stride (heel strike)."""
    phase = (t * freq) % 1.0
    return height * np.exp(-0.5 * (phase / (width_s * freq)) ** 2)


def generate(label: str, seconds: float, rng: np.random.Generator, rate_hz: float = config.RATE_HZ) -> pd.DataFrame:
    """The thigh axis (config.PEAK_AXIS) points toward the hip, so standing reads ~ +1 g on it."""
    t = np.arange(int(seconds * rate_hz)) / rate_hz
    n = len(t)
    jitter = rng.uniform(0.85, 1.15)  # each session a bit different
    if label == "standing":
        theta = np.radians(rng.uniform(80, 90)) * np.ones(n)  # thigh angle from horizontal
        extra = np.zeros(n)
        gyro_y = np.zeros(n)
    elif label in ("walking", "running"):
        stride = (0.9 if label == "walking" else 1.4) * jitter          # Hz, one leg
        swing = np.radians(25 if label == "walking" else 45)
        theta = np.radians(75) + swing * np.sin(2 * np.pi * stride * t)
        extra = _impulses(t, stride, (0.7 if label == "walking" else 2.5) * jitter)
        gyro_y = np.degrees(swing * 2 * np.pi * stride * np.cos(2 * np.pi * stride * t))
    elif label == "biking":
        cadence = 1.2 * jitter
        swing = np.radians(rng.uniform(18, 28))
        theta = np.radians(rng.uniform(10, 30)) + swing * np.sin(2 * np.pi * cadence * t)
        extra = np.zeros(n)
        gyro_y = np.degrees(swing * 2 * np.pi * cadence * np.cos(2 * np.pi * cadence * t))
    else:
        raise ValueError(label)
    noise = lambda s: rng.normal(0, s, n)
    df = pd.DataFrame({
        "t": np.round(t, 2),
        "ax": np.sin(theta) + extra + noise(0.03),
        "ay": noise(0.05 if label == "standing" else 0.15),
        "az": np.cos(theta) + 0.3 * extra + noise(0.03),
        "gx": noise(2 if label == "standing" else 15),
        "gy": gyro_y + noise(2),
        "gz": noise(2 if label == "standing" else 10),
    })
    # Built with the thigh along X; move it to the axis the real IMU uses (config.PEAK_AXIS)
    thigh = config.PEAK_AXIS[1]
    if thigh != "x":
        df[["ax", f"a{thigh}"]] = df[[f"a{thigh}", "ax"]].to_numpy()
        df[["gx", f"g{thigh}"]] = df[[f"g{thigh}", "gx"]].to_numpy()
    return df


def write_sessions(out_dir: Path, per_class: int = 5, seconds: float = 40, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for label in config.LABELS:
        for i in range(per_class):
            path = out_dir / f"{label}_synthetic-{i:02d}.csv"
            with open(path, "w", newline="") as f:
                f.write(f"# label={label} synthetic=1 rate_hz={config.RATE_HZ}\n")
                generate(label, seconds, rng).to_csv(f, index=False, float_format="%.4f")
            paths.append(path)
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=config.HERE / "data_synthetic")
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--seconds", type=float, default=40)
    args = p.parse_args()
    paths = write_sessions(args.out, args.per_class, args.seconds)
    print(f"wrote {len(paths)} recordings to {args.out}")
