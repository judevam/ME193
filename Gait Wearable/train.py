"""Train and compare both models on the recordings in data/.

  python train.py                        # real recordings (pull_data.py first)
  python train.py --data data_synthetic  # fake ones from synthetic.py

Scores use leave-one-session-out: each recording is predicted by models trained on all
the OTHER recordings. Windows from one recording look alike, so a random split would
give fake-high scores.

Model 2 compares a decision tree, k-nearest neighbors, nearest centroid and k-means,
and keeps the best. If it's the tree, its questions are printed.

Saves to models/: peak_model.json (model 1), model2.joblib (best model 2),
peak_hist.png (model 1's input per activity, with its thresholds).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import config
from features import FEATURE_COLUMNS, build_dataset
from model_kmeans import KMeansClassifier
from model_peak import PeakThresholdModel

# Model 2 options, all from class (no neural nets). The distance-based ones standardize the
# features first, so gyro (hundreds of deg/s) doesn't outweigh accel (a few g).
MODEL2_CANDIDATES = {
    "decision tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=0),
    "k-nearest neighbors": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "nearest centroid": lambda: make_pipeline(StandardScaler(), NearestCentroid()),
    "k-means": lambda: KMeansClassifier(clusters_per_class=2),
}


def cross_validate(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Leave-one-session-out predictions for every window, per model."""
    X, y, groups = df[FEATURE_COLUMNS].to_numpy(), df["label"].to_numpy(), df["session"].to_numpy()
    preds = {"model 1 (peak)": np.empty(len(df), dtype=object),
             **{f"model 2 ({k})": np.empty(len(df), dtype=object) for k in MODEL2_CANDIDATES}}
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        m1 = PeakThresholdModel().fit(df["peak"].iloc[train], y[train])
        preds["model 1 (peak)"][test] = m1.predict(df["peak"].iloc[test])
        for name, make in MODEL2_CANDIDATES.items():
            preds[f"model 2 ({name})"][test] = make().fit(X[train], y[train]).predict(X[test])
    return preds


def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(y_true, name="actual"), pd.Series(y_pred, name="predicted"))
    return table.reindex(index=labels, columns=labels, fill_value=0)


def plot_peaks(df: pd.DataFrame, model: PeakThresholdModel, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(df["peak"].min() - 0.1, df["peak"].max() + 0.1, 60)
    for label in model.order:
        ax.hist(df.loc[df["label"] == label, "peak"], bins=bins, alpha=0.6, label=label)
    for t in model.thresholds:
        ax.axvline(t, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"peak {config.PEAK_AXIS} in a {config.WINDOW_S:g} s window (g)")
    ax.set_ylabel("windows")
    ax.set_title("Model 1 input by activity (dashed = thresholds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict[str, float]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATA_DIR)
    p.add_argument("--models", type=Path, default=config.MODEL_DIR)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    paths = sorted(args.data.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no recordings in {args.data}")
    df = build_dataset(paths)
    labels = [l for l in config.LABELS if l in set(df["label"])]
    print(f"{len(paths)} recordings, {len(df)} windows")
    print(df.groupby("label")["session"].agg(sessions="nunique", windows="size").reindex(labels).to_string(), "\n")

    if df["session"].nunique() < 2:
        raise SystemExit("need at least 2 recordings to cross-validate")
    preds = cross_validate(df)
    scores = {}
    for name, pred in preds.items():
        scores[name] = float((pred == df["label"].to_numpy()).mean())
        print(f"== {name}: {scores[name]:.1%} of windows right ==")
        print(confusion(df["label"], pred, labels).to_string(), "\n")

    # Final models, trained on everything
    m1 = PeakThresholdModel().fit(df["peak"], df["label"])
    print("model 1 thresholds:", m1.describe())
    best = max(MODEL2_CANDIDATES, key=lambda k: scores[f"model 2 ({k})"])
    m2 = MODEL2_CANDIDATES[best]().fit(df[FEATURE_COLUMNS].to_numpy(), df["label"].to_numpy())
    print(f"model 2: keeping the {best}")
    if isinstance(m2, DecisionTreeClassifier):
        print("  the tree's questions:")
        print(export_text(m2, feature_names=FEATURE_COLUMNS, decimals=2))

    if not args.no_save:
        args.models.mkdir(exist_ok=True)
        m1.save(args.models / "peak_model.json")
        joblib.dump({"model": m2, "features": FEATURE_COLUMNS, "kind": best}, args.models / "model2.joblib")
        plot_peaks(df, m1, args.models / "peak_hist.png")
        print(f"saved to {args.models}")
    return scores


if __name__ == "__main__":
    main()
