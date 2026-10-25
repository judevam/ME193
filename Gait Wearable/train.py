"""Train and compare both models on the recordings in data/.

  python train.py                        # real recordings (pull_data.py first)
  python train.py --data data_synthetic  # fake ones from synthetic.py

Scores use leave-one-session-out: each recording is predicted by models trained on all
the OTHER recordings. Windows from one recording look alike, so a random split would
give fake-high scores.

Model 2 compares a decision tree, k-nearest neighbors, nearest centroid and k-means,
and keeps the best. If it's the tree, its questions are printed.

Saves to models/:
  gait_models.json  both models as plain numbers (gait_models.py runs them with numpy only;
                    dashboard.py uses it, and push_models.py sends it to the board)
  peak_model.json   model 1 on its own
  peak_hist.png     model 1's input per activity, with its thresholds
  confusion.png     every model's confusion matrix
  examples.png      a few seconds of raw signal per activity
  scores.md         the dataset, scores, thresholds and the tree's questions, for the write-up
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def export_model2(model, kind: str) -> dict:
    """A fitted model 2 -> plain numbers that gait_models.py can run without scikit-learn."""
    def scaler(s: StandardScaler) -> dict:
        return {"mean": s.mean_.tolist(), "scale": s.scale_.tolist()}

    if kind == "decision tree":
        t = model.tree_
        return {"kind": kind, "classes": model.classes_.tolist(),
                "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                "value": t.value[:, 0, :].tolist()}
    if kind == "k-nearest neighbors":
        s, knn = model.named_steps["standardscaler"], model.named_steps["kneighborsclassifier"]
        return {"kind": kind, **scaler(s), "k": knn.n_neighbors, "classes": knn.classes_.tolist(),
                "X": knn._fit_X.tolist(), "y": knn.classes_[knn._y].tolist()}
    if kind == "nearest centroid":
        s, nc = model.named_steps["standardscaler"], model.named_steps["nearestcentroid"]
        return {"kind": kind, **scaler(s), "centers": nc.centroids_.tolist(), "labels": nc.classes_.tolist()}
    if kind == "k-means":
        return {"kind": kind, **scaler(model.scaler_), "centers": model.kmeans_.cluster_centers_.tolist(),
                "labels": [str(l) for l in model.cluster_labels_]}
    raise ValueError(kind)


def export_models(m1: PeakThresholdModel, m2, kind: str) -> dict:
    """Everything the board needs to classify, as JSON-ready data."""
    return {"rate_hz": config.RATE_HZ, "window_s": config.WINDOW_S, "hop_s": config.HOP_S,
            "peak_axis": config.PEAK_AXIS, "peak_sign": config.PEAK_SIGN,
            "features": FEATURE_COLUMNS, "model1": m1.to_dict(), "model2": export_model2(m2, kind)}


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


def plot_confusions(preds: dict, df: pd.DataFrame, labels: list[str], path: Path) -> None:
    """One heatmap per model: rows = what was happening, columns = what the model said."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4.4))
    for ax, (name, pred) in zip(axes, preds.items()):
        table = confusion(df["label"], pred, labels).to_numpy()
        share = table / np.maximum(table.sum(axis=1, keepdims=True), 1)  # color = share of each row
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, table[i, j], ha="center", va="center", fontsize=11,
                        color="white" if share[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
        acc = (pred == df["label"].to_numpy()).mean()
        ax.set_title(f"{name}\n{acc:.1%} right", fontsize=10)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("actual")
    fig.suptitle("Leave-one-recording-out confusion matrices (counts of 2 s windows)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))  # leave room for the title
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(paths: list[Path], path: Path, seconds: float = 4.0) -> None:
    """A few seconds of each activity from its longest recording: thigh-axis accel and gyro size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from features import label_of, load_session

    longest = {}
    for p in paths:
        df = load_session(p)
        label = label_of(p)
        if label not in longest or len(df) > len(longest[label]):
            longest[label] = df
    labels = [l for l in config.LABELS if l in longest]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 2 * len(labels)), sharex=True)
    n = int(seconds * config.RATE_HZ)
    for ax, label in zip(np.atleast_1d(axes), labels):
        df = longest[label]
        mid = len(df) // 2
        w = df.iloc[mid - n // 2: mid + n // 2]
        t = np.arange(len(w)) / config.RATE_HZ
        ax.plot(t, w[config.PEAK_AXIS], color="tab:red", label=f"{config.PEAK_AXIS} (thigh axis), g")
        ax.set_ylabel(label)
        ax2 = ax.twinx()
        ax2.plot(t, np.linalg.norm(w[["gx", "gy", "gz"]], axis=1), color="tab:blue", alpha=0.5,
                 label="|gyro|, deg/s")
        ax2.set_ylim(0, 600)
        ax.set_ylim(-1.5, 5)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("seconds")
    fig.suptitle(f"{seconds:g} s of each activity (middle of its longest recording)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_scores(path: Path, df: pd.DataFrame, labels: list[str], scores: dict,
                 m1: PeakThresholdModel, best: str, tree_text: str, n_recordings: int) -> None:
    """The numbers for the write-up, as Markdown."""
    counts = df.groupby("label")["session"].agg(["nunique", "size"]).reindex(labels)
    lines = ["# Gait Wearable: results", "",
             f"{n_recordings} recordings, {len(df)} windows of {config.WINDOW_S:g} s "
             f"(every {config.HOP_S:g} s), scored leave-one-recording-out.", "",
             "| Activity | Recordings | Windows |", "|---|---|---|"]
    lines += [f"| {l} | {r['nunique']} | {r['size']} |" for l, r in counts.iterrows()]
    lines += ["", "| Model | Windows right |", "|---|---|"]
    lines += [f"| {name}{' (kept)' if name == f'model 2 ({best})' else ''} | {s:.1%} |"
              for name, s in scores.items()]
    lines += ["", f"Model 1 thresholds (peak {config.PEAK_AXIS}): {m1.describe()}", "",
              "Decision tree trained on all windows (its questions explain what separates "
              "the activities):", "", "```", tree_text.rstrip(), "```", "",
              "Figures: `peak_hist.png` (model 1), `confusion.png` (all models), "
              "`examples.png` (raw signals)."]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> dict[str, float]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATA_DIR)
    p.add_argument("--models", type=Path, default=config.MODEL_DIR)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--split", type=int, default=1,
                   help="cut each recording into N chunks scored as separate sessions "
                        "(for pilot data with only one recording per activity)")
    args = p.parse_args(argv)

    paths = sorted(args.data.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no recordings in {args.data}")
    df = build_dataset(paths)
    if args.split > 1:
        # Neighbouring windows overlap by 50 %, so chunks aren't fully independent:
        # scores come out a bit optimistic. Fine for a pilot, not for the write-up.
        n = df.groupby("session")["window"].transform("max") + 1
        chunk = (df["window"] * args.split // n).astype(str)
        df["session"] = df["session"] + "#" + chunk
        print(f"(each recording split into {args.split} chunks; scores are a bit optimistic)")
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
    # A tree is easy to read even when it isn't the one kept, so always show its questions
    tree = m2 if isinstance(m2, DecisionTreeClassifier) else \
        MODEL2_CANDIDATES["decision tree"]().fit(df[FEATURE_COLUMNS].to_numpy(), df["label"].to_numpy())
    tree_text = export_text(tree, feature_names=FEATURE_COLUMNS, decimals=2)
    print("  the decision tree's questions:")
    print(tree_text)

    if not args.no_save:
        args.models.mkdir(exist_ok=True)
        m1.save(args.models / "peak_model.json")
        (args.models / "gait_models.json").write_text(json.dumps(export_models(m1, m2, best)))
        plot_peaks(df, m1, args.models / "peak_hist.png")
        plot_confusions(preds, df, labels, args.models / "confusion.png")
        plot_examples(paths, args.models / "examples.png")
        write_scores(args.models / "scores.md", df, labels, scores, m1, best, tree_text, len(paths))
        print(f"saved to {args.models}: gait_models.json, peak_hist.png, confusion.png, "
              "examples.png, scores.md")
    return scores


if __name__ == "__main__":
    main()
