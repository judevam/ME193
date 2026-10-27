"""The numpy-only models (gait_models.py) must match scikit-learn exactly."""
import json

import numpy as np
import pytest

import config
import train
from features import FEATURE_COLUMNS, build_dataset
from gait_models import GaitModels, StreamClassifier
from model_peak import PeakThresholdModel
from synthetic import generate, write_sessions


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return build_dataset(write_sessions(d, per_class=3, seconds=20, seed=4))


@pytest.mark.parametrize("kind", list(train.MODEL2_CANDIDATES))
def test_exported_model2_matches_sklearn(dataset, kind):
    X, y = dataset[FEATURE_COLUMNS].to_numpy(), dataset["label"].to_numpy()
    # Fit on half the sessions, compare on all windows (including unseen ones)
    sessions = sorted(dataset["session"].unique())
    train_rows = dataset["session"].isin(sessions[::2]).to_numpy()
    model = train.MODEL2_CANDIDATES[kind]().fit(X[train_rows], y[train_rows])
    m1 = PeakThresholdModel().fit(dataset["peak"], y)
    exported = json.loads(json.dumps(train.export_models(m1, model, kind)))  # real JSON round trip
    portable = GaitModels(exported)

    expected = model.predict(X)
    got = [portable._model2(x)[0] for x in X]
    assert list(got) == list(expected)


def test_exported_model1_matches(dataset):
    m1 = PeakThresholdModel().fit(dataset["peak"], dataset["label"])
    model2 = train.MODEL2_CANDIDATES["decision tree"]().fit(dataset[FEATURE_COLUMNS].to_numpy(), dataset["label"])
    portable = GaitModels(train.export_models(m1, model2, "decision tree"))
    peaks = np.linspace(-0.5, 4, 200)
    assert [portable.order[int(np.searchsorted(portable.thresholds, p, side="right"))] for p in peaks] \
        == list(m1.predict(peaks))


def test_peak_only_models_have_no_answers():
    m = GaitModels.peak_only(100, 2.0, 1.0, "ax", 1)
    assert not m.has_models
    w = np.zeros((200, 6)); w[:, 0] = 1.0
    assert m.classify(w) == {"peak": 1.0, "model1": None, "model2": None, "confidence": None}


def test_stream_classifier_timing_and_smoothing():
    m = GaitModels.peak_only(100, 2.0, 1.0, "ax", 1)
    s = StreamClassifier(m)
    results = s.add([[1, 0, 0, 0, 0, 0]] * 450)
    assert [i for i, _ in results] == [199, 299, 399]    # first full window, then every second
    s.reset()
    assert s.add([[1, 0, 0, 0, 0, 0]] * 150) == []         # window starts over after a gap
