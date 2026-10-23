import numpy as np

import train
from model_peak import PeakThresholdModel
from synthetic import write_sessions


def test_peak_model_orders_classes_and_separates_them():
    rng = np.random.default_rng(0)
    peaks = np.concatenate([rng.normal(m, 0.05, 50) for m in (0.7, 1.0, 1.7, 3.2)])
    labels = np.repeat(["biking", "standing", "walking", "running"], 50)
    m = PeakThresholdModel().fit(peaks, labels)
    assert m.order == ["biking", "standing", "walking", "running"]
    assert m.thresholds == sorted(m.thresholds)
    assert (m.predict(peaks) == labels).mean() > 0.98
    assert list(m.predict([0.1, 1.0, 1.8, 9.0])) == ["biking", "standing", "walking", "running"]


def test_peak_model_cut_is_not_pulled_by_class_size():
    # 1000 low windows and 10 high ones, with a clear gap between 1.0 and 2.0
    peaks = np.concatenate([np.linspace(0.5, 1.0, 1000), np.linspace(2.0, 2.5, 10)])
    labels = np.array(["low"] * 1000 + ["high"] * 10)
    m = PeakThresholdModel().fit(peaks, labels)
    assert 1.0 < m.thresholds[0] < 2.0


def test_peak_model_save_and_load(tmp_path):
    m = PeakThresholdModel(["a", "b", "c"], [1.0, 2.0])
    m.save(tmp_path / "m.json")
    m2 = PeakThresholdModel.load(tmp_path / "m.json")
    assert m2.order == m.order and m2.thresholds == m.thresholds
    assert list(m2.predict([0.5, 1.5, 2.5])) == ["a", "b", "c"]


def test_train_end_to_end_on_synthetic_data(tmp_path):
    write_sessions(tmp_path / "data", per_class=3, seconds=20)
    scores = train.main(["--data", str(tmp_path / "data"), "--models", str(tmp_path / "models")])
    # Synthetic data is easy: both models should nail it. Real data won't be this clean.
    assert scores["model 1 (peak)"] > 0.9
    assert scores["model 2 (decision tree)"] > 0.9
    assert (tmp_path / "models" / "peak_model.json").exists()
    assert (tmp_path / "models" / "model2.joblib").exists()
    assert (tmp_path / "models" / "peak_hist.png").exists()


def test_kmeans_classifier_names_clusters_by_majority():
    from model_kmeans import KMeansClassifier
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (40, 3)), rng.normal(5, 0.3, (40, 3))])
    y = np.array(["standing"] * 40 + ["running"] * 40)
    m = KMeansClassifier(clusters_per_class=2).fit(X, y)
    assert set(m.cluster_labels_) == {"standing", "running"}
    assert list(m.predict([[0, 0, 0], [5, 5, 5]])) == ["standing", "running"]
