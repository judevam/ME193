import json

import numpy as np
import pytest

import train
from live import LiveGait, parse_imu
from synthetic import generate, write_sessions


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("gait")
    write_sessions(d / "data", per_class=3, seconds=20)
    train.main(["--data", str(d / "data"), "--models", str(d / "models")])
    return d / "models"


def stream(live, label, seconds, seq0=0, rng=None):
    data = generate(label, seconds, rng or np.random.default_rng(1))[["ax", "ay", "az", "gx", "gy", "gz"]]
    rows = data.to_numpy().tolist()
    for i in range(0, len(rows), 20):
        live.add(rows[i:i + 20], seq0 + i)
    return seq0 + len(rows)


def test_no_models_still_tracks_peaks():
    live = LiveGait(models_dir=None)
    stream(live, "standing", 5)
    assert not live.has_models
    assert len(live.peaks) == 4          # windows ending at 2, 3, 4, 5 s
    assert "gait" not in live.current


@pytest.mark.parametrize("label", ["standing", "walking", "running", "biking"])
def test_live_classifies_synthetic_activities(models, label):
    live = LiveGait(models)
    stream(live, label, 8, rng=np.random.default_rng(7))
    assert live.current["gait"] == label
    assert live.current["model1"] == label
    assert live.current["model2"] == label
    conf = live.current["confidence"]
    assert conf is None or 0 < conf <= 1


def test_smoothing_needs_a_majority_to_switch(models):
    live = LiveGait(models)
    seq = stream(live, "standing", 6)
    stream(live, "running", 1.2, seq0=seq)  # one window that's mostly still standing
    assert live.current["gait"] == "standing"


def test_gap_in_seq_is_counted_and_skipped():
    live = LiveGait(models_dir=None)
    live.add([[0, 0, 1, 0, 0, 0]] * 20, seq=0)
    live.add([[0, 0, 1, 0, 0, 0]] * 20, seq=60)   # 40 samples missing
    assert live.lost == 40
    assert live.total == 80
    assert len(live.samples) == 20                # the buffer restarted after the gap


def test_recent_returns_times_ending_now():
    live = LiveGait(models_dir=None)
    live.add([[i, 0, 0, 0, 0, 0] for i in range(300)], seq=0)
    t, data = live.recent(1.0)
    assert data.shape == (100, 6)
    assert t[-1] == pytest.approx(live.now)
    assert t[-1] - t[0] == pytest.approx(0.99)


def test_parse_imu():
    seq, s = parse_imu(json.dumps({"seq": 40, "rate_hz": 100, "s": [[1, 2, 3, 4, 5, 6]]}).encode())
    assert seq == 40 and s == [[1, 2, 3, 4, 5, 6]]
