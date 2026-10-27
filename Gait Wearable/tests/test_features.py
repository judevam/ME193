import numpy as np
import pandas as pd

import features
from features import AXES, FEATURE_COLUMNS, all_features, dominant_freq, label_of, peak, windows


def session(seconds, rate=100, ax=None):
    n = int(seconds * rate)
    df = pd.DataFrame(np.zeros((n, 6)), columns=AXES)
    df["ax"] = 1.0 if ax is None else ax
    df.insert(0, "t", np.arange(n) / rate)
    return df


def test_window_count_with_trim_and_overlap():
    # 20 s, trim 3 s each end -> 14 s left; 2 s windows every 1 s -> 13 windows
    ws = windows(session(20), rate_hz=100, window_s=2, hop_s=1, trim_s=3)
    assert len(ws) == 13
    assert all(w.shape == (200, 6) for w in ws)


def test_too_short_recording_gives_no_windows():
    assert windows(session(7), rate_hz=100, window_s=2, hop_s=1, trim_s=3) == []


def test_trim_drops_start_and_end():
    df = session(10)
    df.loc[:299, "ax"] = 99.0   # first 3 s
    df.loc[700:, "ax"] = -99.0  # last 3 s
    for w in windows(df, rate_hz=100, window_s=2, hop_s=1, trim_s=3):
        assert np.all(w[:, 0] == 1.0)


def test_peak_uses_axis_and_sign():
    w = np.zeros((200, 6))
    w[:, 0] = -1.0
    w[50, 0] = -3.0
    w[60, 0] = 0.5
    assert peak(w, "ax", 1) == 0.5
    assert peak(w, "ax", -1) == 3.0


def test_dominant_frequency_of_a_sine():
    t = np.arange(400) / 100
    f, share = dominant_freq(np.sin(2 * np.pi * 1.5 * t), 100)
    assert abs(f - 1.5) <= 0.25
    assert share > 0.3


def test_dominant_frequency_of_a_constant_is_zero():
    assert dominant_freq(np.ones(200), 100) == (0.0, 0.0)


def test_features_are_finite_even_for_a_still_sensor():
    f = all_features(session(2).iloc[:, 1:].to_numpy())
    assert list(f) == FEATURE_COLUMNS
    assert all(np.isfinite(v) for v in f.values())


def test_pitch_standing_vs_flat():
    standing = np.zeros((200, 6)); standing[:, 0] = 1.0   # gravity along X (thigh vertical)
    flat = np.zeros((200, 6)); flat[:, 2] = 1.0           # gravity along Z (thigh horizontal)
    assert all_features(standing)["pitch_deg"] > 80
    assert abs(all_features(flat)["pitch_deg"]) < 10


def test_label_from_file_name():
    assert label_of("data/walking_20261022-141500.csv") == "walking"


def test_load_session_skips_header_comment(tmp_path):
    p = tmp_path / "running_x.csv"
    p.write_text("# label=running\nt,ax,ay,az,gx,gy,gz\n0.00,1,0,0,0,0,0\n0.01,2,0,0,0,0,0\n")
    df = features.load_session(p)
    assert list(df.columns) == ["t"] + AXES
    assert len(df) == 2
