"""SwingDetector against hand-made samples and the Phase 1 recordings (tests/fixtures/)."""

from pathlib import Path

import pytest

from imu_paddle import ImuSample, load_csv
from swing import SwingDetector

FIXTURES = Path(__file__).parent / "fixtures"


WHIP = 6000.0   # accel size (milli-g) of a real swing's arc; resting is 1000 (gravity)


def sample(t, gx=0.0, gy=0.0, gz=0.0, accel=1000.0):
    return ImuSample(t, gx, gy, gz, 0.0, 0.0, accel, -1)


def detector():
    return SwingDetector(threshold=600, accel_threshold=3000, pair_window=0.15, cooldown=0.4)


# ── Recordings: every real swing exactly once, nothing else ─────────────────

@pytest.mark.parametrize("name, expected", [
    ("still", 0), ("walk", 0), ("wiggle", 0),
    ("swing_soft", 7), ("swing", 7),
])
def test_recordings(name, expected):
    swings = SwingDetector().update_many(load_csv(FIXTURES / f"{name}.csv"))
    assert len(swings) == expected


def test_backswing_and_forward_stroke_count_once():
    # At t=14.97 the backswing (gy +700) crosses first, then the forward stroke (gy -1500)
    swings = SwingDetector().update_many(load_csv(FIXTURES / "swing.csv"))
    near = [s for s in swings if 14.5 < s.t < 15.6]
    assert len(near) == 1


# ── Hand-made samples ────────────────────────────────────────────────────────

def test_rotation_with_whip_is_a_swing():
    assert detector().update(sample(0.0, gy=-900, accel=WHIP))


def test_below_gyro_threshold_is_not_a_swing():
    assert detector().update(sample(0.0, gy=599, accel=WHIP)) is None


def test_twist_in_place_is_not_a_swing():
    # fast rotation, but the motor isn't moving through space: accel stays near 1 g
    d = detector()
    twist = [sample(i * 0.016, gx=2000, accel=1300) for i in range(30)]
    assert d.update_many(twist) == []


def test_knock_without_rotation_is_not_a_swing():
    assert detector().update(sample(0.0, gy=50, accel=8000)) is None


def test_forehand_and_backhand_both_count():
    d = detector()
    assert d.update(sample(0.0, gy=-900, accel=WHIP))
    assert d.update(sample(1.0, gy=+900, accel=WHIP))


def test_uses_all_three_gyro_axes():
    # 400 on each axis is only ~693 together: under 600 on any one axis, over it in total
    swing = detector().update(sample(0.0, gx=400, gy=400, gz=400, accel=WHIP))
    assert swing is not None
    assert swing.speed == pytest.approx(692.8, abs=0.1)


def test_gyro_then_accel_within_window_fires_on_the_accel():
    # big backswing: rotation first, the whip arrives 60 ms later on the forward stroke
    d = detector()
    assert d.update(sample(0.00, gy=+900)) is None
    swing = d.update(sample(0.06, gy=-500, accel=WHIP))
    assert swing and swing.t == 0.06


def test_gyro_and_accel_too_far_apart_is_not_a_swing():
    d = detector()
    assert d.update(sample(0.0, gy=+900)) is None
    assert d.update(sample(0.3, gy=100, accel=WHIP)) is None


def test_one_swing_reports_once():
    swing_samples = [sample(i * 0.016, gy=-1500, accel=WHIP) for i in range(15)]  # 0.24 s
    assert len(detector().update_many(swing_samples)) == 1


def test_follow_through_bounce_is_ignored():
    d = detector()
    assert d.update(sample(0.00, gy=-1500, accel=WHIP))
    assert d.update(sample(0.30, gz=+700, accel=WHIP)) is None   # bounce inside the cooldown


def test_next_swing_after_cooldown_counts():
    d = detector()
    assert d.update(sample(0.0, gy=-1500, accel=WHIP))
    assert d.update(sample(0.41, gy=-1500, accel=WHIP))
