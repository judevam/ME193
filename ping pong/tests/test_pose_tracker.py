"""Lane mapping and wrist history (no camera or mediapipe needed)."""

import pytest

from pose_tracker import LEFT, CENTER, RIGHT, Smoother, Wrist, WristHistory, hand_lane, lane_of


@pytest.mark.parametrize("x, lane", [(0.0, LEFT), (0.30, LEFT), (0.5, CENTER), (0.70, RIGHT), (1.0, RIGHT)])
def test_lane_of(x, lane):
    assert lane_of(x) == lane


def test_history_empty():
    assert WristHistory().at(1.0) is None


def test_history_returns_last_position_before_the_swing():
    h = WristHistory(max_age=0.35)
    h.add(Wrist(1.00, 0.2, 0.5))
    h.add(Wrist(1.03, 0.5, 0.5))
    h.add(Wrist(1.06, 0.8, 0.5))       # after the swing: not used
    assert h.at(1.05).x == 0.5


def test_history_too_old_is_none():
    # e.g. the wrist blurred out during the swing and nothing new was seen
    h = WristHistory(max_age=0.35)
    h.add(Wrist(1.0, 0.5, 0.5))
    assert h.at(1.30).x == 0.5
    assert h.at(1.40) is None


def test_history_swing_before_any_wrist_is_none():
    h = WristHistory()
    h.add(Wrist(2.0, 0.5, 0.5))
    assert h.at(1.9) is None


# ── Smoothing ────────────────────────────────────────────────────────────────


def test_first_position_is_not_smoothed():
    s = Smoother(amount=0.5)
    assert s(Wrist(0.0, 0.8, 0.4)).x == 0.8


def test_smoothing_moves_part_way():
    s = Smoother(amount=0.5)
    s(Wrist(0.00, 0.2, 0.5))
    w = s(Wrist(0.03, 0.6, 0.5))
    assert w.x == pytest.approx(0.4)
    assert w.t == 0.03


def test_no_smoothing_when_amount_is_one():
    s = Smoother(amount=1.0)
    s(Wrist(0.00, 0.2, 0.5))
    assert s(Wrist(0.03, 0.6, 0.5)).x == pytest.approx(0.6)


def test_smoothing_restarts_after_wrist_was_lost():
    s = Smoother(amount=0.5, reset_after=0.35)
    s(Wrist(0.0, 0.2, 0.5))
    assert s(Wrist(1.0, 0.8, 0.5)).x == 0.8      # 1 s later: don't drag it from the old spot


def test_steady_hand_settles_on_its_position():
    s = Smoother(amount=0.6)
    s(Wrist(0.0, 0.2, 0.5))
    for i in range(1, 15):
        w = s(Wrist(i * 0.03, 0.7, 0.5))
    assert w.x == pytest.approx(0.7, abs=1e-3)


# ── Lane tolerance ───────────────────────────────────────────────────────────


def test_hand_just_over_the_line_still_counts_for_the_ball_lane():
    # center lane is 1/3..2/3; 0.31 is just over into "left" but within the 0.05 tolerance
    assert hand_lane(0.31, CENTER, tolerance=0.05) == CENTER


def test_hand_clearly_in_another_lane_is_that_lane():
    assert hand_lane(0.15, CENTER, tolerance=0.05) == LEFT
    assert hand_lane(0.90, LEFT, tolerance=0.05) == RIGHT
