"""Game rules with a fake clock: level 1 travel is 1.6 s, window is -0.20 / +0.25 s."""

import pytest

import config
import random

from game import Game, SERVE, INCOMING, RETURNING, MISSED

TRAVEL = config.LEVEL_TRAVEL_S[1]
SERVE_AT = config.SERVE_DELAY_S          # ball leaves the wall
ARRIVE = SERVE_AT + TRAVEL               # ball reaches you on the first serve


def started():
    g = Game(level=1)
    g.start(0.0)
    g.update(SERVE_AT)
    assert g.state == INCOMING
    return g


def test_serve_waits_then_ball_comes_in():
    g = Game(level=1)
    assert g.start(0.0) == ["serve"]
    g.update(SERVE_AT - 0.01)
    assert g.state == SERVE
    g.update(SERVE_AT)
    assert g.state == INCOMING
    assert g.ball_z(SERVE_AT + TRAVEL / 2) == pytest.approx(0.5)


@pytest.mark.parametrize("offset", [-config.HIT_EARLY_S, 0.0, config.HIT_LATE_S])
def test_swing_in_window_is_a_hit(offset):
    g = started()
    assert g.swing(ARRIVE + offset) == ["hit"]
    assert g.streak == 1
    assert g.state == RETURNING


def test_too_early_is_a_whiff_not_a_miss():
    g = started()
    assert g.swing(ARRIVE - config.HIT_EARLY_S - 0.01) == ["whiff_early"]
    assert g.state == INCOMING            # ball still coming
    assert g.swing(ARRIVE) == ["hit"]     # swing again in time: still a hit


def test_whiff_does_not_end_the_streak():
    g = started()
    g.swing(ARRIVE)                       # streak 1
    g.update(ARRIVE + TRAVEL)
    second = ARRIVE + 2 * TRAVEL
    assert g.swing(second - config.HIT_EARLY_S - 0.05) == ["whiff_early"]
    assert g.streak == 1
    assert g.swing(second) == ["hit"]
    assert g.streak == 2


def test_swing_while_ball_is_far_away_is_ignored():
    g = started()
    far = SERVE_AT + (config.WHIFF_FROM_Z - 0.05) * TRAVEL
    assert g.swing(far) == []
    assert g.state == INCOMING
    assert g.swing(ARRIVE) == ["hit"]


def test_swing_in_second_half_before_window_is_a_whiff():
    g = started()
    assert g.swing(SERVE_AT + (config.WHIFF_FROM_Z + 0.05) * TRAVEL) == ["whiff_early"]


def test_no_swing_is_a_late_miss():
    g = started()
    assert g.update(ARRIVE + config.HIT_LATE_S) == []          # window still open
    assert g.update(ARRIVE + config.HIT_LATE_S + 0.01) == ["miss"]
    assert g.state == MISSED
    assert g.miss_reason == "late"


def test_miss_reason_is_the_last_whiff():
    g = started()
    g.swing(ARRIVE - config.HIT_EARLY_S - 0.05)                # too early...
    g.swing(ARRIVE, hand_lane=(g.lane + 1) % 3)                # ...then wrong spot
    assert g.update(ARRIVE + config.HIT_LATE_S + 0.01) == ["miss"]
    assert g.miss_reason == "lane"


def test_whiff_reason_resets_for_the_next_ball():
    g = started()
    g.swing(ARRIVE - config.HIT_EARLY_S - 0.05)                # whiff, then hit
    g.swing(ARRIVE)
    g.update(ARRIVE + TRAVEL)                                  # next ball: no swing at all
    g.update(ARRIVE + 2 * TRAVEL + config.HIT_LATE_S + 0.01)
    assert g.miss_reason == "late"


def test_swing_reported_after_frame_but_timestamped_in_window_still_hits():
    # BLE batches: the swing arrives with an in-window timestamp; process swings before update()
    g = started()
    assert g.swing(ARRIVE + 0.1) == ["hit"]
    assert g.update(ARRIVE + 0.3) == []


def test_one_hit_per_ball():
    g = started()
    g.swing(ARRIVE)
    assert g.swing(ARRIVE + 0.1) == []      # ball is going away: extra swings do nothing
    assert g.streak == 1


def test_ball_bounces_off_wall_and_comes_back():
    g = started()
    g.swing(ARRIVE)
    assert g.ball_z(ARRIVE + TRAVEL / 2) == pytest.approx(0.5)
    g.update(ARRIVE + TRAVEL)
    assert g.state == INCOMING
    second_arrival = ARRIVE + 2 * TRAVEL
    assert g.swing(second_arrival) == ["hit"]
    assert g.streak == 2


def test_rally_timing_does_not_drift_with_late_hits():
    # each return trip starts from the hit time, so a late hit gives a late return
    g = started()
    g.swing(ARRIVE + 0.2)
    g.update(ARRIVE + 0.2 + TRAVEL)
    assert g.swing(ARRIVE + 0.2 + 2 * TRAVEL) == ["hit"]


def test_miss_resets_streak_but_keeps_best():
    g = started()
    t = ARRIVE
    for _ in range(3):                       # three hits in a row
        assert g.swing(t) == ["hit"]
        g.update(t + TRAVEL)
        t += 2 * TRAVEL
    assert g.streak == 3
    g.update(t + config.HIT_LATE_S + 0.01)   # then let one go by
    assert g.streak == 0
    assert g.best == 3


def test_new_serve_after_miss_pause():
    g = started()
    g.update(ARRIVE + 1.0)                   # missed
    miss_t = ARRIVE + config.HIT_LATE_S
    assert g.update(miss_t + config.MISS_PAUSE_S - 0.01) == []
    assert g.update(miss_t + config.MISS_PAUSE_S) == ["serve"]
    assert g.state == SERVE


def test_swings_while_serving_or_missed_do_nothing():
    g = Game(level=1)
    g.start(0.0)
    assert g.swing(0.5) == []
    g.update(SERVE_AT)
    g.update(ARRIVE + 1.0)                   # missed
    assert g.swing(ARRIVE + 1.1) == []
    assert g.state == MISSED


def test_higher_level_is_faster():
    assert Game(level=3).travel < Game(level=2).travel < Game(level=1).travel


def test_hit_window_flag():
    g = started()
    assert not g.in_hit_window(ARRIVE - config.HIT_EARLY_S - 0.01)
    assert g.in_hit_window(ARRIVE)
    assert not g.in_hit_window(ARRIVE + config.HIT_LATE_S + 0.01)


def test_swing_timing_is_recorded():
    g = started()
    assert g.last_offset is None
    g.swing(ARRIVE - 0.1)
    assert g.last_offset == pytest.approx(-0.1)


def test_ignored_far_swing_does_not_record_timing():
    g = started()
    g.swing(SERVE_AT + 0.1)
    assert g.last_offset is None



# ── Lanes (Phase 5) ──────────────────────────────────────────────────────────

def test_hand_in_the_ball_lane_is_a_hit():
    g = started()
    assert g.swing(ARRIVE, hand_lane=g.lane) == ["hit"]


def test_hand_in_the_wrong_lane_is_a_whiff():
    g = started()
    wrong = (g.lane + 1) % 3
    assert g.swing(ARRIVE - 0.1, hand_lane=wrong) == ["whiff_lane"]
    assert g.state == INCOMING
    assert g.swing(ARRIVE + 0.1, hand_lane=g.lane) == ["hit"]   # moved over in time


def test_hand_not_seen_is_a_whiff():
    g = started()
    assert g.swing(ARRIVE, hand_lane=None) == ["whiff_no_hand"]
    assert g.state == INCOMING


def test_lane_not_checked_for_early_swings():
    # too early is reported as too early, whatever the hand is doing
    g = started()
    assert g.swing(ARRIVE - config.HIT_EARLY_S - 0.01, hand_lane=None) == ["whiff_early"]


def test_far_practice_swing_with_hand_anywhere_is_ignored():
    g = started()
    assert g.swing(SERVE_AT + 0.1, hand_lane=None) == []


def test_every_lane_comes_up():
    g = Game(level=1, rng=random.Random(0))
    g.start(0.0)
    g.update(SERVE_AT)
    lanes, t = {g.lane}, ARRIVE
    for _ in range(30):                      # rally: each return off the wall picks a new lane
        g.swing(t, hand_lane=g.lane)
        g.update(t + TRAVEL)
        lanes.add(g.lane)
        t += 2 * TRAVEL
    assert lanes == {0, 1, 2}
    assert g.streak == 30


# ── Drawing helpers (feel pass) ──────────────────────────────────────────────

def test_time_to_arrival():
    g = started()
    assert g.time_to_arrival(ARRIVE - 0.5) == pytest.approx(0.5)
    g.swing(ARRIVE)
    assert g.time_to_arrival(ARRIVE + 0.1) is None      # going away


def test_missed_ball_keeps_flying_past():
    g = started()
    g.update(ARRIVE + config.HIT_LATE_S + 0.01)          # missed
    z1 = g.ball_z(ARRIVE + 0.5)
    z2 = g.ball_z(ARRIVE + 0.8)
    assert 1.0 < z1 < z2
