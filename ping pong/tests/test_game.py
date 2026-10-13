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


# ── Hard mode speeds up ──────────────────────────────────────────────────────

def rally(level, hits):
    """Play `hits` perfect hits in a row on `level`; return the game and each flight's travel time."""
    g = Game(level=level, rng=random.Random(0))
    g.start(0.0)
    g.update(SERVE_AT)
    travels, t = [g.travel], SERVE_AT + g.travel           # first ball arrives at t
    for _ in range(hits):
        assert g.swing(t, hand_lane=g.lane) == ["hit"]
        t_back = t + g.travel                                # return trip at the new speed
        g.update(t_back)                                     # bounced off the wall
        travels.append(g.travel)
        t = t_back + g.travel
    return g, travels


def test_hard_gets_faster_with_each_hit():
    _, travels = rally(3, 5)
    assert travels[0] == config.LEVEL_TRAVEL_S[3]
    assert all(b < a for a, b in zip(travels, travels[1:]))
    assert travels[1] == pytest.approx(config.LEVEL_TRAVEL_S[3] * config.LEVEL_SPEEDUP[3])


def test_hard_speed_has_a_floor():
    _, travels = rally(3, 40)
    assert min(travels) == pytest.approx(config.MIN_TRAVEL_S)


def test_a_miss_resets_the_speed():
    g, travels = rally(3, 6)
    assert g.travel < config.LEVEL_TRAVEL_S[3]
    arrive = g._t0 + g.travel
    g.update(arrive + config.HIT_LATE_S + 0.01)              # let it go by
    g.update(arrive + config.HIT_LATE_S + 0.01 + config.MISS_PAUSE_S)   # re-serve
    g.update(arrive + 5.0)                                   # ball leaves the wall
    assert g.state == INCOMING and g.travel == config.LEVEL_TRAVEL_S[3]
    assert g.speed == 1.0


@pytest.mark.parametrize("level", [1, 2])
def test_easy_and_medium_keep_a_steady_speed(level):
    _, travels = rally(level, 8)
    assert set(travels) == {config.LEVEL_TRAVEL_S[level]}


def test_speed_readout():
    g, _ = rally(3, 4)
    assert g.speed == pytest.approx(1 / config.LEVEL_SPEEDUP[3] ** 4)


# ── Incoming sidespin ────────────────────────────────────────────────────────

def serve_many(level, n, seed=0):
    """Start n balls on `level`; return (from_lane, lane) for each."""
    g = Game(level=level, rng=random.Random(seed))
    out = []
    for _ in range(n):
        g.start(0.0)
        g.update(SERVE_AT)
        out.append((g.from_lane, g.lane))
    return out


def test_easy_never_spins():
    assert all(a == b for a, b in serve_many(1, 200))


@pytest.mark.parametrize("level", [2, 3])
def test_spin_rate_matches_the_level(level):
    balls = serve_many(level, 2000)
    rate = sum(a != b for a, b in balls) / len(balls)
    assert rate == pytest.approx(config.LEVEL_SPIN_CHANCE[level], abs=0.03)


def test_spin_only_ever_curves_one_lane():
    assert all(abs(a - b) <= 1 for a, b in serve_many(3, 2000))


def test_aim_starts_on_the_fake_lane_and_ends_on_the_real_one():
    g = started()
    g.lane, g.from_lane = 2, 1
    assert g.spin == +1
    assert g.aim_lane(0.0) == 1
    assert g.aim_lane(1.0) == 2
    assert 1 < g.aim_lane(0.5) < 1.5          # most of the curve comes late
    assert g.aim_lane(1.3) == 2               # a missed ball keeps its line


def test_hit_is_judged_on_the_lane_the_ball_arrives_in():
    g = started()
    g.lane, g.from_lane = 0, 1                 # looks like centre, curves left
    assert g.spin == -1
    assert g.swing(ARRIVE, hand_lane=1) == ["whiff_lane"]   # stayed where it looked headed
    assert g.swing(ARRIVE + 0.15, hand_lane=0) == ["hit"]   # followed the curve


def test_no_spin_aims_straight():
    g = started()
    g.from_lane = g.lane
    assert g.spin == 0
    assert g.aim_lane(0.3) == g.aim_lane(0.9) == g.lane
