"""3D table view: camera maths, the ball's path, paddle placement, and drawing every state."""

import numpy as np
import pytest

import config
from game import Game
from pose_tracker import Wrist, lane_of
from table_view import (BALL_R, BOUNCE_IN_Z, BOUNCE_OUT_Z, EYE, HIT_Y, HIT_Z, LANE_X, LOOK_AT,
                        PADDLE_MAX_TILT, SWING_ANIM_S, TABLE_Y, WALL_HIT_Y, WALL_Z, Camera, TableView,
                        ball_world, paddle_pose, paddle_x)

W, H = 960, 720
TRAVEL = config.LEVEL_TRAVEL_S[1]
SERVE_AT = config.SERVE_DELAY_S
ARRIVE = SERVE_AT + TRAVEL


def incoming(lane=1):
    g = Game(level=1)
    g.start(0.0)
    g.update(SERVE_AT)
    g.lane = g.from_lane = lane       # no spin unless a test asks for it
    return g


def at(g, s):
    """Game time when the incoming ball is fraction s of the way from the wall to you."""
    return SERVE_AT + s * TRAVEL


# ── Camera ───────────────────────────────────────────────────────────────────

def test_look_at_point_projects_to_the_centre():
    cam = Camera(EYE, LOOK_AT, W, H)
    u, v = cam.project(LOOK_AT)
    assert u == pytest.approx(W / 2) and v == pytest.approx(H / 2)


def test_left_is_left_and_up_is_up():
    cam = Camera(EYE, LOOK_AT, W, H)
    left, right = cam.project((-0.5, TABLE_Y, 1.0)), cam.project((0.5, TABLE_Y, 1.0))
    low, high = cam.project((0, TABLE_Y, 1.0)), cam.project((0, TABLE_Y + 0.5, 1.0))
    assert left[0] < W / 2 < right[0]
    assert high[1] < low[1]


def test_far_things_look_smaller():
    cam = Camera(EYE, LOOK_AT, W, H)
    assert cam.radius((0, TABLE_Y, 2.5), BALL_R) < cam.radius((0, TABLE_Y, 0.2), BALL_R)


# ── The ball's path ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("lane", [0, 1, 2])
def test_incoming_ball_leaves_the_wall_and_arrives_at_the_hit_point(lane):
    g = incoming(lane)
    x, y, z = ball_world(g, at(g, 0.0))
    assert (x, y, z) == pytest.approx((0.0, WALL_HIT_Y, WALL_Z))
    x, y, z = ball_world(g, ARRIVE)
    assert (x, y, z) == pytest.approx((LANE_X[lane], HIT_Y, HIT_Z))


def test_incoming_ball_bounces_on_your_half():
    g = incoming()
    s_bounce = (WALL_Z - BOUNCE_IN_Z) / (WALL_Z - HIT_Z)
    _, y, z = ball_world(g, at(g, s_bounce))
    assert z == pytest.approx(BOUNCE_IN_Z)
    assert y == pytest.approx(TABLE_Y + BALL_R)          # touching the table
    _, y_before, _ = ball_world(g, at(g, s_bounce - 0.1))
    _, y_after, _ = ball_world(g, at(g, s_bounce + 0.1))
    assert y_before > TABLE_Y + BALL_R and y_after > TABLE_Y + BALL_R   # above it either side


def test_ball_never_goes_through_the_table():
    g = incoming()
    for s in np.linspace(0, 1, 101):
        x, y, z = ball_world(g, at(g, s))
        if 0 <= z <= 2.74:
            assert y >= TABLE_Y + BALL_R - 1e-9


def test_returned_ball_bounces_on_the_far_half_and_reaches_the_wall():
    g = incoming(lane=2)
    g.swing(ARRIVE, hand_lane=2)
    assert ball_world(g, ARRIVE) == pytest.approx((LANE_X[2], HIT_Y, HIT_Z))
    q_bounce = (BOUNCE_OUT_Z - HIT_Z) / (WALL_Z - HIT_Z)
    _, y, z = ball_world(g, ARRIVE + q_bounce * TRAVEL)
    assert z == pytest.approx(BOUNCE_OUT_Z) and y == pytest.approx(TABLE_Y + BALL_R)
    assert ball_world(g, ARRIVE + TRAVEL * 0.999) == pytest.approx((0, WALL_HIT_Y, WALL_Z), abs=0.02)


def test_missed_ball_carries_on_past_and_drops():
    g = incoming()
    g.update(ARRIVE + config.HIT_LATE_S + 0.01)          # missed
    _, y, z = ball_world(g, ARRIVE + 0.6)
    assert z < HIT_Z and y < HIT_Y


# ── Paddle ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lane", [0, 1, 2])
def test_wrist_in_the_middle_of_a_lane_puts_the_paddle_on_that_lane(lane):
    edges = (0.0, *config.LANE_EDGES, 1.0)
    wrist_x = (edges[lane] + edges[lane + 1]) / 2
    assert lane_of(wrist_x) == lane
    assert paddle_x(wrist_x) == pytest.approx(LANE_X[lane])


# ── Drawing ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def view():
    return TableView(W, H)


@pytest.mark.parametrize("t", [0.0, 1.5, 2.3, 2.6, 2.9, 3.5, 5.0])   # serve, incoming, window, missed...
def test_render_every_state(view, t):
    g = Game(level=1)
    g.start(0.0)
    for step in range(int(t * 100) + 1):
        g.update(step / 100)
    img = view.render(g, t, wrist=Wrist(t, 0.8, 0.6), wrist_lane=2, cam_frame=np.zeros((H, W, 3), np.uint8),
                      message=("HIT!", (0, 255, 0)), swing_flash=True, burst=(0.1, 2), fps="60 fps")
    assert img.shape == (H, W, 3)


def test_render_without_camera_or_wrist(view):
    g = incoming()
    img = view.render(g, at(g, 0.5))
    assert img.shape == (H, W, 3)


def test_render_does_not_change_the_cached_background(view):
    before = view.static.copy()
    g = incoming()
    view.render(g, at(g, 0.9), wrist=Wrist(0, 0.5, 0.5), wrist_lane=1, burst=(0.05, 1))
    assert np.array_equal(before, view.static)


# ── Paddle tilt and swing ────────────────────────────────────────────────────


def test_paddle_upright_in_the_middle():
    (_, y, z), roll, width = paddle_pose(0.0)
    assert roll == 0 and width == 1.0
    assert y == HIT_Y


def test_paddle_tilts_out_symmetrically_at_the_sides():
    _, roll_right, width_right = paddle_pose(LANE_X[2])
    _, roll_left, width_left = paddle_pose(LANE_X[0])
    assert roll_right == pytest.approx(PADDLE_MAX_TILT)       # top leans right, handle back to you
    assert roll_left == pytest.approx(-PADDLE_MAX_TILT)
    assert width_right == width_left < 1.0                    # face turns a bit edge-on


def test_tilt_is_smooth_between_lanes():
    rolls = [paddle_pose(x)[1] for x in np.linspace(LANE_X[0], LANE_X[2], 11)]
    assert rolls == sorted(rolls)


def test_swing_lunges_forward_and_comes_back():
    rest, _, _ = paddle_pose(LANE_X[2])
    mid, _, _ = paddle_pose(LANE_X[2], swing_age=SWING_ANIM_S / 2)
    end, _, _ = paddle_pose(LANE_X[2], swing_age=SWING_ANIM_S)     # animation over
    assert mid[2] > rest[2] + 0.2                                   # pushed toward the table
    assert mid[0] < rest[0]                                         # swept toward the middle
    assert end == rest


def test_backhand_sweeps_the_other_way():
    rest, roll_rest, _ = paddle_pose(LANE_X[0])
    mid, roll_mid, _ = paddle_pose(LANE_X[0], swing_age=SWING_ANIM_S / 2)
    assert mid[0] > rest[0]                                         # left side sweeps right
    assert roll_mid > roll_rest


def test_render_mid_swing(view):
    g = incoming(lane=2)
    img = view.render(g, at(g, 0.9), wrist=Wrist(0, 0.83, 0.6), wrist_lane=2, swing_age=0.1)
    assert img.shape == (H, W, 3)


# ── Hard mode: no helpers ────────────────────────────────────────────────────

class SpyView(TableView):
    """Records which helper drawings happen."""

    def __init__(self, w, h):
        super().__init__(w, h)
        self.drawn = []

    def _draw_lane(self, img, lane, in_lane):
        self.drawn.append("lane")

    def _draw_target(self, img, game, now):
        self.drawn.append("target")

    def _draw_paddle(self, img, x, highlight, swing_age=None):
        self.drawn.append("paddle glow" if highlight else "paddle")


def test_only_hard_turns_the_helpers_off():
    assert config.LEVEL_AIDS == {1: True, 2: True, 3: False}


@pytest.mark.parametrize("aids, expected", [
    (True, ["lane", "target", "paddle glow"]),
    (False, ["paddle"]),                      # hard: just the paddle, no strip, rings or glow
])
def test_helpers_follow_the_aids_flag(aids, expected):
    v = SpyView(W, H)
    g = incoming(lane=2)
    v.render(g, ARRIVE - 0.1, wrist=Wrist(0, 0.83, 0.6), wrist_lane=2, aids=aids)
    assert v.drawn == expected


def test_levels_are_faster_than_before_and_hard_is_still_hittable():
    travel = config.LEVEL_TRAVEL_S
    assert travel[1] < 1.6 and travel[2] < 1.2 and travel[3] < 0.9
    # on hard, the hit window still opens after whiffs start counting, and a whiff leaves
    # time (the swing detector's cooldown) to swing again before the window closes
    window_opens = travel[3] - config.HIT_EARLY_S
    assert window_opens > config.WHIFF_FROM_Z * travel[3]
    assert config.HIT_EARLY_S + config.HIT_LATE_S > config.SWING_COOLDOWN_S


# ── Sidespin in 3D ───────────────────────────────────────────────────────────

def test_spinning_ball_heads_for_the_fake_lane_then_curves_in():
    g = incoming(lane=2)
    g.from_lane = 1                                       # looks centre, curves right
    x_early, _, _ = ball_world(g, at(g, 0.3))
    x_end, _, _ = ball_world(g, ARRIVE)
    assert abs(x_early) < 0.3 * LANE_X[2] * 0.5          # still near the centre line early on
    assert x_end == pytest.approx(LANE_X[2])              # arrives in its real lane


def test_render_spinning_ball(view):
    g = incoming(lane=0)
    g.from_lane = 1
    for t in (at(g, 0.4), at(g, 0.8), ARRIVE):
        assert view.render(g, t, wrist=Wrist(0, 0.17, 0.6), wrist_lane=0, aids=False).shape == (H, W, 3)
