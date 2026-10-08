"""The drawing code runs in every game state without a camera or window."""

import pytest

import config
from game import Game
from main import blank_background, draw_scene, ball_screen, MESSAGES, WIDTH, HEIGHT


def test_ball_grows_and_moves_down_as_it_comes_closer():
    (_, y_far), r_far = ball_screen(0.0, WIDTH, HEIGHT)
    (_, y_mid), r_mid = ball_screen(0.5, WIDTH, HEIGHT)
    (_, y_near), r_near = ball_screen(1.0, WIDTH, HEIGHT)
    assert y_far < y_mid < y_near
    assert r_far < r_mid < r_near


@pytest.mark.parametrize("t", [0.0, 1.5, 2.6, 3.0, 4.0, 5.0])   # serve, incoming, window, missed...
def test_draw_scene_every_state(t):
    g = Game(level=1)
    g.start(0.0)
    for step in range(int(t * 100) + 1):
        g.update(step / 100)
    img = blank_background()
    out = draw_scene(img, g, t, MESSAGES["hit"], swing_flash=True)
    assert out.shape == (HEIGHT, WIDTH, 3)
    assert out.any()


def test_ball_heads_to_its_lane():
    (x_left, _), _ = ball_screen(1.0, WIDTH, HEIGHT, lane=0)
    (x_mid, _), _ = ball_screen(1.0, WIDTH, HEIGHT, lane=1)
    (x_right, _), _ = ball_screen(1.0, WIDTH, HEIGHT, lane=2)
    (x_wall, _), _ = ball_screen(0.0, WIDTH, HEIGHT, lane=0)
    assert x_left < x_mid < x_right
    assert x_wall == WIDTH // 2          # every ball leaves the middle of the wall


def test_draw_scene_with_wrist():
    from pose_tracker import Wrist
    g = Game(level=1)
    g.start(0.0)
    g.update(1.0)
    out = draw_scene(blank_background(), g, 2.0, wrist=Wrist(2.0, 0.8, 0.6))
    assert out.shape == (HEIGHT, WIDTH, 3)


def test_start_screen_draws_tags_and_hold_ring():
    from main import draw_start_screen
    square = [(300, 300), (420, 300), (420, 420), (300, 420)]
    out = draw_start_screen(blank_background(), best=4, tags=[square], held_level=2, progress=0.6)
    assert out.shape == (HEIGHT, WIDTH, 3)
