"""One start tag: the column it's held in picks the level."""

import pytest

import config
from level_tags import LevelPicker, Tag, TagReader, level_at, make_tag, mirrored_center_x

LEFT_X, MID_X, RIGHT_X = 0.15, 0.5, 0.85   # mirrored-screen x in each column


def picker():
    return LevelPicker(hold=0.5)


@pytest.mark.parametrize("x, level", [(LEFT_X, 1), (MID_X, 2), (RIGHT_X, 3)])
def test_column_picks_level(x, level):
    assert level_at(x) == level


def test_mirroring():
    # a tag on the camera's LEFT edge shows up on the RIGHT of the mirrored screen
    tag = Tag(0, [(0, 0), (64, 0), (64, 64), (0, 64)])
    assert mirrored_center_x(tag, 640) == pytest.approx(0.95)


def test_no_tag_keeps_waiting():
    p = picker()
    assert all(p.update([], i * 0.033) is None for i in range(50))


@pytest.mark.parametrize("x, level", [(LEFT_X, 1), (MID_X, 2), (RIGHT_X, 3)])
def test_held_in_a_column_starts_that_level_once(x, level):
    p = picker()
    picked = [p.update([x], i * 0.033) for i in range(40)]   # ~1.3 s
    assert [v for v in picked if v] == [level]


def test_must_be_held_long_enough():
    p = picker()
    assert p.update([MID_X], 0.0) is None
    assert p.update([MID_X], 0.49) is None
    assert p.update([MID_X], 0.50) == 2


def test_moving_to_another_column_restarts_the_hold():
    # sweeping the tag from left to right shouldn't start "easy" on the way past
    p = picker()
    p.update([LEFT_X], 0.0)
    p.update([LEFT_X], 0.3)
    assert p.update([RIGHT_X], 0.4) is None
    assert p.update([RIGHT_X], 0.8) is None
    assert p.update([RIGHT_X], 0.9) == 3


def test_moving_within_a_column_keeps_the_hold():
    p = picker()
    p.update([0.05], 0.0)
    assert p.update([0.30], 0.5) == 1


def test_tag_out_of_view_restarts_the_hold():
    p = picker()
    p.update([MID_X], 0.0)
    p.update([], 0.3)
    assert p.update([MID_X], 0.6) is None
    assert p.update([MID_X], 1.1) == 2


def test_progress_fills_up():
    p = picker()
    p.update([MID_X], 0.0)
    assert p.progress(0.25) == pytest.approx(0.5)


def test_generated_tag_is_detected(tmp_path):
    import cv2

    img = cv2.imread(str(make_tag(tmp_path)))
    assert [t.id for t in TagReader().detect(img)] == [config.START_TAG_ID]
