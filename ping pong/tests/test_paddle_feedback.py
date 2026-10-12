"""PaddleFeedback with a fake motor: right commands, batched, light goes back to idle."""

import time
from contextlib import contextmanager

import legoeducation as le

from paddle_feedback import PaddleFeedback

HAPTIC = {"hit": (90, 100, "green"), "miss": (350, 45, "red"), "whiff": (0, 0, "orange")}


class FakeMotor:
    def __init__(self):
        self.calls = []

    @contextmanager
    def batch(self, blocking=True):
        self.calls.append(("batch",))
        yield
        self.calls.append(("end_batch",))

    def motor_run_for_time(self, ms, *, direction, motor, speed, blocking):
        assert blocking is False
        self.calls.append(("buzz", ms, motor, speed, direction))

    def light_color(self, color, *, blocking):
        assert blocking is False
        self.calls.append(("light", color))


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


CW, CCW = le.MOTOR_MOVE_DIRECTION_CLOCKWISE, le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE


def feedback(motor, flash_s=0.05, side="left"):
    return PaddleFeedback(motor, haptic=HAPTIC, side=side, flash_s=flash_s, idle_color="blue")


def test_hit_buzzes_and_flashes_green_in_one_batch():
    m = FakeMotor()
    fb = feedback(m)
    try:
        fb.send("hit")
        assert wait_for(lambda: ("end_batch",) in m.calls)
        assert m.calls[:4] == [("batch",), ("buzz", 90, le.MOTOR_LEFT, 100, CW),
                               ("light", le.LEGO_COLOR_GREEN), ("end_batch",)]
    finally:
        fb.stop()


def test_opposite_spins_both_sides_opposite_ways_in_the_same_batch():
    m = FakeMotor()
    fb = feedback(m, side="opposite")
    try:
        fb.send("hit")
        assert wait_for(lambda: ("end_batch",) in m.calls)
        assert m.calls[:5] == [("batch",),
                               ("buzz", 90, le.MOTOR_LEFT, 100, CW),
                               ("buzz", 90, le.MOTOR_RIGHT, 100, CCW),
                               ("light", le.LEGO_COLOR_GREEN), ("end_batch",)]
    finally:
        fb.stop()


def test_whiff_is_light_only():
    m = FakeMotor()
    fb = feedback(m)
    try:
        fb.send("whiff")
        assert wait_for(lambda: ("end_batch",) in m.calls)
        assert not any(c[0] == "buzz" for c in m.calls)
        assert ("light", le.LEGO_COLOR_ORANGE) in m.calls
    finally:
        fb.stop()


def test_light_returns_to_idle_after_the_flash():
    m = FakeMotor()
    fb = feedback(m, flash_s=0.05)
    try:
        fb.send("miss")
        assert wait_for(lambda: ("light", le.LEGO_COLOR_BLUE) in m.calls)
        assert m.calls.index(("light", le.LEGO_COLOR_RED)) < m.calls.index(("light", le.LEGO_COLOR_BLUE))
    finally:
        fb.stop()


def test_send_returns_immediately():
    class SlowMotor(FakeMotor):
        def light_color(self, color, *, blocking):
            time.sleep(0.2)              # a slow Bluetooth write
            super().light_color(color, blocking=blocking)

    fb = feedback(SlowMotor())
    try:
        t0 = time.perf_counter()
        fb.send("hit")
        assert time.perf_counter() - t0 < 0.01   # the game loop never waits on BLE
    finally:
        fb.stop()


def test_unknown_events_are_ignored():
    m = FakeMotor()
    fb = feedback(m)
    try:
        fb.send("serve")
        time.sleep(0.1)
        assert m.calls == []
    finally:
        fb.stop()


def test_a_failing_motor_does_not_crash():
    class BrokenMotor(FakeMotor):
        def light_color(self, color, *, blocking):
            raise RuntimeError("BLE dropped")

    fb = feedback(BrokenMotor())
    try:
        fb.send("hit")
        fb.send("hit")
        assert wait_for(lambda: len(fb.send_ms) == 2)   # thread survived the first failure
    finally:
        fb.stop()
