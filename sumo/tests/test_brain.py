"""Behavior tree branches, driven through a fake MQTT bus and a fake clock."""
from __future__ import annotations

import pytest

from brain.brain import Brain
from common import schemas, topics
from common.bus import FakeBus
from common.config import load_config

DT = 0.05  # 20 Hz


class Rig:
    """A brain on a fake bus. `stream` holds sensor values republished every tick,
    like the real sensors do; set an entry to None to stop that stream."""

    def __init__(self, **overrides) -> None:
        self.cfg = load_config()
        for section, values in overrides.items():
            self.cfg[section].update(values)
        self.t = 100.0
        self.bus = FakeBus()
        self.brain = Brain(self.cfg, self.bus, clock=lambda: self.t)
        self.stream: dict[str, object] = {"color": "white", "cm": None, "joy": None}

    def send(self, topic: str, payload) -> None:
        self.bus.publish(topic, payload)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.t += DT
            s = self.stream
            if s["color"] is not None:
                self.send(topics.SENSOR_COLOR, schemas.color(s["color"], [0, 0, 0]))
            if s["cm"] is not None:
                self.send(topics.SENSOR_ULTRASONIC, schemas.ultrasonic(s["cm"]))
            if s["joy"] is not None:
                self.send(topics.TELEOP_JOYSTICK, schemas.joystick(*s["joy"]))
            self.brain.step()

    def seconds(self, s: float) -> None:
        self.tick(round(s / DT))

    @property
    def motors(self) -> tuple[float, float]:
        m = self.bus.messages(topics.CMD_MOTORS)[-1]
        return m["left"], m["right"]

    @property
    def node(self) -> str:
        return self.bus.retained[topics.STATE]["node"]

    @property
    def flipper_node(self) -> str:
        return self.bus.retained[topics.STATE]["flipper_node"]

    def fires(self) -> int:
        return sum(1 for p in self.bus.messages(topics.CMD_FLIPPER) if p["action"] == "fire")

    def songs(self) -> int:
        return len(self.bus.messages(topics.CMD_SOUND))


@pytest.fixture
def rig() -> Rig:
    return Rig()


# ---- DRIVE ------------------------------------------------------------------

def test_teleop_tank_drive(rig):
    rig.stream["joy"] = (0.8, -0.5)
    rig.tick()
    assert rig.node == "TELEOP_DRIVE"
    assert rig.motors == (0.8, -0.5)


def test_blue_tape_stops_backing_up(rig):
    rig.stream["joy"] = (-1.0, -1.0)        # operator reverses toward the edge
    rig.tick()
    assert rig.motors == (-1.0, -1.0)
    rig.stream["color"] = "blue"             # rear sensor hits the tape
    rig.tick()
    assert rig.node == "AVOID_EDGE"
    assert rig.motors == (0.0, 0.0)          # stopped despite full-reverse levers


def test_blue_tape_allows_forward_only(rig):
    rig.stream["color"] = "blue"
    rig.stream["joy"] = (1.0, 1.0)
    rig.tick()
    assert rig.node == "AVOID_EDGE"
    assert rig.motors == (1.0, 1.0)          # straight forward: allowed
    rig.stream["joy"] = (0.8, -0.8)          # spin: the backward wheel is held at 0
    rig.tick()
    assert rig.motors == (0.8, 0.0)
    rig.stream["joy"] = None                 # no levers: stop
    rig.seconds(0.4)
    assert rig.motors == (0.0, 0.0)


def test_reverse_unlocks_after_leaving_the_tape(rig):
    rig.stream["color"] = "blue"
    rig.stream["joy"] = (-1.0, -1.0)
    rig.tick()
    rig.stream["color"] = "white"            # drove forward off the tape
    rig.seconds(0.2)                          # < clear_s: still forward-only
    assert rig.node == "AVOID_EDGE"
    assert rig.motors == (0.0, 0.0)
    rig.seconds(0.2)                          # > clear_s
    assert rig.node == "TELEOP_DRIVE"
    assert rig.motors == (-1.0, -1.0)


def test_stale_joystick_falls_to_idle(rig):
    rig.stream["joy"] = (1.0, 1.0)
    rig.tick()
    assert rig.motors == (1.0, 1.0)
    rig.stream["joy"] = None                 # controller stops reporting
    rig.seconds(0.4)                         # > teleop_stale_s
    assert rig.node == "IDLE"
    assert rig.motors == (0.0, 0.0)


def test_levers_at_rest_let_hold_ground_push(rig):
    rig.stream["joy"] = (0.02, -0.03)        # inside the deadzone
    rig.stream["cm"] = 14
    rig.tick(3)
    assert rig.node == "HOLD_GROUND"
    assert rig.motors == (0.4, 0.4)


def test_no_driving_without_fresh_color(rig):
    rig.stream["joy"] = (1.0, 1.0)
    rig.stream["color"] = None               # color sensor disconnected
    rig.seconds(0.5)
    assert rig.node == "IDLE"
    assert rig.motors == (0.0, 0.0)


# ---- ultrasonic filtering ----------------------------------------------------

def test_ultrasonic_median_and_max_range(rig):
    for cm in (300, 30, 250, 60, 31):        # 300 and 250 are discarded
        rig.send(topics.SENSOR_ULTRASONIC, schemas.ultrasonic(cm))
    rig.brain.step()
    assert rig.brain.world.distance(rig.t) == 31   # median of 30, 60, 31


# ---- FLIPPER ----------------------------------------------------------------

def test_auto_fire_respects_cooldown_and_blanking(rig):
    rig.stream["cm"] = 10
    rig.tick(3)
    assert rig.fires() == 1
    # Readings during the blanking window are ignored, so no distance at all.
    rig.seconds(0.3)
    assert rig.brain.world.distance(rig.t) is None
    # Opponent still there, but the flipper is cooling down.
    rig.seconds(1.5)
    assert rig.fires() == 1
    rig.seconds(0.5)                          # > 2 s since the first fire
    assert rig.fires() == 2


def test_manual_mode_blocks_auto_fire(rig):
    rig.send(topics.TELEOP_FLIPPER_MODE, "manual")
    rig.stream["cm"] = 10
    rig.seconds(1.0)
    assert rig.fires() == 0
    assert rig.flipper_node == "MANUAL_MODE"
    rig.send(topics.TELEOP_FLIPPER_FIRE, schemas.event())   # space bar still works
    rig.tick()
    assert rig.fires() == 1


def test_manual_fire_respects_cooldown(rig):
    rig.send(topics.TELEOP_FLIPPER_FIRE, schemas.event())
    rig.tick()
    rig.send(topics.TELEOP_FLIPPER_FIRE, schemas.event())
    rig.tick()
    assert rig.fires() == 1


def test_no_auto_fire_while_avoiding_edge(rig):
    rig.stream["color"] = "blue"
    rig.stream["cm"] = 10
    rig.tick(3)
    assert rig.node == "AVOID_EDGE"
    assert rig.fires() == 0


def test_flipper_homes_after_stroke(rig):
    rig.tick()
    homes_at_start = rig.bus.messages(topics.CMD_FLIPPER)
    assert homes_at_start == [{"action": "home", "t": homes_at_start[0]["t"]}]
    rig.send(topics.TELEOP_FLIPPER_FIRE, schemas.event())
    rig.tick()
    rig.seconds(0.5)                          # mid-stroke: no home yet
    assert [p["action"] for p in rig.bus.messages(topics.CMD_FLIPPER)] == ["home", "fire"]
    rig.seconds(0.5)
    assert [p["action"] for p in rig.bus.messages(topics.CMD_FLIPPER)] == ["home", "fire", "home"]


# ---- DEFEATED ---------------------------------------------------------------

def defeat(rig: Rig) -> None:
    rig.send(topics.TELEOP_DEFEAT, schemas.event())
    rig.tick()


def test_leaving_the_ring_does_not_defeat(rig):
    rig.stream["color"] = "blue"
    rig.tick()
    rig.stream["color"] = "none"              # well outside the ring
    rig.seconds(3.0)
    assert rig.node != "DEFEATED"
    assert rig.songs() == 0


def test_defeat_key_latches_plays_song_once_and_resets(rig):
    rig.stream["joy"] = (1.0, 1.0)
    defeat(rig)
    assert rig.node == "DEFEATED"
    assert rig.motors == (0.0, 0.0)

    rig.stream["cm"] = 10                     # opponent in range, fire key pressed
    rig.send(topics.TELEOP_FLIPPER_FIRE, schemas.event())
    rig.seconds(3.0)
    assert rig.node == "DEFEATED"             # latched
    assert rig.songs() == 1
    assert rig.fires() == 0                   # flipper never fires while defeated
    assert rig.motors == (0.0, 0.0)

    rig.send(topics.CONTROL_RESET, schemas.event())
    rig.stream["cm"] = None
    rig.tick()
    assert rig.node == "TELEOP_DRIVE"
    assert rig.bus.retained[topics.STATE]["defeated"] is False

    defeat(rig)                               # can be defeated again, song plays again
    assert rig.songs() == 2


def test_state_published_only_on_change(rig):
    rig.tick(10)
    assert len(rig.bus.messages(topics.STATE)) == 1
    rig.stream["joy"] = (0.5, 0.5)
    rig.tick(10)
    assert len(rig.bus.messages(topics.STATE)) == 2
