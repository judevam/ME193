"""Shared modules: config, topics, schemas, fake bus."""
from common import schemas, topics
from common.bus import FakeBus
from common.config import load_config


def test_config_loads_and_force_mock(monkeypatch):
    cfg = load_config()
    assert cfg["robot_id"] == "bot1"
    monkeypatch.setenv("SUMO_FORCE_MOCK", "1")
    assert set(load_config()["devices"].values()) == {"mock"}


def test_topic_qos_and_retain():
    assert topics.qos_for(topics.CMD_FLIPPER) == 1
    assert topics.qos_for(topics.CMD_MOTORS) == 0
    assert topics.retain_for(topics.STATE)
    assert topics.retain_for(topics.status_topic("brain"))
    assert not topics.retain_for(topics.CMD_MOTORS)


def test_motor_payload_is_clamped():
    p = schemas.motors(3.0, -7.0, None)
    assert (p["left"], p["right"]) == (1.0, -1.0)
    assert schemas.parse_motors({"left": 5, "right": -0.5})[:2] == (1.0, -0.5)
    assert schemas.parse_motors({"left": 5, "right": float("nan")}) is None


def test_parsers_reject_garbage():
    assert schemas.parse_ultrasonic({"cm": "far"}) is None
    assert schemas.parse_joystick("hello") is None
    assert schemas.parse_mode("turbo") is None
    assert schemas.parse_flipper({"action": "explode"}) is None
    assert schemas.parse_color({"name": "blue", "t": 1.0}) == ("blue", 1.0)


def test_fake_bus_delivers_retained_to_late_subscribers():
    bus = FakeBus()
    bus.publish(topics.TELEOP_FLIPPER_MODE, "manual")
    got = []
    bus.subscribe(topics.TELEOP_FLIPPER_MODE, lambda t, p: got.append(p))
    assert got == ["manual"]
