"""Actuator bridges, BLE link handling and teleop clients, with mocks."""
import pytest

from bridges.drive_bridge import DriveBridge
from bridges.flipper_bridge import FlipperBridge
from bridges.link import DeviceLink
from common import schemas, topics
from common.bus import FakeBus
from common.config import load_config
from hardware.mock import MockDrive, MockFlipper, MockKeyboard
from sensors.fake_sensors import FakeSensors
from teleop.keyboard_client import KeyboardClient


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def drive_rig():
    cfg = load_config()
    bus, drive, clock = FakeBus(), MockDrive(), Clock()
    bridge = DriveBridge(cfg, bus, drive, clock=clock)
    bus.publish(topics.status_topic("brain"), "online")
    return bus, drive, clock, bridge


def test_drive_follows_commands_and_rate_limits(drive_rig):
    bus, drive, clock, bridge = drive_rig
    for _ in range(5):                       # same command 5 times in 0.4 s
        bus.publish(topics.CMD_MOTORS, schemas.motors(0.5, 0.5, None))
        bridge.step(clock.t)
        clock.t += 0.1
    assert drive.calls == [(0.5, 0.5)]       # only sent once (no change, no heartbeat yet)
    clock.t = 1.05
    bus.publish(topics.CMD_MOTORS, schemas.motors(0.5, 0.5, None))
    bridge.step(clock.t)
    assert drive.calls == [(0.5, 0.5)] * 2   # heartbeat resend


def test_drive_stops_when_commands_stop(drive_rig):
    bus, drive, clock, bridge = drive_rig
    bus.publish(topics.CMD_MOTORS, schemas.motors(1.0, 1.0, None))
    bridge.step(clock.t)
    clock.t += 0.6                           # > cmd_timeout_s
    bridge.step(clock.t)
    assert drive.calls[-1] == (0.0, 0.0)


def test_drive_stops_when_brain_goes_offline(drive_rig):
    bus, drive, clock, bridge = drive_rig
    bus.publish(topics.CMD_MOTORS, schemas.motors(1.0, 1.0, None))
    bridge.step(clock.t)
    bus.publish(topics.status_topic("brain"), "offline")    # Last Will
    clock.t += 0.1
    bridge.step(clock.t)
    assert drive.calls[-1] == (0.0, 0.0)


def test_drive_values_are_clamped(drive_rig):
    bus, drive, clock, bridge = drive_rig
    bus.publish(topics.CMD_MOTORS, {"left": 9, "right": -9, "t": 0, "src_t": None})
    bridge.step(clock.t)
    assert drive.calls[-1] == (1.0, -1.0)


def test_link_reports_and_recovers_from_disconnect():
    cfg = load_config()
    bus, drive = FakeBus(), MockDrive()
    link = DeviceLink(drive, bus, "drive_bridge", cfg)
    assert link.ensure(0.0)
    assert bus.retained[topics.ble_status_topic("drive_bridge")] == "connected"
    drive.is_connected = False
    with pytest.raises(ConnectionError):
        drive.set(1, 1)
    link.lost("write failed")
    assert bus.messages(topics.ble_status_topic("drive_bridge"))[-1] == "disconnected"
    assert link.ensure(1.0)                  # MockDrive.connect() succeeds
    assert bus.retained[topics.ble_status_topic("drive_bridge")] == "connected"


def test_flipper_bridge_refuses_fire_when_defeated():
    import time
    bus, flipper = FakeBus(), MockFlipper()
    bridge = FlipperBridge(bus, flipper)
    bus.publish(topics.STATE, schemas.state("DEFEATED", "DEFEATED", True, "auto"))
    bridge.handle("fire", time.monotonic())
    bridge.handle("home", time.monotonic())
    assert flipper.calls == ["home"]


def test_flipper_bridge_drops_stale_commands():
    import time
    flipper = MockFlipper()
    FlipperBridge(FakeBus(), flipper).handle("fire", time.monotonic() - 2)
    assert flipper.calls == []


def test_flipper_software_limits():
    from hardware.lego import LegoFlipper
    f = load_config()["flipper"]
    with pytest.raises(ValueError):
        LegoFlipper({}, 50, {**f, "home_deg": 0, "fire_deg": 200, "fire_direction": "clockwise"})
    # 0 -> 300 counterclockwise is only 60 degrees: allowed
    flip = LegoFlipper({}, 50, {**f, "home_deg": 0, "fire_deg": 300, "fire_direction": "counterclockwise"})
    assert flip.travel == 60
    assert flip.arc(0, 330) == 30            # inside the sweep
    assert flip.arc(0, 90) == 270            # outside it


def test_keyboard_client():
    bus = FakeBus()
    client = KeyboardClient(bus, MockKeyboard())
    for key in ["space", "m", "d", "r"]:
        assert client.handle(key)
    assert not client.handle("q")
    assert bus.retained[topics.TELEOP_FLIPPER_MODE] == "manual"
    for t in (topics.TELEOP_FLIPPER_FIRE, topics.TELEOP_DEFEAT, topics.CONTROL_RESET):
        assert len(bus.messages(t)) == 1


def test_fake_sensors():
    bus, keys = FakeBus(), MockKeyboard()
    sim = FakeSensors(bus, keys)
    for key in ["b", "2", "j"]:
        sim.handle(key)
    sim.publish()
    assert bus.messages(topics.SENSOR_COLOR)[-1]["name"] == "blue"
    assert bus.messages(topics.SENSOR_ULTRASONIC)[-1]["cm"] == 10
    assert bus.messages(topics.TELEOP_JOYSTICK) == []        # controller "disconnected"
    sim.handle("j")
    keys.held.add("up")
    sim.publish()
    assert bus.messages(topics.TELEOP_JOYSTICK)[-1]["left"] == 1.0


def test_lego_card_filters_from_config():
    from hardware.lego import lego_filters
    assert lego_filters(load_config()["lego"]["drive"]) == {"card_color": 9, "card_serial": 7572}
    with pytest.raises(ValueError):
        lego_filters({"card_color": "chartreuse"})


def test_drive_swap_and_invert_for_reversed_motor():
    from types import SimpleNamespace
    from hardware.lego import LegoDrive
    sent = []
    drive = LegoDrive({}, 50, {"max_speed_pct": 100, "invert_left": True,
                               "invert_right": True, "swap_sides": True})
    drive.dev = SimpleNamespace(movement_move_tank=lambda l, r: sent.append((l, r)))
    drive.set(0.8, 0.2)                       # robot: left side fast, right side slow
    assert sent == [(-20, -80)]               # motor: outputs swapped and reversed


def test_controller_invert_makes_forward_positive():
    from types import SimpleNamespace
    from hardware.lego import LegoController
    ctl = LegoController({}, 50, {"invert_left": True, "invert_right": True, "swap_sides": False})
    ctl.dev = SimpleNamespace(connected=True, sensor=SimpleNamespace(leftPercent=-80, rightPercent=-40))
    ctl._last_note = __import__("time").monotonic()
    assert ctl.read() == (0.8, 0.4)          # levers pushed forward read negative on our Controller
    ctl.swap = True
    assert ctl.read() == (0.4, 0.8)          # right lever now drives the left wheels
