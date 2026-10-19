"""Pick mock or hardware implementations from config["devices"]."""
from __future__ import annotations

from typing import Any

from hardware import mock
from hardware.interfaces import (ColorSensor, DriveBase, FlipperMotor, JoystickInput,
                                 SoundOutput)


def _hw(cfg: dict[str, Any], name: str) -> bool:
    mode = cfg["devices"][name]
    if mode not in ("mock", "hardware"):
        raise ValueError(f"devices.{name} must be mock or hardware, not {mode!r}")
    return mode == "hardware"


def make_drive(cfg: dict[str, Any]) -> DriveBase:
    """Double Motor."""
    if _hw(cfg, "drive"):
        from hardware.lego import LegoDrive
        return LegoDrive(cfg["lego"]["drive"], cfg["lego"]["notify_ms"], cfg["drive_bridge"])
    return mock.MockDrive()


def make_flipper(cfg: dict[str, Any]) -> FlipperMotor:
    if _hw(cfg, "flipper"):
        from hardware.lego import LegoFlipper
        return LegoFlipper(cfg["lego"]["flipper"], cfg["lego"]["notify_ms"], cfg["flipper"])
    return mock.MockFlipper(stroke_s=cfg["flipper"]["sweep_s"])


def make_color(cfg: dict[str, Any]) -> ColorSensor:
    if _hw(cfg, "color"):
        from hardware.lego import LegoColor
        return LegoColor(cfg["lego"]["color"], cfg["lego"]["notify_ms"])
    return mock.MockColor()


def make_joystick(cfg: dict[str, Any]) -> JoystickInput:
    if _hw(cfg, "joystick"):
        from hardware.lego import LegoController
        return LegoController(cfg["lego"]["joystick"], cfg["lego"]["notify_ms"], cfg["joystick"])
    return mock.MockJoystick()


def make_sound(cfg: dict[str, Any]) -> SoundOutput:
    if _hw(cfg, "sound"):
        from hardware.sound import LaptopSpeaker
        return LaptopSpeaker()
    return mock.MockSound()
