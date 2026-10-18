"""Real LEGO Education BLE devices, via the `legoeducation` package.

API used (all from legoeducation 1.1.1):
  DoubleMotor: connect(...), movement_move_tank(l%, r%), movement_stop()
  SingleMotor: connect(...), motor_run_to_absolute_position(deg, direction=, speed=),
               motor_stop(), motor.absolutePosition
  ColorSensor: connect(...), sensor.color / rawRed / rawGreen / rawBlue
  Controller:  connect(...), sensor.leftPercent / rightPercent
  all:         set_notification_callback(fn), connected, disconnect()

A device that is disconnected keeps its last sensor values in memory, so every
read() here also checks that a notification arrived recently.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import legoeducation as le
from legoeducation.basic_device import VersionMismatchError  # noqa: F401  (re-exported)

from common.schemas import clamp
from hardware.interfaces import (ColorReading, ColorSensor, Device, DriveBase,
                                 FlipperMotor, JoystickInput)

log = logging.getLogger("lego")

COLOR_NAMES = {code: name.removeprefix("LEGO_COLOR_").lower()
               for code, name in le.LEGO_COLOR_NAME_MAP.items()}
COLOR_NAMES[le.LEGO_COLOR_NOCOLOR] = "none"

DIRECTIONS = {
    "clockwise": le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
    "counterclockwise": le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
}


def lego_filters(raw: dict[str, Any]) -> dict[str, Any]:
    """Config filters -> search()/connect() kwargs. card_color may be a name
    ("orange"); the library needs the number and silently ignores anything else."""
    filters = {k: v for k, v in raw.items() if v is not None}
    color = filters.get("card_color")
    if isinstance(color, str):
        codes = {name: code for code, name in COLOR_NAMES.items()}
        if color.lower() not in codes:
            raise ValueError(f"unknown card_color {color!r}; use one of {sorted(codes)}")
        filters["card_color"] = codes[color.lower()]
    if "card_serial" in filters:
        filters["card_serial"] = int(filters["card_serial"])
    return filters


def _bad(*values: Any) -> bool:
    return any(isinstance(v, float) and math.isnan(v) for v in values)


class _LegoDevice(Device):
    """Connection handling shared by all four devices."""

    device_class: type = None

    def __init__(self, filters: dict[str, Any], notify_ms: int) -> None:
        self.filters = lego_filters(filters)
        self.notify_ms = notify_ms
        self.stale_s = max(0.3, 5 * notify_ms / 1000)
        self.dev = None
        self._last_note = 0.0

    def _on_notification(self, _data) -> None:
        self._last_note = time.monotonic()

    def connect(self) -> None:
        """Scan and connect. Raises ConnectionError, or VersionMismatchError if
        the firmware and the Python package disagree."""
        self.close()
        self.dev = self.device_class()
        self.dev.set_notification_callback(self._on_notification)
        log.info("connecting %s %s", self.device_class.__name__, self.filters or "(first found)")
        self.dev.connect(**self.filters, device_notification_delay=self.notify_ms)
        if not self.dev.connected:
            raise ConnectionError(f"{self.device_class.__name__} not found / connect failed")
        self._last_note = time.monotonic()
        log.info("connected %s", self.device_class.__name__)

    @property
    def connected(self) -> bool:
        return self.dev is not None and self.dev.connected

    def fresh(self) -> bool:
        return self.connected and time.monotonic() - self._last_note <= self.stale_s

    def close(self) -> None:
        if self.dev is not None:
            try:
                self.dev.disconnect()
            except Exception:
                pass
            self.dev = None


class LegoDrive(_LegoDevice, DriveBase):
    """Double Motor: wheels (tank drive)."""

    device_class = le.DoubleMotor

    def __init__(self, filters: dict[str, Any], notify_ms: int, bridge_cfg: dict[str, Any]) -> None:
        super().__init__(filters, notify_ms)
        self.max_pct = bridge_cfg["max_speed_pct"]
        self.invert_left = bridge_cfg["invert_left"]
        self.invert_right = bridge_cfg["invert_right"]
        self.swap = bridge_cfg["swap_sides"]

    def set(self, left: float, right: float) -> None:
        """left/right are the robot's sides, forward = positive."""
        left, right = clamp(left), clamp(right)
        if self.swap:                       # motor's left output is on the robot's right
            left, right = right, left
        if self.invert_left:
            left = -left
        if self.invert_right:
            right = -right
        self.dev.movement_move_tank(round(left * self.max_pct), round(right * self.max_pct))

    def stop(self) -> None:
        self.dev.movement_stop()


class LegoFlipper(_LegoDevice, FlipperMotor):
    """Single Motor driving the flipper arm by absolute encoder angle.

    Software limit: the arm only ever moves inside the arc from home_deg to
    fire_deg (in fire_direction), and that arc must be <= max_travel_deg."""

    device_class = le.SingleMotor

    def __init__(self, filters: dict[str, Any], notify_ms: int, fcfg: dict[str, Any]) -> None:
        super().__init__(filters, notify_ms)
        self.home_deg = fcfg["home_deg"] % 360
        self.fire_deg = fcfg["fire_deg"] % 360
        self.speed = fcfg["speed_pct"]
        if fcfg["fire_direction"] not in DIRECTIONS:
            raise ValueError("flipper.fire_direction must be clockwise or counterclockwise")
        self.clockwise = fcfg["fire_direction"] == "clockwise"
        self.out_dir = DIRECTIONS[fcfg["fire_direction"]]
        self.back_dir = DIRECTIONS["counterclockwise" if self.clockwise else "clockwise"]
        self.travel = self.arc(self.home_deg, self.fire_deg)
        if not 0 < self.travel <= fcfg["max_travel_deg"]:
            raise ValueError(f"flipper sweep of {self.travel} deg is outside "
                             f"(0, {fcfg['max_travel_deg']}]; check home_deg/fire_deg/fire_direction")

    def arc(self, start: float, end: float) -> float:
        """Degrees from start to end moving in the fire direction."""
        return (end - start) % 360 if self.clockwise else (start - end) % 360

    def position(self) -> float | None:
        """Absolute encoder angle, or None if unknown."""
        if not self.fresh() or _bad(self.dev.motor.absolutePosition):
            return None
        return float(self.dev.motor.absolutePosition)

    def fire(self) -> None:
        self.dev.motor_run_to_absolute_position(self.fire_deg, direction=self.out_dir, speed=self.speed)
        self.dev.motor_run_to_absolute_position(self.home_deg, direction=self.back_dir, speed=self.speed)

    def home(self) -> None:
        pos = self.position()
        # Inside the sweep arc: go back the way we came. Anywhere else (bumped
        # out of the arc) take the shortest way, which can't cross the arc.
        inside = pos is not None and self.arc(self.home_deg, pos) <= self.travel
        direction = self.back_dir if inside else le.MOTOR_MOVE_DIRECTION_SHORTEST
        self.dev.motor_run_to_absolute_position(self.home_deg, direction=direction, speed=self.speed)


class LegoColor(_LegoDevice, ColorSensor):
    device_class = le.ColorSensor

    def read(self) -> ColorReading | None:
        s = self.dev.sensor if self.fresh() else None
        if s is None or _bad(s.color, s.rawRed, s.rawGreen, s.rawBlue):
            return None
        return ColorReading(COLOR_NAMES.get(int(s.color), "unknown"),
                            [int(s.rawRed), int(s.rawGreen), int(s.rawBlue)])


class LegoController(_LegoDevice, JoystickInput):
    """The LEGO Education Controller: two levers, used as tank drive."""

    device_class = le.Controller

    def __init__(self, filters: dict[str, Any], notify_ms: int, jcfg: dict[str, Any]) -> None:
        super().__init__(filters, notify_ms)
        self.sign_left = -1 if jcfg["invert_left"] else 1
        self.sign_right = -1 if jcfg["invert_right"] else 1
        self.swap = jcfg["swap_sides"]

    def read(self) -> tuple[float, float] | None:
        """(left wheels, right wheels) from the levers, forward = positive,
        after the invert and swap_sides settings."""
        s = self.dev.sensor if self.fresh() else None
        if s is None or _bad(s.leftPercent, s.rightPercent):
            return None
        left, right = s.leftPercent, s.rightPercent
        if self.swap:
            left, right = right, left
        return clamp(self.sign_left * left / 100), clamp(self.sign_right * right / 100)
