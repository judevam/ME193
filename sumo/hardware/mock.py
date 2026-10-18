"""Mock devices: log what a real device would do so the system runs with no robot."""
from __future__ import annotations

import logging
import math
import queue
import time

from common.schemas import clamp
from hardware.interfaces import (ColorReading, ColorSensor, DriveBase, FlipperMotor,
                                 JoystickInput, KeyboardInput, SoundOutput, Ultrasonic)

log = logging.getLogger("mock")


class MockDrive(DriveBase):
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []
        self.is_connected = True

    @property
    def connected(self) -> bool:
        return self.is_connected

    def connect(self) -> None:
        self.is_connected = True

    def set(self, left: float, right: float) -> None:
        if not self.is_connected:
            raise ConnectionError("mock drive disconnected")
        left, right = clamp(left), clamp(right)
        if not self.calls or self.calls[-1] != (left, right):
            log.info("wheels  L %+.2f  R %+.2f", left, right)
        self.calls.append((left, right))


class MockFlipper(FlipperMotor):
    def __init__(self, stroke_s: float = 0.0) -> None:
        self.calls: list[str] = []
        self.stroke_s = stroke_s

    def fire(self) -> None:
        self.calls.append("fire")
        log.info("flipper FIRE")
        time.sleep(self.stroke_s)

    def home(self) -> None:
        self.calls.append("home")
        log.info("flipper home")


class MockColor(ColorSensor):
    def __init__(self, name: str = "white") -> None:
        self.name = name

    def read(self) -> ColorReading | None:
        return ColorReading(self.name, [0, 0, 0])


class MockUltrasonic(Ultrasonic):
    """Fixed distance, or a slow sweep between 5 and 100 cm if cm is None."""

    def __init__(self, cm: float | None = None, period_s: float = 8.0) -> None:
        self.cm, self.period_s = cm, period_s

    def read_cm(self) -> float | None:
        if self.cm is not None:
            return self.cm
        phase = (time.monotonic() % self.period_s) / self.period_s
        return 52.5 - 47.5 * math.cos(2 * math.pi * phase)


class MockJoystick(JoystickInput):
    def __init__(self) -> None:
        self.value: tuple[float, float] | None = (0.0, 0.0)

    def read(self) -> tuple[float, float] | None:
        return self.value


class MockSound(SoundOutput):
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, song: str) -> None:
        self.played.append(song)
        log.info("(pretend) playing %r", song)


class MockKeyboard(KeyboardInput):
    """Scripted key presses for tests."""

    def __init__(self, keys: list[str] = ()) -> None:
        self.keys: queue.SimpleQueue[str] = queue.SimpleQueue()
        for k in keys:
            self.keys.put(k)
        self.held: set[str] = set()

    def get_key(self, timeout: float) -> str | None:
        try:
            return self.keys.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_pressed(self, key: str) -> bool:
        return key in self.held
