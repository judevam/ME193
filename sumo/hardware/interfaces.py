"""Abstract interfaces for every device. Each has a mock (hardware/mock.py) and
a real implementation (hardware/lego.py, hardware/sound.py, hardware/keys.py,
and the UNO Q app for the ultrasonic sensor)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ColorReading:
    name: str             # "blue", "white", "none", ...
    rgb: list[int]        # raw sensor values


class Device(ABC):
    """Something with a connection that can drop (BLE)."""

    def connect(self) -> None:
        """Connect; raise on failure."""

    @property
    def connected(self) -> bool:
        return True

    def close(self) -> None:
        """Disconnect and release the device."""


class DriveBase(Device):
    @abstractmethod
    def set(self, left: float, right: float) -> None:
        """Wheel speeds in [-1, 1] (values outside are clamped)."""

    def stop(self) -> None:
        self.set(0.0, 0.0)


class FlipperMotor(Device):
    @abstractmethod
    def fire(self) -> None:
        """One full stroke: home -> fire angle -> home. Blocks until done."""

    @abstractmethod
    def home(self) -> None:
        """Move the arm to its home angle."""


class ColorSensor(Device):
    @abstractmethod
    def read(self) -> ColorReading | None:
        """Latest reading, or None if there is no fresh data."""


class Ultrasonic(ABC):
    @abstractmethod
    def read_cm(self) -> float | None:
        """Distance in cm, or None for no echo."""


class SoundOutput(ABC):
    @abstractmethod
    def play(self, song: str) -> None:
        """Play a song from common.sounds.SONGS. Blocks until finished."""


class JoystickInput(Device):
    @abstractmethod
    def read(self) -> tuple[float, float] | None:
        """(left, right) lever positions in [-1, 1], or None if no fresh data."""


class KeyboardInput(ABC):
    @abstractmethod
    def get_key(self, timeout: float) -> str | None:
        """Next key pressed (e.g. "space", "m"), or None after timeout."""

    @abstractmethod
    def is_pressed(self, key: str) -> bool:
        """Whether key is held down right now."""

    def close(self) -> None:
        """Release keyboard hooks."""
