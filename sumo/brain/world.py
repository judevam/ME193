"""What the brain currently believes about the world, built from MQTT messages.

Freshness uses the brain's own clock at the moment a message *arrives*, not the
sender's "t", because the UNO Q's clock and the laptop's clock can disagree.
The sender's "t" is only used for latency logging.
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Any

from common import schemas, topics


class World:
    """Latest sensor/teleop data plus the filters the spec asks for."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        b, e, u = cfg["brain"], cfg["edge"], cfg["ultrasonic"]
        self.teleop_stale = b["teleop_stale_s"]
        self.sensor_stale = b["sensor_stale_s"]
        self.blue_names = set(e["blue_names"])
        self.max_cm = u["max_cm"]
        self.blank_s = u["blank_after_fire_s"]

        # color
        self.color: str | None = None
        self.color_rx: float | None = None
        self.color_t: float | None = None
        # ultrasonic: (rx_time, cm) of recent valid readings
        self.us: deque[tuple[float, float]] = deque(maxlen=u["median_n"])
        self.us_t: float | None = None
        self.us_raw: float | None = None   # last reading, even if discarded (debug print)
        self.blank_until = 0.0
        # teleop
        self.joy: tuple[float, float] = (0.0, 0.0)
        self.joy_rx: float | None = None
        self.joy_t: float | None = None
        self.fire_rx: float | None = None
        self.flipper_mode = "auto"
        self.manual_defeat = False

    # ---- incoming messages ----------------------------------------------

    def on_message(self, topic: str, payload: Any, now: float) -> None:
        """Fold one message into the world state."""
        if topic == topics.SENSOR_COLOR:
            parsed = schemas.parse_color(payload)
            if parsed:
                self.color, self.color_rx, self.color_t = parsed[0], now, parsed[1]
        elif topic == topics.SENSOR_ULTRASONIC:
            parsed = schemas.parse_ultrasonic(payload)
            if parsed:
                self._on_ultrasonic(parsed[0], parsed[1], now)
        elif topic == topics.TELEOP_JOYSTICK:
            parsed = schemas.parse_joystick(payload)
            if parsed:
                self.joy, self.joy_rx, self.joy_t = parsed[:2], now, parsed[2]
        elif topic == topics.TELEOP_FLIPPER_FIRE:
            self.fire_rx = now
        elif topic == topics.TELEOP_FLIPPER_MODE:
            mode = schemas.parse_mode(payload)
            if mode:
                self.flipper_mode = mode
        elif topic == topics.TELEOP_DEFEAT:
            self.manual_defeat = True

    def _on_ultrasonic(self, cm: float, t: float | None, now: float) -> None:
        self.us_raw = cm
        if now < self.blank_until or not 0 < cm <= self.max_cm:
            return
        if self.us and now - self.us[-1][0] > self.sensor_stale:
            self.us.clear()  # gap in readings: don't mix old and new
        self.us.append((now, cm))
        self.us_t = t

    # ---- queries used by the tree ---------------------------------------

    def color_fresh(self, now: float) -> bool:
        return self.color_rx is not None and now - self.color_rx <= self.sensor_stale

    def on_blue(self, now: float) -> bool:
        return self.color_fresh(now) and self.color in self.blue_names

    def distance(self, now: float) -> float | None:
        """Median of recent valid ultrasonic readings, or None if stale/blanked."""
        if not self.us or now - self.us[-1][0] > self.sensor_stale:
            return None
        return statistics.median(cm for _, cm in self.us)

    def joystick(self, now: float) -> tuple[float, float] | None:
        """Latest lever positions if fresh, else None."""
        if self.joy_rx is None or now - self.joy_rx > self.teleop_stale:
            return None
        return self.joy

    def fire_requested(self, now: float) -> bool:
        return self.fire_rx is not None and now - self.fire_rx <= self.teleop_stale

    # ---- changes made by the brain --------------------------------------

    def consume_fire(self) -> None:
        self.fire_rx = None

    def start_blanking(self, now: float) -> None:
        """Ignore ultrasonic while the flipper arm sweeps through its view."""
        self.blank_until = now + self.blank_s
        self.us.clear()

    def reset(self) -> None:
        """Clear the defeat key press."""
        self.manual_defeat = False
