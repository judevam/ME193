"""Brain: feeds MQTT messages into the World, ticks the tree, publishes commands."""
from __future__ import annotations

import logging
import queue
import time
from typing import Any, Callable

from brain.tree import Context, active_nodes, build_tree
from brain.world import World
from common import schemas, topics
from common.bus import Bus

log = logging.getLogger("brain")

INPUTS = [
    topics.SENSOR_COLOR, topics.SENSOR_ULTRASONIC,
    topics.TELEOP_JOYSTICK, topics.TELEOP_FLIPPER_MODE, topics.TELEOP_FLIPPER_FIRE,
    topics.TELEOP_DEFEAT, topics.CONTROL_RESET,
]


class Brain:
    """Owns the tree. Call step() at brain.tick_hz."""

    def __init__(self, cfg: dict[str, Any], bus: Bus,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.bus = bus
        self.clock = clock
        self.world = World(cfg)
        self.ctx = Context(cfg, self.world)
        self.root = build_tree(self.ctx)
        self.last_state: dict[str, Any] | None = None
        self.print_ultrasonic = cfg["logging"].get("print_ultrasonic", False)
        self.last_us_print = float("-inf")
        # MQTT callbacks run on paho's thread; hand messages to the tick thread.
        self._inbox: queue.SimpleQueue[tuple[str, Any, float]] = queue.SimpleQueue()
        for topic in INPUTS:
            bus.subscribe(topic, self._receive)

    def _receive(self, topic: str, payload: Any) -> None:
        self._inbox.put((topic, payload, self.clock()))

    def step(self) -> None:
        """Apply queued messages, tick the tree once, publish the results."""
        while True:
            try:
                topic, payload, rx = self._inbox.get_nowait()
            except queue.Empty:
                break
            if topic == topics.CONTROL_RESET:
                log.warning("RESET: defeated latch cleared")
                self.ctx.reset()
            else:
                self.world.on_message(topic, payload, rx)

        self.ctx.begin_tick(self.clock())
        self.root.tick_once()

        # cmd/motors goes out every tick; it doubles as the bridge's keepalive.
        self.bus.publish(topics.CMD_MOTORS, schemas.motors(*self.ctx.motors, self.ctx.src_t))
        for action in self.ctx.flipper_cmds:
            self.bus.publish(topics.CMD_FLIPPER, schemas.flipper(action))
        for song in self.ctx.sounds:
            self.bus.publish(topics.CMD_SOUND, schemas.sound(song))

        node, flipper_node = active_nodes(self.root)
        state = {"node": node, "flipper_node": flipper_node,
                 "defeated": self.ctx.defeated, "flipper_mode": self.world.flipper_mode}
        if state != self.last_state:
            log.info("state %s", state)
            self.bus.publish(topics.STATE, schemas.state(**state))
            self.last_state = state
        log.debug("path ROOT>%s | motors %.2f %.2f", f"{node}+{flipper_node}", *self.ctx.motors)

        if self.print_ultrasonic and self.ctx.now - self.last_us_print >= 0.2:
            self.last_us_print = self.ctx.now
            d = self.world.distance(self.ctx.now)
            raw = self.world.us_raw
            log.info("ultrasonic raw %s cm | used %s", "-" if raw is None else f"{raw:5.1f}",
                     "none (stale/blanked)" if d is None else f"{d:5.1f} cm")

    def stop_motors(self) -> None:
        """Publish a final stop (used on shutdown)."""
        self.bus.publish(topics.CMD_MOTORS, schemas.motors(0.0, 0.0, None))
