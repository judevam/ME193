"""Shared helpers for long-running nodes: clean shutdown, fixed-rate loops, backoff."""
from __future__ import annotations

import signal
import threading
import time


class StopFlag:
    """Set on SIGINT, SIGTERM or (Windows) SIGBREAK so loops can exit and stop motors."""

    def __init__(self) -> None:
        self.event = threading.Event()
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, lambda *_: self.event.set())

    def is_set(self) -> bool:
        return self.event.is_set()

    def set(self) -> None:
        self.event.set()

    def sleep(self, seconds: float) -> bool:
        """Sleep, waking early on stop. Returns True if stopping."""
        return self.event.wait(max(0.0, seconds))


class Rate:
    """Sleep so a loop runs at roughly hz iterations per second."""

    def __init__(self, hz: float, stop: StopFlag) -> None:
        self.period = 1.0 / hz
        self.stop = stop
        self._next = time.monotonic()

    def sleep(self) -> bool:
        """Wait for the next iteration. Returns True if stopping."""
        self._next += self.period
        delay = self._next - time.monotonic()
        if delay < 0:              # fell behind; don't try to catch up
            self._next = time.monotonic()
            delay = 0
        return self.stop.sleep(delay)


class Backoff:
    """Doubling reconnect delay: min_s, 2*min_s, ... up to max_s."""

    def __init__(self, min_s: float, max_s: float) -> None:
        self.min_s, self.max_s = min_s, max_s
        self.delay = min_s
        self.next_try = 0.0

    def ready(self, now: float) -> bool:
        return now >= self.next_try

    def failed(self, now: float) -> None:
        self.next_try = now + self.delay
        self.delay = min(self.delay * 2, self.max_s)

    def reset(self) -> None:
        self.delay = self.min_s
        self.next_try = 0.0
