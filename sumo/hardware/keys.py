"""Real keyboard input using the `keyboard` package (global hotkeys: works no
matter which window has focus; on macOS/Linux it needs root)."""
from __future__ import annotations

import queue

import keyboard

from hardware.interfaces import KeyboardInput


class GlobalKeyboard(KeyboardInput):
    def __init__(self, keys: list[str]) -> None:
        """Report presses of the listed keys (names as the keyboard package uses them)."""
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._hooks = [keyboard.on_press_key(k, lambda e, k=k: self._queue.put(k), suppress=False)
                       for k in keys]

    def get_key(self, timeout: float) -> str | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_pressed(self, key: str) -> bool:
        return keyboard.is_pressed(key)

    def close(self) -> None:
        for hook in self._hooks:
            keyboard.unhook(hook)
