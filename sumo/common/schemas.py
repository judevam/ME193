"""Builders and parsers for every payload. Every message carries "t" (epoch seconds)."""
from __future__ import annotations

import math
import time
from typing import Any

FLIPPER_MODES = ("auto", "manual")
FLIPPER_ACTIONS = ("fire", "home")


def now_t() -> float:
    """Wall-clock timestamp used in payloads."""
    return time.time()


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi]; NaN becomes 0."""
    if value != value:  # NaN
        return 0.0
    return max(lo, min(hi, value))


# ---- builders -------------------------------------------------------------

def color(name: str, rgb: list[int]) -> dict[str, Any]:
    return {"name": name, "rgb": list(rgb), "t": now_t()}


def ultrasonic(cm: float) -> dict[str, Any]:
    return {"cm": round(float(cm), 1), "t": now_t()}


def joystick(left: float, right: float) -> dict[str, Any]:
    return {"left": clamp(left), "right": clamp(right), "t": now_t()}


def event() -> dict[str, Any]:
    """Payload for fire, defeat and reset events."""
    return {"t": now_t()}


def motors(left: float, right: float, src_t: float | None) -> dict[str, Any]:
    return {"left": clamp(left), "right": clamp(right), "t": now_t(), "src_t": src_t}


def flipper(action: str) -> dict[str, Any]:
    assert action in FLIPPER_ACTIONS
    return {"action": action, "t": now_t()}


def sound(song: str) -> dict[str, Any]:
    return {"song": song, "t": now_t()}


def state(node: str, flipper_node: str, defeated: bool, flipper_mode: str) -> dict[str, Any]:
    return {"node": node, "flipper_node": flipper_node, "defeated": defeated,
            "flipper_mode": flipper_mode, "t": now_t()}


# ---- parsers (return None for anything malformed) -------------------------

def _num(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    v = payload.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v):
        return None
    return float(v)


def parse_color(p: Any) -> tuple[str, float | None] | None:
    """-> (name, t)"""
    if isinstance(p, dict) and isinstance(p.get("name"), str):
        return p["name"], _num(p, "t")
    return None


def parse_ultrasonic(p: Any) -> tuple[float, float | None] | None:
    """-> (cm, t)"""
    cm = _num(p, "cm")
    return None if cm is None else (cm, _num(p, "t"))


def parse_joystick(p: Any) -> tuple[float, float, float | None] | None:
    """-> (left, right, t)"""
    left, right = _num(p, "left"), _num(p, "right")
    if left is None or right is None:
        return None
    return clamp(left), clamp(right), _num(p, "t")


def parse_motors(p: Any) -> tuple[float, float, float | None] | None:
    """-> (left, right, src_t), clamped to [-1, 1]"""
    left, right = _num(p, "left"), _num(p, "right")
    if left is None or right is None:
        return None
    return clamp(left), clamp(right), _num(p, "src_t")


def parse_flipper(p: Any) -> str | None:
    if isinstance(p, dict) and p.get("action") in FLIPPER_ACTIONS:
        return p["action"]
    return None


def parse_mode(p: Any) -> str | None:
    return p if p in FLIPPER_MODES else None
