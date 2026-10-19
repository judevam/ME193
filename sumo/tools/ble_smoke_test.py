"""Check every LEGO device before running the full system.

  python -m tools.ble_smoke_test                 # all four devices
  python -m tools.ble_smoke_test --only color    # just one (drive|flipper|color|joystick)

It scans, connects to all of them at once (using the filters in config/sumo.yaml),
exercises each one, then checks that all connections stay up together for 10 s.
Use the printed values to fill in the TODO(hardware) settings in the config:
flipper home/fire angles, tape color names, lever and wheel directions.
"""
from __future__ import annotations

import argparse
import logging
import time

import legoeducation as le
from legoeducation.basic_device import VersionMismatchError

from common.config import load_config
from hardware.lego import LegoColor, LegoController, LegoDrive, LegoFlipper, lego_filters

DEVICES = ["drive", "flipper", "color", "joystick"]
CLASSES = {"drive": le.DoubleMotor, "flipper": le.SingleMotor,
           "color": le.ColorSensor, "joystick": le.Controller}


def make(name: str, cfg: dict):
    lego, notify = cfg["lego"], cfg["lego"]["notify_ms"]
    if name == "drive":
        return LegoDrive(lego["drive"], notify, cfg["drive_bridge"])
    if name == "flipper":
        # The test only reads angles and makes its own small move, so it must
        # work before home/fire are calibrated: give it a harmless sweep.
        calibration = {**cfg["flipper"], "home_deg": 0, "fire_deg": 30, "fire_direction": "clockwise"}
        return LegoFlipper(lego["flipper"], notify, calibration)
    if name == "color":
        return LegoColor(lego["color"], notify)
    return LegoController(lego["joystick"], notify, cfg["joystick"])


def watch(label: str, read, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        print(f"  {label}: {read()}          ", end="\r", flush=True)
        time.sleep(0.1)
    print()


def exercise(name: str, dev, ask: bool) -> None:
    if name == "drive":
        if ask:
            input("  Lift the robot so the wheels are off the table, then press Enter...")
        print("  both wheels forward 30% for 1 s (should roll the robot FORWARD)")
        dev.set(0.3, 0.3); time.sleep(1); dev.stop()
        print("  left wheel only, forward 30% for 1 s")
        dev.set(0.3, 0); time.sleep(1); dev.stop()
        print("  -> if a wheel went the wrong way, set drive_bridge.invert_left/right")
    elif name == "flipper":
        def settled() -> float | None:
            time.sleep(0.5)                  # let a fresh position notification arrive
            return dev.position()
        input("  Hold the arm DOWN (resting, home) by hand, then press Enter...")
        down = settled()
        input("  Hold the arm UP (fully flipped, fire) by hand, then press Enter...")
        up = settled()
        print(f"  -> flipper.home_deg: {down}   flipper.fire_deg: {up}")
        if ask:
            input("  Put the arm DOWN and keep it clear. It will turn 30 deg clockwise and back slowly. Enter...")
        before = settled()
        dev.dev.motor_run_for_degrees(30, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=20)
        after = settled()
        print(f"  30 deg clockwise: {before} -> {after}  (and did the arm go up or down?)")
        dev.dev.motor_run_for_degrees(30, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=20)
    elif name == "color":
        print("  hold the sensor over the ring floor, then the blue tape")
        watch("color", lambda: dev.read(), 10)
        print("  -> put the tape's name in edge.blue_names")
    elif name == "joystick":
        print("  push each lever forward and back")
        watch("left, right", dev.read, 8)
        print("  -> forward should be positive; otherwise set joystick.invert_left/right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=DEVICES)
    parser.add_argument("--yes", action="store_true", help="don't pause before moving motors")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    names = [args.only] if args.only else DEVICES
    result: dict[str, str] = {}

    print("== scan ==")
    for name in names:
        filters = lego_filters(cfg["lego"][name])
        found = CLASSES[name]().search(timeout=5, **filters)
        print(f"{name:9s} {len(found or [])} found {found or ''}")

    print("\n== connect (all at once) ==")
    devices = {}
    for name in names:
        dev = make(name, cfg)
        try:
            dev.connect()
            devices[name] = dev
            print(f"{name:9s} connected")
        except VersionMismatchError as e:
            result[name] = "VERSION MISMATCH"
            print(f"{name:9s} VERSION MISMATCH:\n{e}\n"
                  "  -> update firmware in the LEGO Education Coding Canvas app, "
                  "or install the matching legoeducation version")
        except Exception as e:
            result[name] = f"connect failed: {e}"
            print(f"{name:9s} connect failed: {e}")

    try:
        for name, dev in devices.items():
            print(f"\n== {name} ==")
            try:
                exercise(name, dev, ask=not args.yes)
                result[name] = "ok"
            except Exception as e:
                result[name] = f"error: {e}"
                print(f"  error: {e}")

        if len(devices) > 1:
            print(f"\n== holding {len(devices)} connections for 10 s ==")
            drops: set[str] = set()
            end = time.monotonic() + 10
            while time.monotonic() < end:
                for name, dev in devices.items():
                    if not dev.fresh():
                        drops.add(name)
                time.sleep(0.2)
            for name in devices:
                if name in drops:
                    result[name] += ", but DROPPED/went quiet while all were connected"
            print("all stayed connected" if not drops else f"problems: {sorted(drops)}")
    finally:
        for dev in devices.values():
            dev.close()

    print("\n== summary ==")
    for name in names:
        print(f"{name:9s} {result.get(name, '?')}")


if __name__ == "__main__":
    main()
