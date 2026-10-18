"""Start the broker (if needed) and every process.

  python run_demo.py              # everything mocked, fake sensors on the keyboard
  python run_demo.py --hardware   # real LEGO devices per config/sumo.yaml (UNO Q runs separately)

Stop with q (keyboard client) or Ctrl+C. All processes are stopped cleanly, so
the motors get a final stop command.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"

# Re-run with this project's venv if started with some other Python
# (e.g. a different venv that is first on PATH and lacks py_trees).
VENV_PY = ROOT / "venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
if VENV_PY.exists() and Path(sys.prefix).resolve() != (ROOT / "venv").resolve():
    child = subprocess.Popen([str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    while True:
        try:
            sys.exit(child.wait())
        except KeyboardInterrupt:
            pass  # the child got Ctrl+C too and is shutting down

from common.config import load_config  # noqa: E402

MOCK_PROCESSES = ["brain.main", "bridges.drive_bridge", "bridges.flipper_bridge",
                  "bridges.sound_bridge", "teleop.keyboard_client", "sensors.fake_sensors"]
HARDWARE_PROCESSES = ["brain.main", "bridges.drive_bridge", "bridges.flipper_bridge",
                      "bridges.sound_bridge", "sensors.color_bridge",
                      "teleop.joystick_bridge", "teleop.keyboard_client"]


def broker_up(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_broker(port: int) -> subprocess.Popen | None:
    """Start Mosquitto locally, or in Docker. Returns the process, or exits."""
    exe = shutil.which("mosquitto")
    if not exe and WINDOWS and Path(r"C:\Program Files\mosquitto\mosquitto.exe").exists():
        exe = r"C:\Program Files\mosquitto\mosquitto.exe"
    if exe:
        print(f"starting {exe} on port {port}")
        return subprocess.Popen([exe, "-p", str(port)])
    if shutil.which("docker"):
        print("starting Mosquitto in Docker")
        return subprocess.Popen(["docker", "run", "--rm", "-p", f"{port}:1883",
                                 "eclipse-mosquitto:2", "mosquitto", "-c", "/mosquitto-no-auth.conf"])
    sys.exit("No MQTT broker running and couldn't find mosquitto or docker. See README.")


def spawn(module: str, env: dict[str, str]) -> subprocess.Popen:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0
    return subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=env, creationflags=flags)


def stop_all(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
    deadline = time.monotonic() + 5
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true", help="use real devices from config")
    args = parser.parse_args()

    env = dict(os.environ)
    if not args.hardware:
        env["SUMO_FORCE_MOCK"] = "1"
    cfg = load_config()
    host, port = cfg["mqtt"]["host"], cfg["mqtt"]["port"]

    broker = None
    if not broker_up(host, port):
        broker = start_broker(port)
        for _ in range(20):
            if broker_up(host, port):
                break
            time.sleep(0.25)
        else:
            sys.exit("broker didn't come up")

    modules = HARDWARE_PROCESSES if args.hardware else MOCK_PROCESSES
    procs = [spawn(m, env) for m in modules]
    keyboard = procs[modules.index("teleop.keyboard_client")]
    try:
        while keyboard.poll() is None:   # q in the keyboard client ends the demo
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("stopping...")
        stop_all(procs)
        if broker:
            broker.terminate()


if __name__ == "__main__":
    main()
