"""Copy the recordings from the board into data/ (existing files are overwritten).

  python pull_data.py

Uses tools/deploy.py from the local ArduinoApps folder to find the board (USB first,
then Wi-Fi), so the board settings live in one place: ArduinoApps/tools/board_secrets.py.
"""
from __future__ import annotations

import sys

import config

# deploy.py imports its own module called "config", so step ours aside while it loads.
_ours = sys.modules.pop("config")
sys.path.insert(0, str(config.ARDUINO_APPS / "tools"))
try:
    import deploy
finally:
    sys.modules["config"] = _ours


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    before = set(config.DATA_DIR.glob("*.csv"))
    board = deploy.find_board()
    kind, target = board
    if kind == "adb":
        deploy.run(["adb", "pull", f"{config.BOARD_DATA_DIR}/.", str(config.DATA_DIR)])
    else:
        # The * is expanded on the board, not here.
        deploy.run(["scp", *deploy.SSH_OPTS, "-q", f"{target}:{config.BOARD_DATA_DIR}/*.csv",
                    str(config.DATA_DIR)])
    after = set(config.DATA_DIR.glob("*.csv"))
    print(f"{len(after)} recordings in {config.DATA_DIR} ({len(after - before)} new)")


if __name__ == "__main__":
    main()
