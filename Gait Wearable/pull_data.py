"""Copy the recordings from the board into data/ (existing files are overwritten).

  python pull_data.py
  python pull_data.py --out data_pilot --clear-board   # pilot run: separate folder, then
                                                       # delete them on the board

Uses tools/deploy.py from the local ArduinoApps folder to find the board (USB first,
then Wi-Fi), so the board settings live in one place: ArduinoApps/tools/board_secrets.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import config

# deploy.py imports its own module called "config", so step ours aside while it loads.
_ours = sys.modules.pop("config")
sys.path.insert(0, str(config.ARDUINO_APPS / "tools"))
try:
    import deploy
finally:
    sys.modules["config"] = _ours


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=config.DATA_DIR, help="folder to copy into (default data/)")
    p.add_argument("--clear-board", action="store_true",
                   help="delete the recordings on the board after copying them")
    args = p.parse_args()

    out = args.out if args.out.is_absolute() else config.HERE / args.out
    out.mkdir(exist_ok=True)
    before = set(out.glob("*.csv"))
    board = deploy.find_board()
    kind, target = board
    if kind == "adb":
        r = deploy.run(["adb", "pull", f"{config.BOARD_DATA_DIR}/.", str(out)], check=False)
    else:
        # The * is expanded on the board, not here.
        r = deploy.run(["scp", *deploy.SSH_OPTS, "-q", f"{target}:{config.BOARD_DATA_DIR}/*.csv",
                        str(out)], check=False)
    after = set(out.glob("*.csv"))
    print(f"{len(after)} recordings in {out} ({len(after - before)} new)")
    if args.clear_board:
        if r is None or r.returncode != 0:
            sys.exit("copy failed, so the board's recordings were NOT deleted")
        deploy.remote(board, f"rm -f {config.BOARD_DATA_DIR}/*.csv")
        print("deleted the recordings on the board")


if __name__ == "__main__":
    main()
