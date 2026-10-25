"""Send the trained models to the wearable so it classifies on its own.

  python push_models.py                          # models/ -> board, then redeploy the app
  python push_models.py --models data_synthetic/models
  python push_models.py --no-deploy              # only copy into ArduinoApps/gait-wearable
  python push_models.py --remove                 # take the models off the board again

Copies models/gait_models.json plus the code that runs it (gait_models.py, features_core.py)
into ArduinoApps/gait-wearable/python/, then runs `tools/deploy.py gait-wearable`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config

APP_PY = config.ARDUINO_APPS / "gait-wearable" / "python"
CODE = ["gait_models.py", "features_core.py"]
MODEL_FILE = "gait_models.json"


def deploy(*args: str) -> int:
    return subprocess.call([sys.executable, "tools/deploy.py", *args], cwd=config.ARDUINO_APPS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=config.MODEL_DIR)
    p.add_argument("--no-deploy", action="store_true")
    p.add_argument("--remove", action="store_true")
    args = p.parse_args()

    if args.remove:
        (APP_PY / MODEL_FILE).unlink(missing_ok=True)
        # deploy.py copies but never deletes, so remove the board's copy by hand
        board_file = f"/home/arduino/ArduinoApps/gait-wearable/python/{MODEL_FILE}"
        deploy("--cmd", f"rm -f {board_file}")
        print("models removed; redeploying")
        sys.exit(deploy("gait-wearable"))

    src = args.models / MODEL_FILE
    if not src.exists():
        sys.exit(f"{src} not found: run train.py first")
    for name in CODE:
        shutil.copy2(config.HERE / name, APP_PY / name)
    shutil.copy2(src, APP_PY / MODEL_FILE)
    print(f"copied {MODEL_FILE} ({src.stat().st_size // 1024} KB), {', '.join(CODE)} -> {APP_PY}")
    if not args.no_deploy:
        sys.exit(deploy("gait-wearable"))


if __name__ == "__main__":
    main()
