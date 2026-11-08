"""Quick webcam check on the computer (Finger Lock project).

  python src/webcam_test.py --list      find which camera indexes work
  python src/webcam_test.py --index 1   live preview of one camera

In the preview: s = save a snapshot, q or Esc = quit.
Index 0 is usually the laptop's built-in camera. The project must use the
external USB webcam, so look for the other index.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def list_cameras(max_index=6):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, BACKEND)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            print(f"  index {i}: works, {w}x{h}")
            found.append(i)
    if not found:
        print("  no cameras found")
    return found


def preview(index):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        print(f"Could not open camera {index}")
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    frames, t0, fps = 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed")
            break
        frames += 1
        if time.time() - t0 >= 1:
            fps, frames, t0 = frames / (time.time() - t0), 0, time.time()
        h, w = frame.shape[:2]
        shown = frame.copy()
        cv2.putText(shown, f"cam {index}  {w}x{h}  {fps:.0f} FPS  (s=save, q=quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("webcam test", shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = OUT_DIR / f"snapshot_cam{index}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list working camera indexes")
    ap.add_argument("--index", type=int, help="camera index to preview")
    args = ap.parse_args()
    if args.list or args.index is None:
        print("Scanning cameras...")
        list_cameras()
        if args.index is None:
            print("Now run: python src/webcam_test.py --index N")
            sys.exit(0)
    sys.exit(preview(args.index))
