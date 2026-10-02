"""Save webcam frames for labeling in Roboflow. Space = save, Esc = quit."""
import argparse
import os
import time

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--out", default="captures")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
cap = cv2.VideoCapture(args.camera)
count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    preview = frame.copy()
    cv2.putText(preview, f"saved: {count}  (space=save, esc=quit)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("capture", preview)
    key = cv2.waitKey(1)
    if key == 32:
        cv2.imwrite(os.path.join(args.out, f"img_{int(time.time() * 1000)}.jpg"), frame)
        count += 1
    elif key == 27:
        break
cap.release()
cv2.destroyAllWindows()
