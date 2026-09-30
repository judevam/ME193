import os
import time

import cv2

OUTPUT_DIR = "raw_images"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera 0")

    count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")])
    print("space = save frame, q = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            preview = frame.copy()
            cv2.putText(preview, f"saved: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("capture (space=save, q=quit)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                path = os.path.join(OUTPUT_DIR, f"minifig_{count:04d}_{int(time.time())}.jpg")
                cv2.imwrite(path, frame)
                count += 1
                print(f"saved {path}")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
