import sys
import time

import cv2
import numpy as np

from mqtt_utils import MQTT_HOST, MQTT_PORT, TOPIC, connect, publish_position

# Tune these for your green minifig / lighting. OpenCV hue range is 0-179.
GREEN_LOWER = np.array([40, 60, 60])
GREEN_UPPER = np.array([85, 255, 255])

MIN_CONTOUR_AREA = 200  # px^2, filters out small color noise
PUBLISH_HZ = 12


def find_green_centroid(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(largest)
    return (x + w / 2, y + h / 2, x, y, w, h), mask


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0", file=sys.stderr)
        sys.exit(1)

    client = connect()
    print(f"Publishing to {MQTT_HOST}:{MQTT_PORT} topic '{TOPIC}' (press q to quit)")

    period = 1.0 / PUBLISH_HZ
    last_publish = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            result, mask = find_green_centroid(frame)

            now = time.time()
            if now - last_publish >= period:
                if result:
                    cx, cy, *_ = result
                    publish_position(client, cx / w, cy / h, found=True)
                else:
                    publish_position(client, 0, 0, found=False)
                last_publish = now

            if result:
                cx, cy, bx, by, bw, bh = result
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 6, (255, 0, 0), -1)

            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            cv2.imshow("Part 1: green minifig tracker", frame)
            cv2.imshow("mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
