import sys
import time

import cv2

from mqtt_utils import MQTT_HOST, MQTT_PORT, TOPIC, connect, publish_position

MODEL_PATH = "models/minifig_yolo.pt"
CONFIDENCE_THRESHOLD = 0.5
PUBLISH_HZ = 12

# Must match the deadband used in the board's Minifig Motor Tracker app.
CENTER_DEADBAND = 0.08


def main():
    from ultralytics import YOLO  # deferred: slow import, only needed here

    model = YOLO(MODEL_PATH)

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
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

            best = None
            if results.boxes is not None and len(results.boxes) > 0:
                best_idx = results.boxes.conf.argmax().item()
                best = results.boxes[best_idx]

            norm_x = norm_y = None
            if best is not None:
                x1, y1, x2, y2 = best.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                norm_x, norm_y = cx / w, cy / h

            now = time.time()
            if now - last_publish >= period:
                if best is not None:
                    publish_position(client, norm_x, norm_y, found=True)
                else:
                    publish_position(client, 0, 0, found=False)
                last_publish = now

            if best is not None:
                conf = best.conf[0].item()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 6, (255, 0, 0), -1)
                cv2.putText(frame, f"minifig {conf:.2f}", (int(x1), max(0, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if abs(norm_x - 0.5) <= CENTER_DEADBAND:
                    cv2.putText(frame, "CENTERED", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            cv2.imshow("Part 2: YOLO minifig tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
