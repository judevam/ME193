"""Detect the green AND blue minifigs in one webcam feed and publish both over MQTT.

Runs every color's model on each frame and publishes one message per color to
the same topic. Each UNO Q runs unoq-drive-app with its own TARGET_COLOR and
ignores the other color's messages.

Topic minifig/pos, x/y normalized 0..1 (origin top-left):
  {"color": "green", "found": true, "x": 0.42, "y": 0.61, "conf": 0.88}
  {"color": "blue", "found": false}

Test without MQTT first:  python detect_publish_multi.py --no-mqtt
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--colors", nargs="+", default=["green", "blue"], help="trained models to run")
parser.add_argument("--broker", default="localhost")
parser.add_argument("--port", type=int, default=1883)
parser.add_argument("--topic", default="minifig/pos")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--flip", action="store_true", help="mirror the image horizontally")
parser.add_argument("--rate", type=float, default=10.0, help="max MQTT messages per second, per color")
parser.add_argument("--no-mqtt", action="store_true", help="only show detections, don't publish")
args = parser.parse_args()

# Dot color drawn on the preview for each minifig (BGR)
DOT_BGR = {"green": (0, 200, 0), "blue": (255, 0, 0)}

models = {}
for color in args.colors:
    weights = Path(__file__).parent / "runs" / "detect" / f"{color}_minifig" / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"No model at {weights} - run: python train.py --color {color}")
    models[color] = YOLO(weights)

client = None
if not args.no_mqtt:
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {args.camera}")

last_pub = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if args.flip:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    annotated = frame.copy()
    msgs = []
    for color, model in models.items():
        result = model(frame, conf=args.conf, verbose=False)[0]
        if len(result.boxes):
            best = int(result.boxes.conf.argmax())
            x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            conf = float(result.boxes.conf[best])
            msgs.append({"color": color, "found": True, "x": round(cx / w, 3), "y": round(cy / h, 3),
                         "conf": round(conf, 2)})
            bgr = DOT_BGR.get(color, (0, 255, 255))
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)
            cv2.putText(annotated, f"{color} {conf:.2f}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
            cv2.circle(annotated, (int(cx), int(cy)), 6, bgr, -1)
        else:
            msgs.append({"color": color, "found": False})

    now = time.time()
    if client and now - last_pub >= 1.0 / args.rate:
        for msg in msgs:
            client.publish(args.topic, json.dumps(msg))
        last_pub = now

    for i, msg in enumerate(msgs):
        cv2.putText(annotated, json.dumps(msg), (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.line(annotated, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)  # center line
    cv2.imshow("minifig detection (Esc to quit)", annotated)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
if client:
    client.loop_stop()
