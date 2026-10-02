# Green Minifig Tracker (ME193)

A YOLO model on the laptop finds a green LEGO minifigure in the webcam feed and publishes its position over MQTT. An Arduino UNO Q receives the position and lights a blue dot at the matching spot on its 8×13 LED matrix.

```
webcam → YOLOv8 (laptop/detect_publish.py) → MQTT topic minifig/pos → UNO Q python/main.py → Bridge → sketch.ino → LED matrix
```

MQTT payload (x, y normalized 0..1, origin top-left):
`{"found": true, "x": 0.42, "y": 0.61, "conf": 0.88}` or `{"found": false}`

## Folders
- `laptop/`: runs on the laptop. Webcam capture, YOLO training, detection + MQTT publisher, Mosquitto config. See `laptop/README.md`.
- `unoq-led-app/`: the Arduino App for the UNO Q (`app.yaml`, `python/main.py`, `sketch/sketch.ino`).

## Training
Images were labeled in Roboflow and exported in YOLOv8 format. Training is done locally with Ultralytics (`laptop/train.py`), starting from `yolov8n.pt`.

## Running
1. Laptop: `mosquitto -c laptop/mosquitto.conf -v` (allow inbound TCP 1883 in the firewall).
2. UNO Q: set `MQTT_HOST` in `unoq-led-app/python/main.py` to the laptop's IP and start the app from App Lab.
3. Laptop: `python laptop/detect_publish.py`
