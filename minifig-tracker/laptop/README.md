# Laptop side: YOLO detection + MQTT publisher

Run these on the laptop (it has the webcam and the MQTT broker), not on the UNO Q.

1. `pip install -r requirements.txt`
2. Unzip the Roboflow export (YOLOv8 / YOLO11 format) into `dataset/` so `dataset/data.yaml` exists.
3. Train: `python train.py` → `runs/detect/minifig/weights/best.pt`
4. Start the broker: `mosquitto -c mosquitto.conf -v` and allow TCP 1883 inbound in the firewall.
5. Run: `python detect_publish.py` (add `--flip` if left/right looks mirrored, `--camera 1` for another webcam)

Debug: `mosquitto_sub -h <laptop-ip> -t minifig/pos -v`
