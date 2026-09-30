import json

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "192.168.1.100"  # TODO: set this to your laptop's IP on Tufts_Wireless (ipconfig)
MQTT_PORT = 1883
TOPIC = "minifig/pos"

# Must match CENTER_DEADBAND in laptop/part2_yolo_track.py
CENTER_DEADBAND = 0.08
DRIVE_SPEED = 150  # 0-255 PWM duty cycle

# Direction codes sent to the sketch: 1 = forward, -1 = backward, 0 = stop.
# If the robot drives the wrong way relative to the camera, swap these two.
FORWARD = 1
BACKWARD = -1
STOP = 0


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
    except ValueError:
        return

    if not payload.get("found"):
        Bridge.call("drive", STOP, 0)
        return

    x = payload["x"]
    if x < 0.5 - CENTER_DEADBAND:
        Bridge.call("drive", FORWARD, DRIVE_SPEED)
    elif x > 0.5 + CENTER_DEADBAND:
        Bridge.call("drive", BACKWARD, DRIVE_SPEED)
    else:
        Bridge.call("drive", STOP, 0)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT} (rc={reason_code})")
    client.subscribe(TOPIC)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
client.loop_start()

print(f"Minifig Motor Tracker: waiting for messages on '{TOPIC}'")

App.run()
