import json
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "10.247.137.40"  # laptop running Mosquitto (check with ipconfig if it changes)
MQTT_PORT = 1883
TOPIC = "minifig/pos"

# x within 0.5 +/- CENTER_DEADBAND counts as centered (stop)
CENTER_DEADBAND = 0.08
DRIVE_SPEED = 150  # 0-255 PWM duty cycle

# Direction codes sent to the sketch: 1 = forward, -1 = backward, 0 = stop.
# Swapped after testing: the motors are wired so that 1 physically drives backward.
FORWARD = -1
BACKWARD = 1
STOP = 0


def drive(direction, speed):
    try:
        Bridge.call("drive", direction, speed)
        print(f"drive({direction}, {speed}) ok")
    except Exception as e:
        print(f"drive({direction}, {speed}) failed: {e!r}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
    except ValueError:
        return
    print(f"recv {payload}")

    if not payload.get("found"):
        drive(STOP, 0)
        return

    x = payload["x"]
    if x < 0.5 - CENTER_DEADBAND:
        drive(FORWARD, DRIVE_SPEED)
    elif x > 0.5 + CENTER_DEADBAND:
        drive(BACKWARD, DRIVE_SPEED)
    else:
        drive(STOP, 0)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT} (rc={reason_code})")
    client.subscribe(TOPIC)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
# Keep retrying until the broker is reachable (Wi-Fi may not be up yet at boot,
# or the laptop's broker may not be running yet). After the first connection,
# paho's loop thread reconnects on its own.
while True:
    try:
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
        break
    except OSError as e:
        print(f"MQTT broker {MQTT_HOST}:{MQTT_PORT} not reachable ({e}), retrying in 5 s")
        time.sleep(5)
client.loop_start()

print(f"Minifig Motor Tracker: waiting for messages on '{TOPIC}'")

App.run()
