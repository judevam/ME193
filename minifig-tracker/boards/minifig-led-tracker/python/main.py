import json

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "192.168.1.100"  # TODO: set this to your laptop's IP on Tufts_Wireless (ipconfig)
MQTT_PORT = 1883
TOPIC = "minifig/pos"

MATRIX_COLS = 13
MATRIX_ROWS = 8


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
    except ValueError:
        return

    if not payload.get("found"):
        Bridge.call("clear_matrix")
        return

    col = min(MATRIX_COLS - 1, max(0, int(payload["x"] * MATRIX_COLS)))
    row = min(MATRIX_ROWS - 1, max(0, int(payload["y"] * MATRIX_ROWS)))
    Bridge.call("show_dot", col, row)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT} (rc={reason_code})")
    client.subscribe(TOPIC)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
client.loop_start()

print(f"Minifig LED Tracker: waiting for messages on '{TOPIC}'")

App.run()
