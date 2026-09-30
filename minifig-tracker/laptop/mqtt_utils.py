import json
import os

import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get("MQTT_HOST", "JudeUnoQ.local")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC = "minifig/pos"


def connect():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
    client.loop_start()
    return client


def publish_position(client, x, y, found=True):
    payload = {"found": found}
    if found:
        payload["x"] = round(float(x), 4)
        payload["y"] = round(float(y), 4)
    client.publish(TOPIC, json.dumps(payload), qos=0)
