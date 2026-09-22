"""Round-trip MQTT pub/sub demo: subscribe and publish in the same script to
confirm the connection to the broker actually works end to end - the broker
relays every publish back to any subscriber on that topic, including
yourself.

Usage:
    python pubsub_demo.py
"""

import time

from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    time.sleep(1)  # give the subscription time to reach the broker

    client.publish(TOPIC, "3 definitely 3")
    print(f"Published 'hello world' to '{TOPIC}' on test.mosquitto.org")

    time.sleep(1)  # give the message time to come back before disconnecting
