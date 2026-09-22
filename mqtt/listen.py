"""Listen for anything published to the shared ME193 MQTT topic and print it -
and type a line + Enter to publish it back, like a text chat.

Runs until you stop it with Ctrl+C. Since test.mosquitto.org is public,
you'll see messages from anyone else using the same topic name too - not
just the people you're actually chatting with.

Incoming messages arrive on a background thread (started inside connect())
and can print at any moment, including while you're mid-line typing your
own message - that's normal here, not a bug, just a quirk of a plain
terminal not knowing to redraw your half-typed input around it.

Usage:
    python listen.py
"""

from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"\n[{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    print(f"Chatting on '{TOPIC}' - type a message and press Enter to send, Ctrl+C to quit.")
    try:
        while True:
            message = input("> ")
            if message:
                client.publish(TOPIC, message)
    except KeyboardInterrupt:
        print("\nStopped.")
