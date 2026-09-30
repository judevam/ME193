"""Publish a single message to an MQTT topic, or listen on one and print
incoming messages, then exit - for one-off pub/sub commands where
listen.py's interactive chat loop or pubsub_demo.py's fixed topic/message
aren't what you want.

Usage:
    python publish.py <topic> <message>    # publish once and exit
    python publish.py listen <topic>       # print incoming messages until Ctrl+C

Example:
    python publish.py ME193/Rogers start
    python publish.py listen ME193/Rogers
"""

import sys
import time

from mqttlib import MQTTClient


def publish(topic: str, message: str) -> None:
    with MQTTClient() as client:
        client.publish(topic, message)
        time.sleep(1)  # give the message time to actually reach the broker before disconnecting
        print(f"Published {message!r} to {topic!r}")


def listen(topic: str) -> None:
    def on_message(topic, payload):
        print(f"[{topic}] {payload}")

    with MQTTClient() as client:
        client.subscribe(topic, on_message)
        print(f"Listening on {topic!r} - Ctrl+C to quit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "listen":
        listen(sys.argv[2])
    elif len(sys.argv) == 3:
        publish(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python publish.py <topic> <message>    # publish once and exit")
        print("  python publish.py listen <topic>       # print incoming messages until Ctrl+C")
        sys.exit(1)


if __name__ == "__main__":
    main()
