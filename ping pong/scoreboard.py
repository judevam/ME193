"""Publish the current streak to MQTT as a float, whenever it changes.

    python scoreboard.py 3       # publish 3.0 once, as a test
    python ../mqtt/publish.py listen ME193/Rogers/Jude    # watch it from another terminal

If the broker can't be reached, the game keeps running and just doesn't publish.
"""

import sys
import time

import config


class Scoreboard:
    def __init__(self, topic=config.MQTT_TOPIC, client=None):
        self.topic = topic
        self.client = client      # tests pass a fake; connect() makes the real one
        self.last = None          # last streak published

    def connect(self):
        from mqttlib import MQTTClient
        try:
            client = MQTTClient(broker=config.MQTT_BROKER)
            client.connect()
        except (OSError, ConnectionError) as e:
            print(f"MQTT offline ({e}); the score won't be published.")
            return False
        self.client = client
        print(f"Publishing the streak to {config.MQTT_BROKER} {self.topic}")
        return True

    def update(self, streak):
        """Publish float(streak) if it changed since the last publish (the first call always does)."""
        if streak == self.last or self.client is None:
            return False
        self.client.publish(self.topic, str(float(streak)))
        self.last = streak
        return True

    def close(self):
        if self.client is not None:
            self.client.disconnect()
            self.client = None


def main():
    value = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    board = Scoreboard()
    if board.connect():
        board.update(value)
        time.sleep(1)   # let it reach the broker before disconnecting
        print(f"Published {float(value)} to {board.topic}")
        board.close()


if __name__ == "__main__":
    main()
