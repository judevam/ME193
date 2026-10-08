"""mqttlib - a small wrapper around paho-mqtt for pub/sub over MQTT.

By default connects to the public test.mosquitto.org broker (unauthenticated
- anyone can publish/subscribe to any topic, so use a specific topic name,
not something generic like "test", to avoid crosstalk with other users'
traffic). See MQTTLIB.md (course "useful libraries") for the full reference
this was built from.

    from mqttlib import MQTTClient

    with MQTTClient() as client:
        client.publish("ME193", "hello world")
"""

import threading

import paho.mqtt.client as mqtt

DEFAULT_BROKER = "test.mosquitto.org"
DEFAULT_PORT = 1883


class MQTTClient:
    def __init__(self, broker: str = DEFAULT_BROKER, port: int = DEFAULT_PORT, client_id: str = ""):
        self._broker = broker
        self._port = port
        self._connected_event = threading.Event()

        self._client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties):
        if reason_code == 0:
            self._connected_event.set()

    def connect(self, timeout: float = 5):
        """Connect to the broker and start the background network thread.
        Blocks until the broker confirms the connection, or raises
        ConnectionError if that doesn't happen within `timeout` seconds."""
        self._connected_event.clear()
        self._client.connect(self._broker, self._port)
        self._client.loop_start()

        if not self._connected_event.wait(timeout):
            self._client.loop_stop()
            raise ConnectionError(f"Could not connect to MQTT broker {self._broker}:{self._port} within {timeout}s")

    def disconnect(self):
        """Stop the background thread and close the connection."""
        self._client.disconnect()
        self._client.loop_stop()

    def publish(self, topic: str, message: str, qos: int = 0, retain: bool = False):
        self._client.publish(topic, message, qos=qos, retain=retain)

    def subscribe(self, topic: str, callback, qos: int = 0):
        """Call callback(topic, payload) - both strings - every time a
        message arrives on `topic`. Replaces any previous callback
        registered for that exact topic string."""

        def _on_message(client, userdata, message):
            callback(message.topic, message.payload.decode("utf-8", errors="replace"))

        self._client.message_callback_add(topic, _on_message)
        self._client.subscribe(topic, qos=qos)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
