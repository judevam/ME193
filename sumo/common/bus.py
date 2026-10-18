"""Message bus: the real MQTT client, and an in-memory fake for tests.

Both take short topic names ("cmd/motors") and add the robot prefix.
Payloads are JSON on the wire; handlers receive the decoded value.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Protocol

import paho.mqtt.client as mqtt

from common import topics

Handler = Callable[[str, Any], None]   # (short_topic, payload)
log = logging.getLogger(__name__)


class Bus(Protocol):
    def publish(self, topic: str, payload: Any) -> None: ...
    def subscribe(self, topic: str, handler: Handler) -> None: ...


def decode(raw: bytes) -> Any:
    """JSON-decode a payload; fall back to the plain string."""
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


class MqttBus:
    """paho-mqtt client with a retained online/offline status and Last Will."""

    def __init__(self, cfg: dict[str, Any], node: str) -> None:
        self.node = node
        self.prefix = topics.prefix(cfg["robot_id"])
        self._handlers: dict[str, list[Handler]] = {}
        self._status = self.prefix + topics.status_topic(node)
        self._host, self._port = cfg["mqtt"]["host"], cfg["mqtt"]["port"]

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"sumo-{cfg['robot_id']}-{node}")
        self.client.will_set(self._status, json.dumps("offline"), qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=5)
        self.client.connect_async(self._host, self._port, keepalive=cfg["mqtt"]["keepalive_s"])
        self.client.loop_start()
        # Give the first connection a moment, so early publishes aren't dropped.
        deadline = time.monotonic() + 5
        while not self.client.is_connected() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.client.is_connected():
            log.error("can't reach the MQTT broker at %s:%s yet; still trying", self._host, self._port)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code.is_failure:
            log.error("MQTT connect to %s:%s failed: %s", self._host, self._port, reason_code)
            return
        log.info("MQTT connected to %s:%s", self._host, self._port)
        client.publish(self._status, json.dumps("online"), qos=1, retain=True)
        for topic in self._handlers:
            client.subscribe(self.prefix + topic, qos=topics.qos_for(topic))

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code.is_failure:
            log.warning("MQTT disconnected (%s), retrying", reason_code)

    def _on_message(self, client, userdata, msg) -> None:
        short = msg.topic[len(self.prefix):]
        payload = decode(msg.payload)
        for handler in self._handlers.get(short, []):
            try:
                handler(short, payload)
            except Exception:
                log.exception("handler for %s failed", short)

    def publish(self, topic: str, payload: Any) -> None:
        """Publish with the QoS/retain settings from common.topics."""
        self.client.publish(self.prefix + topic, json.dumps(payload),
                            qos=topics.qos_for(topic), retain=topics.retain_for(topic))

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Call handler(topic, payload) for every message on topic."""
        self._handlers.setdefault(topic, []).append(handler)
        if self.client.is_connected():
            self.client.subscribe(self.prefix + topic, qos=topics.qos_for(topic))

    def close(self) -> None:
        """Publish offline and disconnect cleanly."""
        info = self.client.publish(self._status, json.dumps("offline"), qos=1, retain=True)
        try:
            info.wait_for_publish(timeout=1.0)
        except RuntimeError:
            pass  # not connected
        self.client.disconnect()
        self.client.loop_stop()


class FakeBus:
    """In-memory bus for tests: delivers synchronously, records everything."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []
        self.retained: dict[str, Any] = {}
        self._handlers: dict[str, list[Handler]] = {}

    def publish(self, topic: str, payload: Any) -> None:
        payload = json.loads(json.dumps(payload))  # same round trip as the wire
        self.published.append((topic, payload))
        if topics.retain_for(topic):
            self.retained[topic] = payload
        for handler in self._handlers.get(topic, []):
            handler(topic, payload)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers.setdefault(topic, []).append(handler)
        if topic in self.retained:
            handler(topic, self.retained[topic])

    def messages(self, topic: str) -> list[Any]:
        """All payloads published on topic, oldest first."""
        return [p for t, p in self.published if t == topic]

    def clear(self) -> None:
        self.published.clear()
