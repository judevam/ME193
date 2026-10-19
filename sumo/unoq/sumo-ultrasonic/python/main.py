"""Sumo robot sensor node on the UNO Q.

- Reads the HC-SR04 (via the sketch) and publishes sumo/<ROBOT_ID>/sensor/ultrasonic
  {"cm": .., "t": ..} at RATE_HZ. No echo -> nothing published (the brain treats
  silence as "no opponent").
- Plays the sad song on the buzzer when the brain publishes cmd/sound.
- Reports online/offline on status/unoq (Last Will).

The laptop runs the broker and the brain: see sumo/README.md in the ME193 repo.
"""
import json
import threading
import time
from collections import deque

import paho.mqtt.client as mqtt
from arduino.app_utils import *

BROKER_HOST = "10.247.137.66"  # the LAPTOP's Wi-Fi IP (ipconfig). Changes between networks!
BROKER_PORT = 1883
ROBOT_ID = "bot1"              # must match robot_id in sumo/config/sumo.yaml
RATE_HZ = 20
USE_BUZZER = True

# Copy of SAD in sumo/common/sounds.py: (frequency Hz, duration ms), 0 Hz = rest.
SAD = [
    (392, 450), (0, 80),
    (370, 450), (0, 80),
    (349, 450), (0, 80),
    (330, 300), (311, 300), (330, 300), (311, 300), (330, 900),
]
SONGS = {"sad": SAD}

PREFIX = f"sumo/{ROBOT_ID}/"
STATUS = PREFIX + "status/unoq"
notes = deque()          # notes still to play
notes_lock = threading.Lock()


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"MQTT connected to {BROKER_HOST}:{BROKER_PORT} (rc={reason_code})")
    client.publish(STATUS, json.dumps("online"), qos=1, retain=True)
    client.subscribe(PREFIX + "cmd/sound", qos=1)


def on_message(client, userdata, msg):
    try:
        song = json.loads(msg.payload).get("song")
    except (ValueError, AttributeError):
        return
    if USE_BUZZER and song in SONGS:
        print(f"playing {song!r}")
        with notes_lock:
            notes.clear()
            notes.extend(SONGS[song])


def sensor_loop(client):
    """Read the sensor and step the song, both from this one thread."""
    period = 1.0 / RATE_HZ
    note_end = 0.0
    while True:
        start = time.monotonic()
        try:
            cm = Bridge.call("read_cm")
            if cm is not None and cm > 0:
                client.publish(PREFIX + "sensor/ultrasonic",
                               json.dumps({"cm": round(cm, 1), "t": time.time()}))
            with notes_lock:
                due = notes.popleft() if notes and start >= note_end else None
            if due:
                freq, ms = due
                Bridge.call("play_tone", freq, ms)
                note_end = start + ms / 1000
        except Exception as e:
            print(f"sensor loop error: {e!r}")
            time.sleep(1)
        time.sleep(max(0.0, period - (time.monotonic() - start)))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sumo-{ROBOT_ID}-unoq")
client.will_set(STATUS, json.dumps("offline"), qos=1, retain=True)
client.on_connect = on_connect
client.on_message = on_message
# Keep retrying until the laptop's broker is reachable; after that paho reconnects by itself.
while True:
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=5)
        break
    except OSError as e:
        print(f"MQTT broker {BROKER_HOST}:{BROKER_PORT} not reachable ({e}), retrying in 5 s")
        time.sleep(5)
client.loop_start()

threading.Thread(target=sensor_loop, args=(client,), daemon=True).start()
print(f"Sumo ultrasonic: publishing to {PREFIX}sensor/ultrasonic at {RATE_HZ} Hz")

App.run()
