# Runs on the UNO Q's Linux side. Subscribes to the laptop's detections over
# MQTT, decides how to drive so the minifig ends up centered in the camera
# view, and sends drive(left, right) to the sketch (Maker Drive PWM).
import json
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

MQTT_HOST = "192.168.1.100"  # <-- CHANGE to your laptop's IP (run ipconfig on the laptop)
MQTT_PORT = 1883
TOPIC = "minifig/pos"
TARGET_COLOR = "blue"  # ignore detections of any other color

# --- How the camera and car are set up --------------------------------------
# "drive": the camera is fixed and watches the car, which has the minifig on
#          it. Both wheels go the same way (forward/back) until the minifig
#          is centered in the image.
# "turn":  the camera is mounted on the car and the minifig is somewhere in
#          front of it. The car spins in place until the minifig is centered.
MODE = "drive"
# If the car moves AWAY from center, flip this (or flip --flip on the laptop).
INVERT = False

# --- Tuning -------------------------------------------------------------------
# Error is normalized: 0 = minifig at the center of the image, +/-1 = at the edge.
STOP_ZONE = 0.04   # centered once within this of the middle
START_ZONE = 0.09  # once stopped, only restart when farther out than this
MIN_SPEED = 30     # percent; lowest duty that still gets the wheels turning
MAX_SPEED = 60     # percent; speed when the minifig is far from center
GAIN = 100         # percent of speed per unit of error (before clamping)
MAX_STEP = 15      # max change in commanded speed per message, so it glides

# Set True once, with the wheels OFF THE GROUND, to check each motor spins
# forward (left, then right). Fix wrong ones with INVERT_LEFT/RIGHT in the sketch.
SELF_TEST = False

centered = True
last_speed = 0.0
sent_zero = False


def send(left, right):
    """Always re-send while moving (the sketch stops if commands go quiet);
    send a stop only once."""
    global sent_zero
    if left == 0 and right == 0:
        if not sent_zero:
            Bridge.notify("drive", 0, 0)
            sent_zero = True
    else:
        Bridge.notify("drive", int(left), int(right))
        sent_zero = False


def target_speed(error):
    """Signed speed for a normalized error (+ = minifig right of center)."""
    global centered
    size = abs(error)
    if size <= STOP_ZONE or (centered and size <= START_ZONE):
        centered = True
        return 0.0
    centered = False
    magnitude = min(MAX_SPEED, max(MIN_SPEED, size * GAIN))
    return magnitude if error > 0 else -magnitude


def on_message(client, userdata, msg):
    global last_speed
    try:
        payload = json.loads(msg.payload)
    except ValueError:
        return
    if payload.get("color", TARGET_COLOR) != TARGET_COLOR:
        return

    if not payload.get("found"):
        last_speed = 0.0
        send(0, 0)  # never drive blind
        return

    error = (payload["x"] - 0.5) * 2
    desired = target_speed(error)
    if INVERT:
        desired = -desired
    step = max(-MAX_STEP, min(MAX_STEP, desired - last_speed))
    last_speed += step
    speed = round(last_speed)

    if MODE == "turn":
        send(speed, -speed)
    else:
        send(speed, speed)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to {MQTT_HOST}:{MQTT_PORT} (rc={reason_code})")
    client.subscribe(TOPIC)


def self_test():
    print("SELF_TEST: left motor forward, then right motor forward (wheels off the ground!)")
    time.sleep(3)
    for left, right in ((40, 0), (0, 40)):
        Bridge.notify("drive", left, right)
        time.sleep(1.0)
        Bridge.notify("drive", 0, 0)
        time.sleep(0.5)
    print("SELF_TEST done")


if SELF_TEST:
    self_test()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)  # keeps retrying if broker isn't up yet
client.loop_start()

print(f"Minifig Drive: following the {TARGET_COLOR} minifig on '{TOPIC}' (mode={MODE})")
App.run()
