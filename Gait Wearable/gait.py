"""Control the wearable's recordings and watch its status over MQTT.

  python gait.py watch                     # print what the wearable reports
  python gait.py record walking 90         # 5 s to get going, then record 90 s, then stop
  python gait.py record running 60 --delay 10
  python gait.py start biking              # start / stop by hand
  python gait.py stop
  python gait.py show biking               # play an animation on the LED matrix for 10 s
  python gait.py demo                      # cycle through all animations (Ctrl+C to stop)
  python gait.py demo --seconds 3 --loops 2

Ctrl+C during `record` stops the recording on the board too.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import paho.mqtt.client as mqtt

import config


def connect(on_message=None, imu: bool = False) -> mqtt.Client:
    """Connect and subscribe to the status topics (and the raw sample stream if imu)."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    connected = threading.Event()

    def on_connect(c, userdata, flags, reason_code, properties=None):
        topics = [(config.TOPIC, 0), (config.TOPIC + "/status", 1)]
        if imu:
            topics.append((config.TOPIC + "/imu", 0))
        c.subscribe(topics)
        connected.set()

    client.on_connect = on_connect
    if on_message:
        client.on_message = on_message
    client.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=10)
    client.loop_start()
    if not connected.wait(10):
        sys.exit(f"couldn't connect to {config.BROKER_HOST}")
    return client


def send(client: mqtt.Client, cmd: dict) -> None:
    client.publish(config.TOPIC + "/cmd", json.dumps(cmd), qos=1).wait_for_publish(5)


def format_status(topic: str, payload: bytes) -> str:
    if topic.endswith("/status"):
        return f"wearable is {payload.decode()}"
    try:
        s = json.loads(payload)
    except ValueError:
        return payload.decode(errors="replace")
    # Replies to a command arrive straight away, without the once-a-second live numbers.
    live = (f"{s['rate_hz']:>3} Hz  peak x {s.get('peak_x_g', '?')} g  |a| {s.get('mag_g', '?')} g"
            if "rate_hz" in s else "")
    if s.get("mode") == "record":
        lost = f"  LOST {s['lost']}" if s.get("lost") else ""
        return f"REC {s['label']:8s} {s['seconds']:6.1f} s  {live}{lost}"
    return f"idle              {live}"


def print_messages(client, userdata, msg) -> None:
    print(format_status(msg.topic, msg.payload), flush=True)


def record(client: mqtt.Client, label: str, seconds: float, delay: float) -> None:
    try:
        for left in range(int(delay), 0, -1):
            print(f"start {label} in {left}...", flush=True)
            time.sleep(1)
        send(client, {"cmd": "start", "label": label})
        print(f"recording {label} for {seconds:g} s")
        time.sleep(seconds)
    except KeyboardInterrupt:
        print("\ncancelled")
    finally:
        send(client, {"cmd": "stop"})
        print("stopped")
        time.sleep(1.5)  # show the board's last status
        client.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("watch")
    s = sub.add_parser("start")
    s.add_argument("label", choices=config.LABELS)
    sub.add_parser("stop")
    sh = sub.add_parser("show", help="play an activity's animation on the LED matrix")
    sh.add_argument("label", choices=config.LABELS + ["idle"])
    sh.add_argument("--seconds", type=float, default=10)
    d = sub.add_parser("demo", help="cycle through every animation on the LED matrix")
    d.add_argument("--seconds", type=float, default=5, help="per animation")
    d.add_argument("--loops", type=int, default=0, help="how many times round (0 = until Ctrl+C)")
    r = sub.add_parser("record")
    r.add_argument("label", choices=config.LABELS)
    r.add_argument("seconds", type=float)
    r.add_argument("--delay", type=float, default=5, help="seconds to get going before recording starts")
    args = p.parse_args()

    if args.action == "show":
        client = connect()
        gait = None if args.label == "idle" else args.label
        send(client, {"cmd": "show", "gait": gait, "seconds": args.seconds})
        print(f"showing {args.label} for {args.seconds:g} s")
        client.disconnect()
        return
    if args.action == "demo":
        client = connect()
        n = 0
        try:
            while args.loops == 0 or n < args.loops:
                for label in config.LABELS:
                    print(f"showing {label}", flush=True)
                    send(client, {"cmd": "show", "gait": label, "seconds": args.seconds + 1})
                    time.sleep(args.seconds)
                n += 1
        except KeyboardInterrupt:
            pass
        send(client, {"cmd": "show", "gait": None, "seconds": 0})  # back to idle
        client.disconnect()
        return
    if args.action in ("start", "stop"):
        client = connect()
        send(client, {"cmd": args.action, "label": getattr(args, "label", None)})
        print(f"sent {args.action}")
        client.disconnect()
        return

    client = connect(print_messages)
    if args.action == "record":
        record(client, args.label, args.seconds, args.delay)
        return
    print(f"watching {config.TOPIC} (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.disconnect()


if __name__ == "__main__":
    main()
