"""Gait wearable on the UNO Q: record labeled IMU sessions, controlled over MQTT.

- Pulls MPU-6050 samples from the sketch (100 Hz, nothing lost thanks to the IMU's FIFO).
- On {"cmd": "start", "label": "walking"} it records to data/<label>_<time>.csv on the
  board; {"cmd": "stop"} closes the file. Only one recording at a time.
- Publishes a status message about once a second on TOPIC (see below), and
  online/offline on TOPIC/status (Last Will).

Laptop side: "Gait Wearable/gait.py" in the ME193 repo sends the commands and shows the
status; "Gait Wearable/pull_data.py" copies the recordings off the board.

Topics (TOPIC = ME193/Jude/gait):
  TOPIC/cmd     laptop -> board   {"cmd": "start", "label": "running"} | {"cmd": "stop"}
                                  | {"cmd": "show", "gait": "walking", "seconds": 5}  (LED matrix)
  TOPIC         board -> laptop   {"mode": "idle"|"record", "label", "seconds", "samples",
                                   "lost", "file", "peak_axis_g", "mag_g", "rate_hz", "t",
                                   with models also "gait", "model1", "model2", "confidence", "peak_g"}
  TOPIC/status  board -> laptop   "online" / "offline" (retained)
  TOPIC/imu     board -> laptop   {"seq": n, "rate_hz": 100, "s": [[ax, ay, az, gx, gy, gz], ...]}
                                  every sample, in batches every STREAM_S (for dashboard.py)
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from arduino.app_utils import *

import matrix_art

# Trained models, copied here by "Gait Wearable/push_models.py" in ME193 (with
# gait_models.py and features_core.py). Without them the board just records and streams.
MODELS_PATH = Path(__file__).resolve().parent / "gait_models.json"
try:
    from gait_models import GaitModels, StreamClassifier
    classifier = StreamClassifier(GaitModels.load(MODELS_PATH)) if MODELS_PATH.exists() else None
except Exception as e:  # missing numpy or a broken file: keep recording anyway
    print(f"couldn't load the gait models: {e!r}")
    classifier = None
print("gait models:", "loaded" if classifier else "none (record and stream only)")

BROKER_HOST = "broker.hivemq.com"   # public broker (was test.mosquitto.org, down 2026-10-25); must match config.py
BROKER_PORT = 1883
TOPIC = "ME193/Jude/gait"
RATE_HZ = 100                       # must match RATE_HZ in the sketch
LABELS = {"standing", "walking", "running", "biking"}
MAX_RECORD_S = 600                  # stop a forgotten recording after 10 minutes
SYNC_S = 1.0                        # write recordings to disk at least this often
POLL_S = 0.05
STREAM = True                       # send every sample on TOPIC/imu for dashboard.py
STREAM_S = 0.2                      # ... in batches this often (~20 samples each)
SHOW_S = 5                          # a "show" command lasts this long unless repeated
PEAK_AXIS_INDEX = 0                 # thigh axis for the status message: 0=X 1=Y 2=Z
DATA_DIR = Path(__file__).resolve().parent / "data"


ACCEL_COUNTS_PER_G = 4096     # sketch uses ACCEL_8G
GYRO_COUNTS_PER_DPS = 32.8    # sketch uses GYRO_1000DPS


def parse(reply):
    """'0|ax,ay,az,gx,gy,gz;...' in raw counts -> (overflow, [[g, g, g, dps, dps, dps], ...])"""
    flag, _, body = reply.partition("|")
    samples = []
    for s in body.split(";"):
        if s:
            ax, ay, az, gx, gy, gz = (int(v) for v in s.split(","))
            samples.append([round(ax / ACCEL_COUNTS_PER_G, 4), round(ay / ACCEL_COUNTS_PER_G, 4),
                            round(az / ACCEL_COUNTS_PER_G, 4), round(gx / GYRO_COUNTS_PER_DPS, 2),
                            round(gy / GYRO_COUNTS_PER_DPS, 2), round(gz / GYRO_COUNTS_PER_DPS, 2)])
    return flag == "1", samples


class Recorder:
    """Writes samples to one CSV at a time. Called from the sampler thread and from MQTT
    callbacks, so everything goes through the lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.file = None
        self.path = None
        self.label = None
        self.samples = 0
        self.lost = 0
        self.started = 0.0
        self.synced = 0.0

    def start(self, label):
        with self.lock:
            self._close()
            DATA_DIR.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path = DATA_DIR / f"{label}_{stamp}.csv"
            self.file = open(self.path, "w")
            self.file.write(f"# label={label} start={stamp} rate_hz={RATE_HZ}\n")
            self.file.write("t,ax,ay,az,gx,gy,gz\n")
            self.label, self.samples, self.lost = label, 0, 0
            self.started = self.synced = time.monotonic()
            print(f"recording {label} -> {self.path.name}")

    def stop(self):
        with self.lock:
            self._close()

    def add(self, samples, overflow):
        with self.lock:
            if not self.file:
                return
            if overflow:
                self.lost += 1  # FIFO overflowed at least once: there's a gap here
            for s in samples:
                # Time from the sample count: the IMU's clock is steadier than ours.
                t = self.samples / RATE_HZ
                self.file.write(f"{t:.2f}," + ",".join(f"{v:g}" for v in s) + "\n")
                self.samples += 1
            now = time.monotonic()
            if now >= self.synced + SYNC_S:
                # Push it all the way to the SD card, so a power cut loses at most SYNC_S
                self.file.flush()
                os.fsync(self.file.fileno())
                self.synced = now
            if now - self.started > MAX_RECORD_S:
                print("recording hit MAX_RECORD_S, stopping")
                self._close()

    def status(self):
        with self.lock:
            if not self.file:
                return {"mode": "idle"}
            return {"mode": "record", "label": self.label, "file": self.path.name,
                    "seconds": round(self.samples / RATE_HZ, 1),
                    "samples": self.samples, "lost": self.lost}

    def _close(self):
        if self.file:
            self.file.close()
            print(f"saved {self.path.name}: {self.samples} samples, "
                  f"{self.samples / RATE_HZ:.1f} s, overflows: {self.lost}")
            self.file = None


recorder = Recorder()


class Matrix:
    """Which animation the LED matrix plays: the activity being recorded, else a gait the
    laptop sent with "show" (for its seconds), else the board's own classification, else idle."""

    def __init__(self):
        self.lock = threading.Lock()
        self.gait = None
        self.until = 0.0
        self.own = None          # the board's own (smoothed) answer, if it has models
        self.frame_i = 0
        self.next_frame = 0.0
        self.playing = "unset"
        self.last = None

    def show(self, gait, seconds=SHOW_S):
        with self.lock:
            self.gait = gait if gait in matrix_art.ANIMATIONS else None
            self.until = time.monotonic() + float(seconds)

    def step(self, now):
        """Draw the next frame if it's due. Called from the sampler thread only."""
        if now < self.next_frame:
            return
        rec = recorder.status()
        recording = rec["mode"] == "record"
        with self.lock:
            if recording:
                gait = rec["label"]
            elif now < self.until:
                gait = self.gait
            else:
                gait = self.own
        if gait != self.playing:
            self.playing, self.frame_i = gait, 0
        frames, period = matrix_art.animation(gait)
        frame = frames[self.frame_i % len(frames)]
        # REC dot blinks once a second while recording
        encoded = matrix_art.encode(frame, rec_dot=recording and int(now * 2) % 2 == 0)
        if encoded != self.last:
            Bridge.call("draw", encoded)
            self.last = encoded
        self.frame_i += 1
        self.next_frame = now + min(period, 0.5)


matrix = Matrix()


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"MQTT connected to {BROKER_HOST} (rc={reason_code})")
    client.publish(TOPIC + "/status", "online", qos=1, retain=True)
    client.subscribe(TOPIC + "/cmd", qos=1)


def on_message(client, userdata, msg):
    try:
        cmd = json.loads(msg.payload)
        name = cmd.get("cmd")
    except (ValueError, AttributeError):
        print(f"ignoring bad command: {msg.payload!r}")
        return
    if name == "start":
        label = cmd.get("label")
        if label in LABELS:
            recorder.start(label)
        else:
            print(f"unknown label {label!r}; use one of {sorted(LABELS)}")
    elif name == "stop":
        recorder.stop()
    elif name == "show":
        matrix.show(cmd.get("gait"), cmd.get("seconds", SHOW_S))
        return
    publish_status(client, {})


def publish_status(client, extra):
    msg = {**recorder.status(), **extra, "t": time.time()}
    client.publish(TOPIC, json.dumps(msg))


def sampler_loop(client):
    # The sketch may still be starting up: keep asking instead of letting this thread die
    # (which would leave the app running but reading nothing).
    while True:
        try:
            print("IMU:", Bridge.call("imu_status"))
            print("IMU registers:", Bridge.call("imu_debug"))
            break
        except Exception as e:
            print(f"sketch not answering yet ({e!r}), retrying in 2 s")
            time.sleep(2)
    window = []  # samples from the last second, for the status message
    stream = []  # samples not yet sent on TOPIC/imu
    seq = 0      # index of stream[0] since start, so the laptop can spot gaps
    count = 0
    report_at = time.monotonic() + 1
    stream_at = time.monotonic() + STREAM_S
    while True:
        try:
            overflow, samples = parse(Bridge.call("read_samples"))
        except Exception as e:
            print(f"read error: {e!r}")
            time.sleep(1)
            continue
        recorder.add(samples, overflow)
        window.extend(samples)
        count += len(samples)
        now = time.monotonic()
        if classifier:
            if overflow:
                classifier.reset()  # samples were lost: don't classify across the gap
            try:
                for _, result in classifier.add(samples):
                    with matrix.lock:
                        matrix.own = result.get("gait")
            except Exception as e:
                print(f"classify error: {e!r}")
        try:
            matrix.step(now)
        except Exception as e:
            print(f"matrix error: {e!r}")
        if STREAM:
            stream.extend(samples)
            if now >= stream_at and stream:
                client.publish(TOPIC + "/imu", json.dumps({"seq": seq, "rate_hz": RATE_HZ, "s": stream},
                                                         separators=(",", ":")))
                seq += len(stream)
                stream = []
                stream_at = now + STREAM_S
        if now >= report_at:
            extra = {"rate_hz": count}
            if window:
                # peak along the thigh axis and the mean size of the acceleration
                extra["peak_axis_g"] = round(max(s[PEAK_AXIS_INDEX] for s in window), 2)
                extra["mag_g"] = round(sum(math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)
                                           for s in window) / len(window), 2)
                ax, ay, az, gx, gy, gz = window[-1]
                r = classifier.current if classifier else {}
                if r.get("gait"):
                    extra.update(gait=r["gait"], model1=r["model1"], model2=r["model2"],
                                 confidence=None if r["confidence"] is None else round(r["confidence"], 2),
                                 peak_g=round(r["peak"], 2))
                print(f"{count:3d} Hz  accel g [{ax:6.2f} {ay:6.2f} {az:6.2f}] |a|={extra['mag_g']:.2f}  "
                      f"gyro dps [{gx:7.1f} {gy:7.1f} {gz:7.1f}]  {recorder.status()['mode']}"
                      + (f"  gait: {r['gait']} (m1 {r['model1']}, m2 {r['model2']})" if r.get("gait") else ""))
            else:
                try:
                    extra["imu"] = Bridge.call("imu_status")  # why: e.g. still starting / no MPU
                except Exception as e:
                    extra["imu"] = f"sketch not answering ({e!r})"
                print(f"no IMU samples: {extra['imu']}")
            publish_status(client, extra)
            window, count = [], 0
            report_at = max(report_at + 1, now)
        time.sleep(POLL_S)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.will_set(TOPIC + "/status", "offline", qos=1, retain=True)
client.on_connect = on_connect
client.on_message = on_message
# Connect in the background: the IMU and recording work even while the broker is
# unreachable, and paho keeps retrying by itself.
client.on_disconnect = lambda c, u, flags, rc, props=None: print(f"MQTT disconnected (rc={rc}), retrying")
client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=10)
client.loop_start()

threading.Thread(target=sampler_loop, args=(client,), daemon=True).start()
print(f"Gait wearable: commands on {TOPIC}/cmd, status on {TOPIC}")

App.run()
