"""VAULT 193: heist-style live lock page.

Reads the camera, runs the CNN on every frame, feeds the predictions to the lock
policy and shows everything on a web page (live feed, detected number, code entry).

  python src/lock_server.py                               # camera on the UNO Q (default); code from secret_code.txt
  python src/lock_server.py --source cam:1     # a webcam plugged into this computer
  python src/lock_server.py --source demo      # no camera: replays saved images

Then open http://localhost:8090
"""
import argparse
import glob
import json
import os
import random
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lock_policy import LockPolicy
from maps_view import render_jpeg
from model import FingerNet, preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = "http://10.247.137.25:8080"  # the UNO Q app (camera + servo)
BOARD_RAW = BOARD + "/raw"  # clean (no overlay) stream from the UNO Q app

ap = argparse.ArgumentParser()
def _default_code():
    """--code, else FINGER_CODE, else the git-ignored secret_code.txt (never committed)."""
    if os.environ.get("FINGER_CODE"):
        return os.environ["FINGER_CODE"]
    f = os.path.join(ROOT, "secret_code.txt")
    return open(f).read().strip() if os.path.exists(f) else None


ap.add_argument("--code", default=_default_code(),
                help="secret 3-digit code, digits 1-5 (default: FINGER_CODE or secret_code.txt)")
ap.add_argument("--source", default="board", help="board | cam:N | demo | http://.../raw")
ap.add_argument("--model", default=None, help="path to a .pt checkpoint (default: newest)")
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--servo", choices=["auto", "on", "off"], default="auto",
                help="drive the latch servo on the UNO Q (auto = on when --source board)")
ap.add_argument("--hz", type=float, default=12, help="predictions per second")
ap.add_argument("--conf", type=float, default=0.75)
ap.add_argument("--hold", type=int, default=8, help="frames in a row to accept a digit")
args = ap.parse_args()
if not args.code:
    sys.exit("No code set: create secret_code.txt (3 digits, 1-5) or pass --code")

# ---- model --------------------------------------------------------------------
final = os.path.join(ROOT, "models", "finger_cnn_final.pt")  # the model trained on everyone
ckpt_path = args.model or (final if os.path.exists(final)
                           else max(glob.glob(os.path.join(ROOT, "models", "*.pt")), key=os.path.getmtime))
ckpt = torch.load(ckpt_path, map_location="cpu")
classes, roi = ckpt["classes"], ckpt["roi"]
net = FingerNet(n_classes=len(classes))
net.load_state_dict(ckpt["state_dict"])
net.eval()
print(f"model: {os.path.basename(ckpt_path)}  classes: {classes}  roi: {roi}")

# ---- policy -------------------------------------------------------------------
SERVO_ON = args.servo == "on" or (args.servo == "auto" and args.source == "board")


def servo(action):
    """Ask the UNO Q to move the latch to the saved LOCKED/UNLOCKED angle (non-blocking)."""
    def go():
        try:
            req = urllib.request.Request(f"{BOARD}/api/servo/{action}", data=b"{}", method="POST")
            urllib.request.urlopen(req, timeout=4).read()
        except Exception as e:
            print(f"[hardware] servo {action} FAILED: {e}")
    threading.Thread(target=go, daemon=True).start()


def on_unlock():
    print("[hardware] UNLOCK")
    if SERVO_ON:
        servo("unlock")


def on_lock():
    print("[hardware] LOCK")
    if SERVO_ON:
        servo("lock")


def on_alarm():
    print("[hardware] ALARM (latch stays locked)")


policy = LockPolicy(args.code, conf_min=args.conf, hold_frames=args.hold,
                    on_unlock=on_unlock, on_lock=on_lock, on_alarm=on_alarm)
plock = threading.Lock()

# ---- frame sources ------------------------------------------------------------
frame_lock = threading.Lock()
latest = {"frame": None, "id": 0, "ok": False}
maps_store = {"jpg": b""}


def publish(frame):
    with frame_lock:
        latest.update(frame=frame, id=latest["id"] + 1, ok=True)


def reader_stream(url):
    while True:
        cap = cv2.VideoCapture(url)
        got = False
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            got = True
            publish(frame)
        cap.release()
        latest["ok"] = False
        time.sleep(1.0 if got else 2.0)


def reader_demo():
    """Replays saved images so the page and policy can be tried with no camera."""
    raw = os.path.join(ROOT, "data", "raw")
    files = {c: sorted(glob.glob(os.path.join(raw, c, "*.jpg"))) for c in classes}
    files = {c: f for c, f in files.items() if f}
    # a scripted visitor: tries the right code, then a wrong one
    script = list(args.code) + ["empty" if "empty" in files else "0"]
    script_i, t_switch, label = 0, 0.0, "0"
    while True:
        if time.time() > t_switch:
            label = script[script_i % len(script)] if script_i % 2 == 0 else ("empty" if "empty" in files else "0")
            script_i += 1
            t_switch = time.time() + 2.2
        publish(cv2.imread(random.choice(files[label])))
        time.sleep(0.08)


if args.source == "demo":
    threading.Thread(target=reader_demo, daemon=True).start()
else:
    if args.source == "board":
        url = BOARD_RAW
    elif args.source.startswith("cam:"):
        url = int(args.source[4:])
    else:
        url = args.source
    threading.Thread(target=reader_stream, args=(url,), daemon=True).start()

# ---- inference + policy loop --------------------------------------------------
status = {"label": "-", "conf": 0.0, "probs": {c: 0.0 for c in classes}, "fps": 0.0, "ms": 0.0}


def infer_loop():
    last_id, n, t0 = -1, 0, time.time()
    period = 1.0 / args.hz
    while True:
        t_start = time.time()
        with frame_lock:
            frame, fid = latest["frame"], latest["id"]
        if frame is not None and fid != last_id:
            last_id = fid
            x = torch.from_numpy(preprocess(frame, roi))[None, None]
            with torch.no_grad():
                out, (a1, a2, a3) = net(x, return_maps=True)
                p = torch.softmax(out, dim=1)[0].numpy()
            if n % 3 == 0:
                maps_store["jpg"] = render_jpeg(x[0, 0].numpy(), a1[0].numpy(), a2[0].numpy(), a3[0].numpy())
            k = int(p.argmax())
            with plock:
                policy.update(classes[k], float(p[k]), time.time())
                status.update(label=classes[k], conf=float(p[k]),
                              probs={c: float(v) for c, v in zip(classes, p)},
                              ms=(time.time() - t_start) * 1000)
            n += 1
            if time.time() - t0 >= 1:
                status["fps"], n, t0 = n / (time.time() - t0), 0, time.time()
        time.sleep(max(0.0, period - (time.time() - t_start)))


threading.Thread(target=infer_loop, daemon=True).start()


def state_json():
    now = time.time()
    with plock:
        ev = [{"ago": round(now - e["t"], 1), "msg": e["msg"]} for e in policy.events[-12:]]
        return {
            "state": policy.state, "entered": len(policy.entered), "of": policy.code_length,
            "hold": policy.hold / policy.hold_frames, "candidate": policy.candidate,
            "need_clear": policy.need_clear, "remaining": round(policy.remaining(now), 1),
            "attempts": policy.attempts, "max_attempts": policy.max_attempts,
            "label": status["label"], "conf": status["conf"], "probs": status["probs"],
            "classes": classes, "fps": round(status["fps"], 1), "ms": round(status["ms"], 1),
            "cam_ok": latest["ok"], "roi": roi, "model": os.path.basename(ckpt_path),
            "conf_min": policy.conf_min, "events": ev,
        }


# ---- web page -----------------------------------------------------------------
PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_page.html"),
            encoding="utf-8").read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]  # ignore cache-busting ?query
        if path == "/api/state":
            self._send(200, "application/json", json.dumps(state_json()).encode())
        elif path == "/maps.jpg":
            self._send(200, "image/jpeg", maps_store["jpg"])
        elif path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    with frame_lock:
                        frame, fid = latest["frame"], latest["id"]
                    if frame is not None and fid != last:
                        last = fid
                        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                             + jpg.tobytes() + b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        with plock:
            if self.path == "/api/arm":
                policy.arm(time.time())
            elif self.path == "/api/lock":
                policy.lock(time.time())
        self._send(200, "application/json", b'{"ok":true}')


if SERVO_ON:
    servo("lock")  # start in a known state: latch closed
print(f"servo: {'on' if SERVO_ON else 'off'}")
print(f"VAULT 193 online: http://localhost:{args.port}   (source: {args.source}, code hidden)")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
