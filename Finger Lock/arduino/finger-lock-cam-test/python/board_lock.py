"""Finger lock running ON the UNO Q: NumPy CNN + lock policy + vault page (port 8090).

Started from main.py, which owns the camera and the servo. Everything here is
plain Python + numpy + cv2 (no PyTorch).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from finger_infer import FingerNetNumpy
from lock_policy import LockPolicy
from maps_view import render_jpeg

HERE = Path(__file__).parent


def start(get_frame, servo_action, matrix_action=None, port=8090, hz=10):
    """get_frame() -> latest BGR frame or None.  servo_action("lock"|"unlock").
    matrix_action(mode, val) draws on the onboard LED matrix (see sketch.ino)."""
    code_file = HERE / "secret_code.txt"
    if not code_file.exists():
        print("board_lock: NO secret_code.txt next to main.py, lock page disabled")
        return
    code = code_file.read_text().strip()
    net = FingerNetNumpy()
    classes = net.classes

    def act(name):
        threading.Thread(target=servo_action, args=(name,), daemon=True).start()

    policy = LockPolicy(code, on_unlock=lambda: act("unlock"), on_lock=lambda: act("lock"),
                        on_alarm=lambda: print("[lock] ALARM"))
    plock = threading.Lock()
    status = {"label": "-", "conf": 0.0, "probs": {c: 0.0 for c in classes}, "fps": 0.0, "ms": 0.0}
    seen = {"frame": None, "maps": b""}
    shown = {"matrix": None}
    confirm = {"until": 0.0, "n": 0}  # check mark on the matrix right after a digit is accepted
    act("lock")  # start with the latch closed

    def matrix_for(label, conf):
        """What the LED matrix should show right now."""
        if policy.state == "UNLOCKED":
            return (3, 0)
        if policy.state == "DENIED":
            return (4, 0)
        if policy.state == "LOCKDOWN":
            return (5, 0)
        if policy.state == "ARMED":
            if time.time() < confirm["until"]:
                return (3, 0)                       # check mark: that digit is in
            if policy.need_clear:
                return (2, len(policy.entered))     # digit locked, waiting for the hand to leave
            if label in "012345" and len(label) == 1 and conf >= 0.6:
                return (6, int(label))              # live reading of what the camera sees
            return (2, len(policy.entered))
        return (1, 0)

    def infer_loop():
        n, t0, k = 0, time.time(), 0
        while True:
            t_start = time.time()
            frame = get_frame()
            if frame is not None:
                seen["frame"] = frame
                probs, label, conf = net.predict(frame)
                with plock:
                    policy.update(label, conf, time.time())
                    status.update(label=label, conf=conf, ms=(time.time() - t_start) * 1000,
                                  probs={c: float(p) for c, p in zip(classes, probs)})
                k += 1
                if k % 3 == 0:  # feature-map picture at ~3 Hz
                    seen["maps"] = render_jpeg(*net.last)
                n_in = len(policy.entered)
                if policy.state == "ARMED" and n_in > confirm["n"]:
                    confirm["until"] = time.time() + 0.8   # a digit was just accepted
                    print(f"[lock] digit {n_in} accepted")
                confirm["n"] = n_in
                if matrix_action is not None:
                    want = matrix_for(label, conf)
                    if want != shown["matrix"]:
                        shown["matrix"] = want
                        try:
                            matrix_action(*want)
                        except Exception as e:
                            print("matrix failed:", e)
                n += 1
                if time.time() - t0 >= 1:
                    status["fps"], n, t0 = n / (time.time() - t0), 0, time.time()
            time.sleep(max(0.0, 1.0 / hz - (time.time() - t_start)))

    def state_json():
        now = time.time()
        with plock:
            return {
                "state": policy.state, "entered": len(policy.entered), "of": policy.code_length,
                "hold": policy.hold / policy.hold_frames, "candidate": policy.candidate,
                "need_clear": policy.need_clear, "remaining": round(policy.remaining(now), 1),
                "attempts": policy.attempts, "max_attempts": policy.max_attempts,
                "label": status["label"], "conf": status["conf"], "probs": status["probs"],
                "classes": classes, "fps": round(status["fps"], 1), "ms": round(status["ms"], 1),
                "cam_ok": seen["frame"] is not None, "roi": net.roi, "model": "on UNO Q (numpy)",
                "conf_min": policy.conf_min,
                "events": [{"ago": round(now - e["t"], 1), "msg": e["msg"]} for e in policy.events[-12:]],
            }

    page = (HERE / "vault_page.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, ctype, body):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]  # ignore cache-busting ?query
            if path == "/api/state":
                self._send("application/json", json.dumps(state_json()).encode())
            elif path == "/maps.jpg":
                self._send("image/jpeg", seen["maps"])
            elif path == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        frame = seen["frame"]
                        if frame is not None and frame is not last:
                            last = frame
                            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ok:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                                 + jpg.tobytes() + b"\r\n")
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                self._send("text/html; charset=utf-8", page)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            with plock:
                if self.path == "/api/arm":
                    policy.arm(time.time())
                elif self.path == "/api/lock":
                    policy.lock(time.time())
            self._send("application/json", b'{"ok":true}')

    def arm():
        """Same as the ARM button on the page (used by the physical button on D7)."""
        with plock:
            policy.arm(time.time())

    threading.Thread(target=infer_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"VAULT 193 (on board): http://<board-ip>:{port}  classes={classes}")
    return arm
