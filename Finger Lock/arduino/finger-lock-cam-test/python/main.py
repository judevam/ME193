"""Finger Lock: webcam live view + data collection page for the UNO Q.

Open  http://<board-ip>:8080  in a browser on the same network.
  - Pick a label (0 = fist, 1-5 = fingers, empty = no hand), type your name, press
    "Start taking pictures": after a 3 s countdown it saves a burst of frames.
  - Full camera frames are saved (no overlay) to data/<label>/. The red box on
    the live view is only a guide; its position is saved in data/roi.json so the
    crop can be applied later, at training time.
  - "Download all" gives a zip to copy the dataset to your computer.
"""
import io
import json
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from arduino.app_utils import *

PORT = 8080
WIDTH, HEIGHT = 640, 480
DATA = Path(__file__).parent / "data"
LABELS = ["0", "1", "2", "3", "4", "5", "empty"]  # 0 = fist, empty = no hand
for lab in LABELS:
    (DATA / lab).mkdir(parents=True, exist_ok=True)
ROI_FILE = DATA / "roi.json"

def open_camera():
    """The webcam's /dev/video number changes between boots, so try each one."""
    for i in range(8):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            ok, _ = c.read()
            if ok:
                print(f"Using camera index {i}")
                return c
        c.release()
    print("No working camera found. Check the USB cable/hub.")
    return cv2.VideoCapture()


cap = open_camera()

SERVO_FILE = DATA / "servo.json"
try:
    servo_cfg = json.loads(SERVO_FILE.read_text())
except Exception:
    servo_cfg = {"locked": 0, "unlocked": 90, "current": None}


def move_servo(angle):
    angle = max(0, min(180, int(angle)))
    servo_cfg["current"] = angle
    try:
        Bridge.call("servo", angle)
    except Exception as e:
        print("servo call failed:", e)
    return angle


lock = threading.Lock()
latest = None
fps_now = 0.0

try:
    roi = json.loads(ROI_FILE.read_text())
except Exception:
    roi = {"x": 170, "y": 90, "size": 300}  # pixels in the full frame

# capture/burst state
cap_state = {"active": False, "phase": "idle", "label": "", "person": "", "taken": 0,
             "target": 0, "countdown": 0.0, "stop": False, "last_burst": []}


def grab_loop():
    global latest, fps_now
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        n += 1
        if time.time() - t0 >= 1:
            fps_now, n, t0 = n / (time.time() - t0), 0, time.time()
        with lock:
            latest = frame


def burst(label, person, count, interval):
    st = cap_state
    st.update(active=True, stop=False, label=label, person=person, taken=0,
              target=count, phase="countdown", last_burst=[])
    end = time.time() + 3.0
    while time.time() < end and not st["stop"]:
        st["countdown"] = end - time.time()
        time.sleep(0.05)
    st["phase"] = "recording"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = []
    while st["taken"] < count and not st["stop"]:
        with lock:
            frame = None if latest is None else latest.copy()
        if frame is not None:
            p = DATA / label / f"{person}_{stamp}_{st['taken']:03d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(p))
            st["taken"] += 1
        time.sleep(interval)
    st.update(last_burst=paths, active=False, phase="idle")


def counts():
    out = {}
    for lab in LABELS:
        for f in (DATA / lab).glob("*.jpg"):
            person = f.name.split("_")[0]
            out.setdefault(person, {l: 0 for l in LABELS})[lab] += 1
    return out


def overlay(frame):
    h, w = frame.shape[:2]
    x, y, s = roi["x"], roi["y"], roi["size"]
    cv2.rectangle(frame, (x, y), (x + s, y + s), (0, 0, 255), 2)
    cv2.putText(frame, f"{w}x{h} {fps_now:.0f} FPS", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    st = cap_state
    if st["phase"] == "countdown":
        cv2.putText(frame, f"GET READY: show {st['label']}  {st['countdown']:.1f}", (8, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif st["phase"] == "recording":
        cv2.circle(frame, (w - 24, 24), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC {st['label']}  {st['taken']}/{st['target']}", (8, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


PAGE = """<!doctype html><meta charset=utf-8><title>Finger Lock capture</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:16px}
.row{display:flex;gap:24px;flex-wrap:wrap}
img{max-width:100%;border:1px solid #444}
button,input{font-size:16px;padding:6px 10px;margin:3px}
.lab{display:inline-block;min-width:42px;text-align:center;border:2px solid #555;border-radius:6px;padding:8px 6px;margin:3px;cursor:pointer;font-size:20px}
.lab.sel{background:#2a7;border-color:#2a7;color:#000}
#go{background:#2a7;color:#000;border:0;border-radius:6px;font-weight:bold}
#stop{background:#c33;color:#fff;border:0;border-radius:6px}
table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 8px;text-align:center}
</style>
<h2>Finger Lock: capture</h2>
<div class=row>
 <div><img src="/stream" width=640></div>
 <div style="min-width:340px">
  <div>Label: <span id=labs></span></div>
  <div>Name: <input id=person value="jude" size=10>
   Pictures: <input id=count type=number value=40 min=1 max=500 style="width:70px"></div>
  <div>Seconds between pictures: <input id=interval type=number value=0.3 step=0.1 min=0.05 style="width:70px"></div>
  <div><button id=go onclick=start()>Start taking pictures</button>
   <button id=stop onclick=stopCap()>Stop</button>
   <button onclick=undo()>Delete last burst</button></div>
  <p id=msg style="font-size:18px">idle</p>
  <p>Keys: 0-5 pick label, e = empty, Space = start. (0 = fist, empty = no hand in the box)</p>
  <h4>Red box (crop area)</h4>
  <div>x <input id=rx type=range min=0 max=640 oninput=roi()> <span id=vx></span></div>
  <div>y <input id=ry type=range min=0 max=480 oninput=roi()> <span id=vy></span></div>
  <div>size <input id=rs type=range min=100 max=480 oninput=roi()> <span id=vs></span></div>
  <h4>Servo latch (D2)</h4>
  <div>angle <input id=sa type=range min=0 max=180 value=90 oninput=servoMove(this.value) style="width:220px"> <b id=sv>90</b></div>
  <div><button onclick="servoNudge(-5)">-5</button><button onclick="servoNudge(5)">+5</button>
   <button onclick="servoSave('locked')">Set as LOCKED</button>
   <button onclick="servoSave('unlocked')">Set as UNLOCKED</button></div>
  <div><button onclick="post('/api/servo/lock')">Test LOCK</button>
   <button onclick="post('/api/servo/unlock')">Test UNLOCK</button></div>
  <div id=scfg style="color:#9bd"></div>
  <p><a style="color:#8cf" href="/download.zip">Download all (zip)</a></p>
 </div>
</div>
<h4>Pictures saved</h4><div id=tbl></div>
<script>
const L=["0","1","2","3","4","5","empty"];
let label="1", inited=false;
const labs=document.getElementById('labs');
for(const l of L){const d=document.createElement('span');d.className='lab';d.textContent=l;d.onclick=()=>pick(l);d.id='l'+l;labs.appendChild(d)}
function pick(l){label=l;for(const x of L)document.getElementById('l'+x).classList.toggle('sel',x==l)}
pick("1");
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(L.includes(e.key))pick(e.key);if(e.key==="e")pick("empty");if(e.code==='Space'){e.preventDefault();start()}});
const post=(u,b)=>fetch(u,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json());
function start(){post('/api/start',{label:label,person:document.getElementById('person').value.trim().replace(/[^A-Za-z0-9-]/g,'')||'anon',count:+document.getElementById('count').value,interval:+document.getElementById('interval').value})}
function stopCap(){post('/api/stop')}
function undo(){post('/api/undo')}
let sT=null;
function servoMove(v){sv.textContent=v;clearTimeout(sT);sT=setTimeout(()=>post('/api/servo',{angle:+v}),120)}
function servoNudge(d){sa.value=Math.max(0,Math.min(180,+sa.value+d));servoMove(sa.value)}
function servoSave(w){post('/api/servo/save',{which:w,angle:+sa.value})}
function roi(){post('/api/roi',{x:+rx.value,y:+ry.value,size:+rs.value})}
async function poll(){
 try{const s=await (await fetch('/api/status')).json();
  document.getElementById('msg').textContent=s.phase=='idle'?'idle':(s.phase=='countdown'?'get ready... show '+s.label:'recording '+s.label+': '+s.taken+'/'+s.target);
  if(!inited){rx.value=s.roi.x;ry.value=s.roi.y;rs.value=s.roi.size;if(s.servo.current!==null){sa.value=s.servo.current;sv.textContent=s.servo.current}inited=true}
  scfg.textContent='saved: LOCKED='+s.servo.locked+' deg, UNLOCKED='+s.servo.unlocked+' deg';
  vx.textContent=rx.value;vy.textContent=ry.value;vs.textContent=rs.value;
  let h='<table><tr><th>person</th>'+L.map(l=>'<th>'+l+'</th>').join('')+'<th>total</th></tr>';
  let tot=L.map(()=>0);
  for(const [p,c] of Object.entries(s.counts)){let t=0;h+='<tr><td>'+p+'</td>';L.forEach((l,i)=>{h+='<td>'+c[l]+'</td>';t+=c[l];tot[i]+=c[l]});h+='<td>'+t+'</td></tr>'}
  h+='<tr><th>all</th>'+tot.map(t=>'<th>'+t+'</th>').join('')+'<th>'+tot.reduce((a,b)=>a+b,0)+'</th></tr></table>';
  document.getElementById('tbl').innerHTML=h}catch(e){}
 setTimeout(poll,500)}
poll();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/stream", "/raw"):
            # /stream has the red box and text drawn on it; /raw is the clean
            # camera image (what the lock server feeds to the model)
            clean = self.path == "/raw"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        frame = None if latest is None else latest.copy()
                    if frame is not None:
                        ok, jpg = cv2.imencode(".jpg", frame if clean else overlay(frame),
                                               [cv2.IMWRITE_JPEG_QUALITY, 90 if clean else 80])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                             + jpg.tobytes() + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/api/status":
            self._json({**{k: cap_state[k] for k in ("active", "phase", "label", "person", "taken", "target")},
                        "roi": roi, "servo": servo_cfg, "counts": counts()})
        elif self.path == "/download.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for f in DATA.rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(DATA).as_posix())
            self._send(200, "application/zip", buf.getvalue())
        else:
            self._send(200, "text/html", PAGE.encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/start":
            if cap_state["active"] or body.get("label") not in LABELS:
                return self._json({"ok": False})
            threading.Thread(target=burst, daemon=True, args=(
                body["label"], str(body.get("person", "anon"))[:20] or "anon",
                max(1, min(500, int(body.get("count", 40)))),
                max(0.05, float(body.get("interval", 0.3))))).start()
            self._json({"ok": True})
        elif self.path == "/api/stop":
            cap_state["stop"] = True
            self._json({"ok": True})
        elif self.path == "/api/undo":
            if not cap_state["active"]:
                for p in cap_state["last_burst"]:
                    Path(p).unlink(missing_ok=True)
                cap_state["last_burst"] = []
            self._json({"ok": True})
        elif self.path == "/api/servo":
            self._json({"ok": True, "angle": move_servo(body.get("angle", 90))})
        elif self.path == "/api/servo/save":
            which = body.get("which")
            if which in ("locked", "unlocked"):
                servo_cfg[which] = max(0, min(180, int(body.get("angle", 90))))
                SERVO_FILE.write_text(json.dumps(servo_cfg))
            self._json({"ok": which in ("locked", "unlocked")})
        elif self.path == "/api/servo/lock":
            self._json({"ok": True, "angle": move_servo(servo_cfg["locked"])})
        elif self.path == "/api/servo/unlock":
            self._json({"ok": True, "angle": move_servo(servo_cfg["unlocked"])})
        elif self.path == "/api/roi":
            s = max(100, min(HEIGHT, int(body.get("size", roi["size"]))))
            roi.update(x=max(0, min(WIDTH - s, int(body.get("x", roi["x"])))),
                       y=max(0, min(HEIGHT - s, int(body.get("y", roi["y"])))), size=s)
            ROI_FILE.write_text(json.dumps(roi))
            self._json({"ok": True})
        else:
            self._json({"ok": False})


threading.Thread(target=grab_loop, daemon=True).start()


def _latest_frame():
    with lock:
        return latest


def _servo_action(name):
    move_servo(servo_cfg["locked" if name == "lock" else "unlocked"])


def _matrix_action(mode, val):
    Bridge.call("matrix", int(mode), int(val))


try:
    import board_lock
    _arm = board_lock.start(_latest_frame, _servo_action, _matrix_action, port=8090)

    def _button_loop():
        """Poll the sketch for presses of the physical ARM button (D7)."""
        fails = 0
        while True:
            try:
                if Bridge.call("button"):
                    print("[button] pressed -> arm")
                    _arm()
                fails = 0
            except Exception as e:
                fails += 1
                if fails in (1, 100):
                    print("button poll failed:", e)
            time.sleep(0.05 if fails < 5 else 1.0)

    if _arm is not None:
        threading.Thread(target=_button_loop, daemon=True).start()
except Exception as e:  # keep the capture page alive even if the lock fails to start
    import traceback
    print("board_lock failed to start:", e)
    traceback.print_exc()

server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"Capture page: http://<board-ip>:{PORT}")

App.run()
