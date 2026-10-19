# MQTT LEGO Sumo Robot

A LEGO sumo robot driven by a behavior tree. The brain runs on the laptop. The
LEGO devices talk Bluetooth (BLE) to the laptop, and the Arduino UNO Q rides on
the robot with an ultrasonic sensor and a buzzer. Every hop between the
devices and the brain goes through MQTT, so everything can be logged and replayed.

```
color sensor (BLE) ─► sensors/color_bridge ───┐                 ┌─► cmd/motors  ─► bridges/drive_bridge   ─► Double Motor (BLE)
ultrasonic (UNO Q) ─► sumo-ultrasonic app ────┼─► sensor/*  ─►  │─► cmd/flipper ─► bridges/flipper_bridge ─► Single Motor (BLE)
controller (BLE)   ─► teleop/joystick_bridge ─┤    brain/       ├─► cmd/sound   ─► UNO Q buzzer / bridges/sound_bridge
keyboard           ─► teleop/keyboard_client ─┘    (tree)       └─► state       ─► logs
                                       teleop/*
```

## Setup (laptop)

```powershell
cd sumo
py -3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pytest                      # 31 tests, no hardware needed
```

### Mosquitto reachable from the UNO Q

By default Mosquitto 2.x only accepts connections from the laptop itself. Open
PowerShell **as Administrator**:

```powershell
# 1. Listen on the network (allow_anonymous is fine on a class network, not the internet)
Add-Content "C:\Program Files\mosquitto\mosquitto.conf" "`nlistener 1883 0.0.0.0`nallow_anonymous true"
Restart-Service mosquitto

# 2. Firewall: allow port 1883 in
New-NetFirewallRule -DisplayName "Mosquitto 1883" -Direction Inbound -Protocol TCP -LocalPort 1883 -Action Allow
```

Then set `BROKER_HOST` in `ArduinoApps/sumo-ultrasonic/python/main.py` to the
laptop's Wi-Fi IP (`ipconfig`). It changes when you switch networks.
Laptop-side processes use `mqtt.host: localhost` in `config/sumo.yaml`.

### LEGO firmware

`legoeducation` checks that the device firmware matches the package's RPC version.
If they don't match, the bridges stop with a **VERSION MISMATCH** message that
says which side is out of date. Then either:
- update the device firmware: connect it once in the LEGO Education Coding Canvas app (code.legoeducation.com), or
- change the `legoeducation==` pin in `requirements.txt` (`pip install --upgrade legoeducation`).

## Running

**Mock mode (no robot):**

```powershell
python run_demo.py
```

This starts the brain, the drive, flipper and sound bridges (all mocked), the
keyboard client and the fake sensors. Keys are global, so they work from any window:

| Key | Fake world (`sensors/fake_sensors.py`) | | Key | Controls (`teleop/keyboard_client.py`) |
|---|---|---|---|---|
| `f` / `b` | floor = ring / blue tape | | `space` | fire flipper |
| `1`..`5`, `0` | opponent at 5/10/14/30/80 cm, none | | `m` | flipper auto / manual |
| arrows | levers (tank drive) | | `d` | manual defeat |
| `j` | controller connected / disconnected | | `r` | reset after defeat |
| | | | `q` | quit everything |

To demo each branch:

| Branch | Keys |
|---|---|
| TELEOP_DRIVE | hold ↑ |
| AVOID_EDGE: only forward on tape | press `b`, then hold ↓ (stays stopped) or ↑ (drives forward); `f` to clear |
| HOLD_GROUND | `3` (14 cm) |
| AUTO_FIRE + cooldown | `1` (5 cm): fires every 2 s |
| MANUAL_MODE blocks auto | `m`, then `1`; `space` still fires |
| stale joystick → IDLE | `j` while holding ↑ |
| DEFEATED | `d`, then `r` to reset |

**Hardware:**

1. Turn on all four LEGO devices and run `python -m tools.ble_smoke_test`. Use its
   output to fill in the `TODO(hardware)` settings in `config/sumo.yaml`.
2. Deploy the UNO Q app: from `ArduinoApps`, run `python tools/deploy.py sumo-ultrasonic`.
   Before the sensor is wired, run `python -m sensors.mock_ultrasonic` on the laptop instead.
3. Run `python run_demo.py --hardware` (uses the `devices:` modes from the config).

To mix and match, set any entry under `devices:` in `config/sumo.yaml` to `mock` or
`hardware`, and run the processes yourself, each in its own terminal:
`python -m brain.main`, `python -m bridges.drive_bridge`, and so on.

## Topics

Prefix `sumo/<robot_id>/` (default `bot1`). Defined in `common/topics.py`; payloads in `common/schemas.py`.

| Topic | Direction | Payload | QoS | Retained |
|---|---|---|---|---|
| `sensor/color` | color bridge → | `{"name":"blue","rgb":[r,g,b],"t"}` | 0 | |
| `sensor/ultrasonic` | UNO Q → | `{"cm":34.5,"t"}` | 0 | |
| `teleop/joystick` | joystick bridge → | `{"left":-1..1,"right":-1..1,"t"}` | 0 | |
| `teleop/flipper_mode` | keyboard → | `"auto"` / `"manual"` | 1 | ✓ |
| `teleop/flipper_fire` | keyboard → | `{"t"}` | 1 | |
| `teleop/defeat` | keyboard → | `{"t"}` | 1 | |
| `cmd/motors` | brain → | `{"left","right","t","src_t"}` every tick | 0 | |
| `cmd/flipper` | brain → | `{"action":"fire"\|"home","t"}` | 1 | |
| `cmd/sound` | brain → | `{"song":"sad","t"}` | 1 | |
| `state` | brain → | `{"node","flipper_node","defeated","flipper_mode","t"}` on change | 0 | ✓ |
| `control/reset` | keyboard → | `{"t"}` | 1 | |
| `status/<node>` | every node | `"online"` / `"offline"` (Last Will) | 1 | ✓ |
| `status/<node>/ble` | BLE bridges | `"connected"` / `"disconnected"` | 1 | ✓ |

## The behavior tree and the assignment

The tree is in `brain/tree.py` (py_trees; the docstring has the full diagram).
DEFEATED comes first and latches. It is triggered only by the **d** key: the robot does not detect being pushed out by itself. Under it, a Parallel node decides DRIVE and
FLIPPER independently on every tick.

| Requirement | How it's met |
|---|---|
| Decision policy | Behavior tree (py_trees), `brain/tree.py` |
| Autonomous sensor #1 | LEGO color sensor: edge avoidance (AVOID_EDGE) |
| Autonomous sensor #2 | HC-SR04 on the UNO Q: HOLD_GROUND and AUTO_FIRE |
| Teleoperated input | LEGO Controller levers (TELEOP_DRIVE) and the laptop keyboard (flipper fire/mode, defeat, reset) |
| UNO Q as a sensor over MQTT | `ArduinoApps/sumo-ultrasonic` publishes `sensor/ultrasonic` |
| Stop and play a sad song when out | Press **d** when pushed out. DEFEATED stops the motors, parks the flipper and sends `cmd/sound` once (UNO Q buzzer and/or laptop) |

## Safety

- Edge avoidance outranks the levers. The color sensor is at the back, so on blue tape (and for `edge.clear_s` after) no wheel may turn backwards: the operator can't reverse off the edge.
- If the color sensor goes quiet, there's no driving at all (`brain.require_color_for_drive`).
- Stale lever data (> 0.3 s) is ignored. The tree falls through to HOLD_GROUND (only if an opponent is close) or IDLE.
- The drive bridge stops the wheels when no command arrives for 0.5 s, when the brain's Last Will says offline, on a BLE write failure, and on exit (Ctrl+C / `run_demo.py` shutdown).
- The flipper never fires while defeated. The tree can't do it, and the flipper bridge refuses too.
- Motor values are clamped to [-1, 1] by the brain, the bridge and the motor class.
- BLE devices reconnect with backoff (1 s doubling to 10 s) and report on `status/<node>/ble`.

## Design decisions and assumptions

- **Tank drive.** The LEGO Controller has two levers (`leftPercent`/`rightPercent`), not an x/y stick, so `teleop/joystick` carries `{left, right}` and each lever drives its own wheels.
- **"Joystick fresh" means levers moved.** Levers at rest (inside `joystick.deadzone`) fall through to HOLD_GROUND/IDLE. Otherwise the connected controller would always win and HOLD_GROUND could never run.
- **Freshness uses arrival time** on the laptop, not the sender's `t`, because the UNO Q's clock may not match the laptop's. `t` is used for latency logging: the drive bridge logs median/max sensor→motor latency every 5 s.
- **Added topics:** `teleop/defeat` (the brief has a defeat key but no topic) and per-node `status/<node>` (one shared `status` topic couldn't tell nodes apart).
- **Color sensor at the back.** On blue the robot stops and the levers can only drive forward (backward lever input is clamped to 0, so a spin becomes a forward pivot). There's no automatic maneuver.
- **Flipper stroke:** `fire` moves out to `fire_deg` and back to `home_deg` in one command. The brain re-sends `home` once the stroke time (`flipper.sweep_s`) has passed, as a backup.
- **Defeat is manual only** (the `d` key). Automatic out-of-ring and tilt detection were removed at the team's request.
- **The UNO Q app lives in `ArduinoApps/sumo-ultrasonic`** (the App Lab folder that `deploy.py` copies), not in `sumo/unoq/`. The sad melody is copied there from `common/sounds.py`.

## TODO(hardware) checklist

- [ ] Check that all four BLE connections stay up together (the smoke test's last step)
- [ ] UNO Q pins `TRIG_PIN`, `ECHO_PIN`, `BUZZER_PIN` in `ArduinoApps/sumo-ultrasonic/sketch/sketch.ino`
- [ ] **HC-SR04 echo is 5 V; the UNO Q is 3.3 V.** Use a 1k/2k divider on ECHO (see the sketch), or a 3.3 V sensor
- [ ] `BROKER_HOST` in `ArduinoApps/sumo-ultrasonic/python/main.py` = the laptop's current IP
- [ ] Tune speeds, distances and timings in `config/sumo.yaml` on the real ring
- [ ] Weigh the robot (≤ 1 kg) and check it fits a 30 cm cube with the flipper raised
