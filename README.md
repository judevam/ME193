# ME193

Python scripts for controlling LEGO Education hardware using the
[legoeducation](https://github.com/LEGO/LEGOEducation) library.

## Setup

```powershell
:: 1. Create a virtual environment named "my_env"
python -m venv my_env

:: 2. Activate the virtual environment
my_env\Scripts\activate.bat

:: 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Files

- [single_motor.py](single_motor.py) - connects to a Single Motor, runs it at low
  speed, speeds up after one full rotation, then stops and disconnects.
- [arm_control_car.py](arm_control_car.py) - drives a LEGO car (Double Motor) with
  your arms: a webcam + mediapipe track your pose, and raising/leveling your
  arms sets the car's throttle and steering in real time. Also runs a second,
  independent Single Motor at a constant speed.
- [find_devices.py](find_devices.py) - scans for nearby LEGO Bluetooth
  hardware and prints each one's real card color/serial - use this if a
  `connect()` call can't find your hardware.

## Usage

### Single Motor

With the virtual environment activated and a Single Motor connected via its
Connection Card, update `CARD_COLOR` and `CARD_SERIAL` in `single_motor.py` to
match your card, then run:

```powershell
python single_motor.py
```

### Arm-controlled race car

With a webcam attached and the car's Double Motor powered on, update
`CARD_COLOR` and `CARD_SERIAL` in `arm_control_car.py` to match its Connection
Card, then run:

The script also connects to a separate Single Motor (its own Connection Card
via `SINGLE_MOTOR_CARD_COLOR`/`SINGLE_MOTOR_CARD_SERIAL`) and spins it at a
constant `SINGLE_MOTOR_SPEED` for as long as the script runs, independent of
arm gestures.

```powershell
python arm_control_car.py
```

The first run downloads mediapipe's `pose_landmarker_lite.task` model into
`models/` (not committed to git - it's fetched automatically). A window shows
the camera feed with dots on your shoulders/wrists and the live left/right
motor speeds. Controls:

- **Left hand raised alone** - forward.
- **Right hand raised alone** - backward.
- **Both hands raised** - spin left in place.
- **Hands together** (close to each other) - spin right in place.
- **Neither hand raised** - stop.
- **Press `q`** - quits and stops the motors.

(This mapping came from testing on the actual robot - it isn't the most
"obvious" gesture-to-motion mapping on paper, but it's what felt right with a
hand on the controls.)

If the car isn't connected, the script still runs in camera preview-only mode
(useful for tuning the gesture logic without hardware).

**If a device won't connect** even after tapping it with its Connection Card:
double-check `CARD_COLOR`/`CARD_SERIAL` (or `SINGLE_MOTOR_CARD_*`) actually
match that specific card - it's easy to leave a placeholder value in place
(the AZURE/3683 example values from LEGO's own docs look like real values).
Run `python find_devices.py` while the hardware is powered on and freshly
tapped to scan for what's actually broadcasting nearby and read off its true
card color/serial.

## Answers

**How is Python talking to the LEGO hardware?**
Over Bluetooth Low Energy (BLE). `legoeducation` depends on
[`bleak`](https://pypi.org/project/bleak/), a cross-platform BLE library
(backed here by Windows's `winrt` bindings), and `connect()` explicitly scans
for and connects to the hardware over Bluetooth using the Connection Card's
color/serial as a filter. No cable or radio dongle is involved - it's the same
kind of link a phone uses to pair with a Bluetooth speaker.

**Is it synchronous or asynchronous?**
Both, depending on how you call it. Every motor command (`motor_run`,
`motor_stop`, `movement_move_for_time`, ...) takes a `blocking` argument
(default `True`), so by default the library presents a synchronous API - your
code waits for the hardware to acknowledge (or, for timed/degree moves,
finish) the command before the next line runs. Underneath, BLE communication
via `bleak` is inherently asynchronous (`asyncio`-based), so the library must
be running an internal event loop that the blocking calls wait on. We use
`blocking=False` in `arm_control_car.py` specifically because the control loop
is asynchronous by nature: it must keep reading camera frames at whatever rate
the webcam and pose model deliver them, and can't afford to stall on a BLE
round-trip every frame - it just fires the latest speed and moves on.

**How did you train it, and what are its limitations?**
We didn't train the pose-tracking model ourselves - `arm_control_car.py` uses
mediapipe's pre-trained `pose_landmarker_lite` model (Google's BlazePose,
trained on their own large annotated pose dataset) as a fixed, off-the-shelf
detector. The part we did "train" (really: hand-tune) is the mapping from
landmark positions to motor speeds - the `MAX_RAISE`, `STEER_GAIN`, and
`SEND_THRESHOLD` constants in `arm_control_car.py`, adjusted by watching the
on-screen L/R speed readout while moving our arms and iterating until the
response felt right.

Limitations:
- **Lighting/background/occlusion sensitive**: the pretrained model can lose
  the pose in poor lighting, cluttered backgrounds, or when arms leave the
  frame or cross in front of the body.
- **Single person, single camera**: `num_poses=1` and one webcam mean no
  depth perception and no support for multiple people in frame.
- **Hand-tuned thresholds don't generalize**: `MAX_RAISE`/`STEER_GAIN` were
  tuned for one person's arm length and one camera distance/angle - a
  different driver or camera setup likely needs re-tuning.
- **BLE latency and range**: Bluetooth adds noticeable lag and has limited
  range, so there is a delay between an arm movement and the car reacting,
  and the car must stay reasonably close to the computer.
- **No obstacle awareness**: the system only reads arms; it has no sensor
  feedback from the car's environment, so a "race out of the room" still
  depends entirely on the driver watching where the car is going.
