# Gait Wearable: plan

A wearable made from the Arduino UNO Q and an IMU that tells apart four activities:
**standing still, walking, running, and biking** (or lying on your back and air-biking).
We build two classifiers and compare them: one that only sees the peak acceleration on
one axis, and one that uses every IMU output. No neural nets.

**Priority:** get clean, labeled IMU recordings first (Phases 0–2). Both models are
quick to build once we have good data.

## Assignment prompt

> Make a wearable out of the UNO Q and an IMU (5060) that can differentiate between standing
> still, walking, running, and biking (or lying on your back and air-biking). You should have
> two models: (1) only uses the peak acceleration in one direction, (2) uses all of the IMU
> outputs. You are welcome to place your wearable anywhere you want. You cannot use neural nets
> for this one.

## Decisions (2026-10-22)

| Question | Decision | Why |
|---|---|---|
| IMU | **MPU-6050** (3-axis accel + 3-axis gyro, I2C) ✅ confirmed | |
| Biking | **Both** a real bike and air-biking on your back, all labeled "biking" ✅ | More varied data makes the model more robust |
| Data and models | **Everything stays on the board** ✅: it records, stores and classifies. The laptop only displays | The wearable works on its own; no data streaming over Wi-Fi |
| Placement | **Front of the thigh** ✅, sensor's X axis along the thigh bone, in a 3D-printed enclosure with straps | See "Ankle vs. thigh" and "Enclosure" below |
| IMU driver | Our own (`MPU6050.h/.cpp`) ✅: there's no class library. Reads the chip's FIFO so no samples are lost | |
| Sample rate | 100 Hz; accel range ±8 g, gyro ±1000 °/s | Running peaks at the thigh stay well under 8 g; 100 Hz catches impacts |
| Window | 2 s windows, 50 % overlap (1 new prediction per second) | About 2 strides of walking, 1 pedal stroke is under 1 s |
| Model 1 | Thresholds on a single number: the peak of thigh-axis acceleration in each window | Required by the prompt; tiny and explainable |
| Model 2 | Hand-made window features, then the best of **decision tree, kNN, nearest centroid, k-means** ✅ | Methods from class, no neural net |
| Training | Train on the laptop (scikit-learn) from CSVs copied off the board, then deploy the models back | Plotting and model comparison are easier on the laptop; the board only predicts |
| Output | Board publishes the gait and key numbers on MQTT about once a second; `dashboard.py` on the laptop shows them. The UNO Q LED matrix shows the letter too | See "What the board sends" below |
| Broker | **test.mosquitto.org** by default; the laptop's Mosquitto as an option in the config | Board and laptop both connect *out* to it, so it works even when Tufts Wi-Fi puts them on subnets that can't reach each other (the sumo robot's problem) |
| Power | USB-C power bank in a pocket or on the strap | The UNO Q has no battery input |

### Why the thigh (and not the ankle)

The single-axis model only gets one number per window, so the four classes have to sort
themselves along it. At the thigh, with the sensor axis along the thigh bone, they do:

| Activity | What the thigh axis sees | Expected peak (rough) |
|---|---|---|
| Biking / air-biking | Thigh near horizontal, so gravity along the axis is small and swings with each pedal stroke | **below 1 g** |
| Standing | Thigh vertical: gravity along the axis, nothing else | **about 1 g**, very steady |
| Walking | Gravity plus heel-strike shock | **about 1.3–2 g** |
| Running | Gravity plus much bigger impacts | **above 2.5 g** |

These numbers are guesses; Phase 2 measures the real ones.

### Ankle vs. thigh

| | Thigh (front, mid-way up) | Ankle (outside, above the ankle bone) |
|---|---|---|
| Model 1 (one axis) | ✅ All four classes sort along one axis, thanks to the thigh angle (table above) | ⚠️ Walking vs. running is easy, but biking also swings the ankle, so its peak lands near walking's |
| Model 2 (all outputs) | ✅ Thigh angle alone almost gives away biking and lying down | ✅ Strongest step impacts and cleanest step timing |
| Signal quality | ⚠️ Thigh muscle jiggles during running, adding noise | ✅ Bone close to the skin: sharp, clean impacts |
| Accel range | ✅ Running stays around 3–5 g, so ±8 g is plenty | ⚠️ Running heel strikes can pass 8 g; would need ±16 g |
| A bulky enclosure (battery + UNO Q) | ✅ Big, fairly flat surface; the weight barely changes how you move | ❌ Heavy box on the ankle bounces, changes your gait, and can hit the other leg or the bike crank |
| LED matrix | ✅ You can glance down at it | ❌ Hard to see while moving |
| Strap | ⚠️ Can slide down the thigh when running; needs a wide elastic strap or to go under shorts | ✅ Stays put on the narrow part of the leg |
| Air-biking on your back | ✅ Fine | ✅ Fine |

**Decision: thigh.** With an enclosure holding the battery, the UNO Q and the IMU, the ankle
would be too heavy, and the thigh is also the better spot for model 1. The ankle remains the
backup only if the thigh data turns out too noisy.

### Enclosure (3D printed)

One box on the front of the thigh holding the power bank, the UNO Q and the MPU-6050.

- **IMU screwed tight to the box** (standoffs or a press-fit pocket), never hanging on its
  wires. If it can wobble, you'll record the wobble, not your leg.
- **IMU X axis along the thigh bone**, with the axis printed or engraved on the lid. Model 1
  depends on it, and it lets you put the box back on the same way every time.
- **Window over the LED matrix**: a cut-out, or a thin (0.4–0.6 mm) layer of white or clear
  filament that lets the LEDs shine through.
- **Heaviest part (the battery) at the bottom, against the leg**, so the box doesn't rock.
- **Curved back** that matches the thigh, plus **slots for two wide (≈ 50 mm) elastic velcro
  straps**, top and bottom, so it can't rotate.
- **Openings for the USB-C ports** on the UNO Q and the power bank, so you can charge and
  plug in without opening the box. Add a power switch if the power bank has none.
- Keep it as small and light as possible: measure your power bank first and design around it.
- The power bank must not switch itself off at low current. Some banks turn off below about
  100 mA; test that it keeps the UNO Q running for 10 minutes before you print.

One caveat: lying still on your back also puts the thigh horizontal, so model 1 could call
lying still "biking". Model 2 sees the gyro and the motion rhythm, so it can tell them apart.
Lying still isn't one of the four classes, so we'll just note it.

## Data flow

```
MPU-6050 --I2C--> sketch (STM32) --Bridge--> python/main.py (UNO Q Linux)
                                               |-- record mode: CSV file on the board, labeled
                                               '-- live mode: window -> features -> both models -> label
                                                        |-- MQTT ME193/Jude/gait --> dashboard.py (laptop)
                                                        '-- LED matrix shows S / W / R / B
laptop (training only): copy CSVs back -> train.py -> models (thresholds JSON + model 2 .joblib) -> deploy to board
```

Recording to a file on the board means data collection doesn't depend on the Wi-Fi at all.
In live mode, MQTT is only for display: if the connection drops, the board keeps classifying
and the dashboard catches up when it reconnects.

### What the board sends

Topic `ME193/Jude/gait`, JSON, once per window (about 1 per second):

```json
{"gait": "walking", "confidence": 0.92,
 "model1": "walking", "model2": "walking",
 "peak_g": 1.62, "cadence_hz": 1.8, "pitch_deg": 12,
 "mode": "live", "t": 1761150000.0}
```

- `gait`: the smoothed answer (model 2, majority vote over the last 3 windows)
- `model1` / `model2`: each model's raw answer, so the demo shows them side by side
- `confidence`: share of votes for its answer (tree leaf or kNN neighbors); none for nearest centroid / k-means
- `peak_g`: model 1's single input, `cadence_hz`: steps or pedal strokes per second, `pitch_deg`: thigh angle
- In record mode it sends `{"mode": "record", "label": "running", "seconds": 42}` instead, so the
  laptop can show that a recording is in progress

A retained `ME193/Jude/gait/status` topic says `online` / `offline` (MQTT Last Will), so the
dashboard knows when the wearable drops off.

`dashboard.py` shows the current gait in big text, both models' answers, the live numbers, and a
timeline of the last few minutes. It can also save what it receives to a CSV, which is handy for
the write-up.

## Files

The UNO Q app lives in the private `ArduinoApps` repo (so `tools/deploy.py` can push it), and
a snapshot is copied into `Gait Wearable/unoq/`, like the sumo robot.

| File | Job |
|---|---|
| `ArduinoApps/gait-wearable/sketch/MPU6050.h`, `.cpp` | MPU-6050 driver: setup, ranges, gyro calibration, FIFO reading ✅ written |
| `ArduinoApps/gait-wearable/sketch/sketch.ino` | Sets the IMU to 100 Hz; Python pulls everything in the FIFO with `read_samples()` ✅ written |
| `ArduinoApps/gait-wearable/python/main.py` | Record mode (CSV per session) and live mode (classify and publish) |
| `ArduinoApps/gait-wearable/python/main.py` | Records on MQTT command to `data/<label>_<time>.csv` on the board; status once a second ✅ written |
| `Gait Wearable/config.py` | Every tunable number for the laptop side ✅ |
| `Gait Wearable/gait.py` | `watch`, `record <label> <seconds>`, `start`, `stop` over MQTT ✅ |
| `Gait Wearable/pull_data.py` | Copies the board's recordings into `data/` (uses ArduinoApps' `deploy.py` to find the board) ✅ |
| `Gait Wearable/features.py` | Recording → 2 s windows → model 1's peak and model 2's ~50 features ✅ |
| `Gait Wearable/model_peak.py` | Model 1: single-axis peak plus thresholds ✅ |
| `Gait Wearable/train.py` | Leave-one-session-out scores and confusion matrices for both models; saves them to `models/` ✅ |
| `Gait Wearable/synthetic.py` | Fake recordings for tests and for trying `train.py` before real data ✅ |
| `Gait Wearable/dashboard.py` | Laptop display of the live gait (Phase 5; `gait.py watch` covers recording for now) |
| `Gait Wearable/data/` | Real recordings, one CSV per session, named `<activity>_<date-time>.csv` |
| `Gait Wearable/tests/` | pytest: windowing, features, model 1, end-to-end training on synthetic data ✅ (17 pass) |

Setup: `py -3 -m venv venv`, `venv\Scripts\activate`, `pip install -r requirements.txt`, `pytest`.

### Recording workflow

```powershell
python gait.py watch                  # in one terminal: is the wearable online? live peak / |a|
python gait.py record walking 90      # 5 s countdown, record 90 s, stop
python pull_data.py                   # copy the CSVs off the board (USB or Wi-Fi)
python train.py                       # scores, confusion matrices, models/peak_hist.png
```

The broker is public, so anyone who knows the topic could send a start/stop. Fine for a
class project; we'd add a shared secret to the commands if it ever mattered.

## The two models

**Model 1: peak acceleration in one direction**

- Input per window: `peak = max(accel_x)` over the 2 s window (X = along the thigh).
- Classify by three thresholds: `peak < t1` → biking, `< t2` → standing, `< t3` → walking,
  otherwise running.
- Thresholds come from the training data: the midpoint between neighboring classes'
  peak distributions, or equivalently a depth-2 decision tree on that one feature.
- Plot a histogram of `peak` per class. That plot is the whole story of this model.

**Model 2: all IMU outputs**

Features per window, for each of the six axes and the accel and gyro magnitudes:

- mean, standard deviation, min, max, range
- dominant frequency and its strength (FFT): step or pedal cadence
- mean orientation (pitch and roll from gravity), so lying and standing look different

Plus a few cross-axis ones: correlation between accel axes, and the energy ratio of
gyro to accel. That's roughly 60 numbers per window.

- Classifiers we've covered in class (decided 2026-10-23: no random forest), compared and the
  best one kept:
  - **Decision tree** (depth ≤ 5): `train.py` prints its questions, which explain the model
  - **k-nearest neighbors** (k = 5): the most common label among the 5 closest training windows
  - **Nearest centroid**: the activity whose average window is closest
  - **k-means**: cluster the windows without labels (2 clusters per activity), name each cluster
    after its most common activity, then classify by the nearest cluster center (`model_kmeans.py`)
- The distance-based ones standardize the features first, so gyro (hundreds of °/s) doesn't
  outweigh accel (a few g).

**Evaluation (both models)**

- Split by **session**, not by window: windows from the same recording are almost identical,
  so a random split would give fake-high accuracy. Use leave-one-session-out cross-validation.
- Report accuracy and a confusion matrix for each model, side by side.
- Ideally include a session recorded by someone else, to show it generalizes beyond one person.

## Phases

### Phase 0: Setup
- [x] IMU part confirmed: MPU-6050
- [x] Wire it to the UNO Q: VCC → 3.3 V, GND, SDA, SCL (header pins by AREF, `Wire`)
- [x] Python venv for the laptop side (`requirements.txt`)
- [x] Create `ArduinoApps/gait-wearable` (app.yaml and sketch.yaml copied from sumo-ultrasonic)

### Phase 1: See the IMU
- [x] MPU-6050 driver and sketch written
- [x] Board logs sample rate and the latest reading once a second
- [x] Deployed and checked (2026-10-23): 100 Hz, |a| = 0.99 g at rest, gyro 0.0 ± 0.1 °/s, 0 lost

Lessons from bring-up:
- The sketch sends **raw integer counts**; Python converts to g and °/s (`ACCEL_COUNTS_PER_G`,
  `GYRO_COUNTS_PER_DPS` in `main.py`, which must match the ranges in `sketch.ino`).
- Don't name anything `word` in a sketch: Arduino.h's `word()` macro silently replaces it.
  That bug made every reading come out as 0, 2, 4, ...
- Flashing sometimes fails ("Error requesting gpio line swdio"); just deploy again. After a
  failed flash the board may restart the sumo app, which blocks starting ours until it's stopped.
- `imu_debug` (Bridge) dumps the config registers and live data bytes if readings look wrong.
- [ ] Measure the real sample rate and check no samples are dropped (batch them if the Bridge can't keep up)

### Phase 2: Wearable and data collection
- [ ] Mount the board, IMU and power bank on the thigh. The IMU must not wobble on its own
- [x] Record mode over MQTT ✅ (decided 2026-10-22): `gait.py record <label> <seconds>`; files stay on the board
- [x] Checked (2026-10-23): a 10 s recording over test.mosquitto.org gave 1002 samples, 0 lost, and `pull_data.py` copied it back
- [ ] Record at least 5 sessions × 1–2 min per activity, on different days and in different shoes or pants
- [ ] Plot raw traces per activity and sanity-check them against the "Why the thigh" table

### Phase 3: Model 1 (peak, one axis)
- [x] `features.py` windowing plus unit tests on synthetic signals
- [x] Code: thresholds, leave-one-session-out accuracy, confusion matrix, peak histogram
- [ ] Run on real data and look at `models/peak_hist.png`

### Phase 4: Model 2 (all outputs)
- [x] Full feature set plus unit tests
- [x] Code: decision tree vs. kNN vs. nearest centroid vs. k-means, confusion matrices, the tree's questions
- [ ] Run on real data

### Phase 5: Live on the wearable
- [ ] Run both models on the UNO Q in live mode; publish to MQTT and show on the LED matrix
- [ ] Smooth the output: majority vote over the last 3 predictions so it doesn't flicker
- [ ] `dashboard.py` on the laptop; check it with a fake publisher first, then the real wearable
- [ ] Live test: walk → run → stand → bike, and check the switch-over delay (about 2 s expected)

### Phase 6: Write-up and demo
- [ ] Compare model 1 vs. model 2: accuracy, which classes get confused and why
- [ ] Short video of the live demo

## Open questions

- Is `ME193/Jude/gait` on test.mosquitto.org OK, or does the class expect a specific topic?
- Does "peak acceleration in one direction" have to be the raw peak, or is peak-to-peak on
  one axis allowed? Peak-to-peak removes gravity, which helps if the strap rotates.
- What are the deliverables and the due date?
