# Ping Pong midterm: plan

Virtual solo ping-pong against a wall. The LEGO **Double Motor is the paddle**: its IMU tells
us when you swing. The **webcam** does three jobs: pose (is your hand where the ball is?),
AprilTags (pick a level to start the game), and the background of the game window. The
current streak of continuous hits is published as a float to `ME193/Rogers/Jude`.

**Priority:** an IMU swing hits a ball (Phases 0–3). Pose, AprilTags and the extras come after.

## Assignment prompt

> Play virtual ping-pong with your Double Motor or UNO Q. Include pose information (to know if
> the paddle is in the right location), IMU information (to know if the person is swinging it
> or not), AprilTag info for player level (ball speed) and the start of the game, and the score
> should be posted in real time to ME193/Rogers/YourName with the score as a floating point
> number that is the record number of continuous hits.
> - Needs to use everything you have learned so far in class
> - Needs to do something fun/cool that we have not done in class (like haptic feedback)

## Decisions so far (2026-10-06)

| Question | Decision |
|---|---|
| Paddle / IMU | LEGO Double Motor held as the paddle (IMU over BLE via `lelib`) |
| Display | Start with the webcam window + overlay; later a separate Wii Sports Resort-style table window |
| Game mode | Solo vs. a wall; streak = consecutive good hits |
| MQTT | `ME193/Rogers/Jude` on test.mosquitto.org, publish the **current streak** as a float on every change (back to `0.0` on a miss) |
| Workflow | Unit test and debug each phase before moving on |

The IMU provides `accelerometerX/Y/Z`, `gyroscopeX/Y/Z`, yaw/pitch/roll and built-in gestures
(tap, double tap, collision, **shake**, freefall), so a swing can be detected from the gyro.

**Status (2026-10-07):** Phase 0 done ✅ (`pytest "ping pong/tests"` passes, run with `my_env`).
Phase 1 done ✅ (63 Hz; gyro Y is the main swing axis).
Swing = gyro size > 600 **and** accel > 3 g within 0.15 s, cooldown 0.4 s. The accel check was
added after live testing: twisting the motor in place set off the gyro-only version.
Phase 2 done ✅ (live check passed).
Phase 3 done ✅ (hit window moved earlier to -0.35/+0.20 s after live play; `main.py` shows each
swing's timing in ms). Swinging before the window is a miss only in the second half of the ball's
flight. `--fake-imu` (space bar) lives in `main.py` instead of a `KeyboardPaddle` class.
Phase 4 done ✅ (`scoreboard.py`; round trip through test.mosquitto.org verified; 53 tests pass).
Phase 5 code written (`pose_tracker.py`, lanes in `game.py`/`main.py`; 70 tests pass): three lanes
(screen thirds), wrist checked at the moment of the swing (last position within 0.35 s), pose runs
on the un-mirrored frame so mediapipe's left/right stay correct. Tracking upgrades (option A,
2026-10-08): POSE_MODEL lite/full/heavy, wrist smoothing, confidence settings in config.py (75 tests).
Training our own model (YOLO on the paddle, like minifig-blue) is the fallback if that's not enough.
Kept POSE_MODEL = lite after live comparison (heavy was bad, full ~ lite).
Phase 6 code written (`level_tags.py`, start screen in `main.py`; 90 tests pass): ONE tag (id 0,
36h11, same as aprilTags/apriltag_0.png); where it's held picks the level, using the lane columns:
left = easy, middle = medium, right = hard. Hold 0.5 s in one column; moving column restarts the hold.
Keys 1/2/3 do the same; r returns to the start screen; best streak carries over.
Printable tag: `python level_tags.py --make` -> tags/start_tag.png.
Gameplay tweaks (2026-10-08, after play-testing; 98 tests): only a ball getting past you ends the
streak. Too early / wrong spot / hand not seen are "whiffs" (feedback, swing again), and the MISS
message gives the last whiff's reason. The ball now arrives at the wrist's screen height
(HIT_Y_* in config.py; follows slowly, frozen during the hit window) so backing up doesn't make
you swing early.
Feel pass (2026-10-08, 105 tests): camera + pose moved to a background thread (`camera_worker.py`)
so drawing runs at ~60 fps; FPS counter bottom-right. Timing ring closes onto the hit circle at
arrival; flatter ball motion (PERSPECTIVE 1.25). Hit burst; a missed ball flies past instead of
freezing. Ball's lane column is highlighted (screen thirds, same as the wrist zones); the hit circle
locks once the ball is halfway; LANE_TOLERANCE 0.05 forgives near-the-line hands.
Phase 7 started (2026-10-08, 111 tests): `paddle_feedback.py` buzzes the paddle (motor_run_for_time
on HAPTIC_MOTOR side) and flashes its light (hit green, miss red + longer weaker buzz, whiff orange
light only), from a background thread, motor + light in one BLE batch. `python paddle_feedback.py`
checks the IMU rate and false swings while buzzing. HAPTIC_MOTOR = "opposite" (both outputs spin opposite
ways in one batch) works on the real paddle (2026-10-08); buzz is light. Stronger: an off-centre LEGO
beam on each output (eccentric mass), or longer buzz ms in config.HAPTIC.
Forehand **and** backhand both count (detector uses total gyro size, no direction).

## How one hit works

```
ball flies toward you ──► reaches the hit zone (a ~0.4 s window)
                              │
      IMU: swing detected? ───┤  no  → miss → streak = 0 → publish 0.0
                              │ yes
      pose: wrist near ball? ─┤  no  → miss (added in Phase 5)
                              │ yes
                              ▼
          HIT → streak += 1 → publish → ball bounces off the wall and comes back
```

## Files (each one runs and is tested on its own)

| File | Job | Runs standalone as | Hardware-free stand-in |
|---|---|---|---|
| `config.py` | every tunable number in one place | – | – |
| `imu_paddle.py` | read gyro/accel from the Double Motor | live readout + CSV logger | `KeyboardPaddle` (space = swing) |
| `swing.py` | turn IMU samples into "swing!" events | – (pure logic) | recorded CSVs |
| `game.py` | ball, hit window, streak; no camera/BLE/MQTT | – (pure logic) | fake clock |
| `scoreboard.py` | publish streak to MQTT | publishes a test value | `FakeScoreboard` |
| `pose_tracker.py` | wrist position from mediapipe | webcam with wrist dot | mouse position |
| `level_tags.py` | AprilTag ID → level/ball speed, start game | webcam with tag outline | keys 1/2/3 |
| `main.py` | wires it all together, draws the window | the game | `--fake-imu`, `--fake-pose` flags |
| `tests/` | pytest for `swing`, `game`, `level_tags`, `scoreboard` | `pytest "ping pong/tests"` | – |

The pure logic (`swing.py`, `game.py`) has no hardware in it, so it can be unit tested on
any laptop. The hardware files each have a demo you run once to confirm the hardware works
before it's plugged into the game.

## Phases

Each phase ends with a **check**. Don't move on until that check passes.

### Phase 0: Setup
- Copy `lelib.py` and `mqttlib.py` in, add `pytest` to `requirements.txt`, make `tests/`.
- **Check:** `pytest "ping pong/tests"` runs (0 tests is fine).

### Phase 1: See the IMU (hardware)
- `imu_paddle.py`: connect, print gyro X/Y/Z, accel X/Y/Z, measure samples/second.
- Record CSVs: holding still, walking around, small wiggles, real swings (5–10 each).
- **Check:** we know the sample rate and BLE lag, which gyro axis a forehand swing shows up on,
  and roughly what number separates "swing" from "not a swing".

### Phase 2: Swing detector (pure logic + unit tests)
- `SwingDetector.update(sample, t)` returns `True` once per swing: gyro magnitude above a
  threshold, and a cooldown so one swing doesn't count twice.
- Phase 1's CSVs become test fixtures: every real swing is detected exactly once, and
  still/walking/wiggle recordings give zero swings.
- **Check:** tests pass; live demo prints `SWING` once per swing.

### Phase 3: First playable (milestone: the IMU hits the ball)
- `game.py`: the ball starts far away (small), grows as it approaches, has a hit window, and
  bounces back on a hit. Tracks the streak. Time is passed in, so tests can fast-forward.
- `main.py`: webcam window with the ball drawn on top, streak counter, swing flash.
- Tests: a swing in the window counts as a hit; too early or too late is a miss; a miss
  resets the streak; there's one hit per ball, however much you swing.
- **Check:** play with `--fake-imu` (space bar), then with the real paddle.

### Phase 4: Live score over MQTT
- `scoreboard.py` publishes `float(streak)` to `ME193/Rogers/Jude` on test.mosquitto.org
  whenever the streak changes, including going back to `0.0`.
- **Check:** `python mqtt/publish.py listen ME193/Rogers/Jude` shows `1.0, 2.0, … 0.0` while playing.

### Phase 5: Pose (the hand must be in the right place)
- `pose_tracker.py`: hitting-hand wrist position (same mediapipe setup as `poserace/`).
- Each ball arrives left, center or right. A hit needs a swing **and** the wrist in that zone.
- **Check:** `--fake-pose` (mouse) tests, then live: swinging with your hand in the wrong
  zone is a miss.

### Phase 6: AprilTags (level and start)
- Show tag 1/2/3 to the camera to pick easy/medium/hard (ball speed) and start the game.
- **Check:** each tag starts a game at its speed; no tag means waiting on the start screen.

### Phase 7: The fun stuff (later)
- Haptic feedback: a short motor buzz in the paddle on each hit (and a different one on a miss).
- A separate Wii Sports Resort-style game window: a 3D-looking table with a perspective ball and shadow.
- Sound effects, a high-score screen, UNO Q LED matrix scoreboard.

## Answered questions (2026-10-07)
- Hitting hand: **right** (pose tracks the right wrist).
- Paddle: Double Motor with **green 0026** Connection Card, **gripped like a handle** (custom grip).
- The hub's default IMU rate is 100 ms (10 Hz), too slow for a swing; `imu_paddle.py` requests
  15 ms (the fastest allowed, about 66 Hz).
