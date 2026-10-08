"""Every tunable number for the ping pong game, in one place."""

# ── Paddle (Double Motor) ────────────────────────────────────────────────────
PADDLE_CARD_COLOR = "green"   # converted to le.LEGO_COLOR_GREEN in imu_paddle.py
PADDLE_CARD_SERIAL = "0026"

# How often the hub sends IMU data. The library default is 100 ms (10 Hz), too
# slow to catch a swing; 15 ms is the fastest the hub allows (~66 Hz).
IMU_NOTIFY_MS = 15

# ── Swing detection (from the Phase 1 recordings, 2026-10-07) ───────────────
# A swing needs the paddle rotating (gyro) AND whipped through an arc (accel). Twisting the
# motor in place rotates it but barely accelerates it, so the accel check filters that out.
# Peak gyro size:  still 0, walk 236, wiggle 326        | soft swing 874+, hard 1261+
# Peak accel size: still 1007, walk 1502, wiggle 1513   | soft swing 5554+, hard 7661+  (milli-g)
SWING_THRESHOLD = 600         # total gyro size that counts as a swing (either direction)
SWING_ACCEL_THRESHOLD = 3000  # total accel size, milli-g (3 g)
SWING_PAIR_WINDOW_S = 0.15    # gyro and accel must both cross within this long of each other
SWING_COOLDOWN_S = 0.4        # ignore the follow-through bounce right after a swing

# ── Game ─────────────────────────────────────────────────────────────────────
# Seconds for the ball to travel between the wall and you, per level (Phase 6 picks the level
# with an AprilTag). A full rally is wall -> you -> wall, so twice this.
LEVEL_TRAVEL_S = {1: 1.6, 2: 1.2, 3: 0.9}
DEFAULT_LEVEL = 1

# Hit window around the moment the ball reaches you. Wider on the early side: the swing detector
# fires at the start of the forward stroke (when it passes 3 g), before the paddle would actually
# meet the ball. Live play on 2026-10-08 was consistently too early with -0.20 s.
# main.py prints each swing's timing ("swing -180 ms") to tune these.
HIT_EARLY_S = 0.35
HIT_LATE_S = 0.20

# A swing that isn't a hit (too early, wrong lane, hand not seen) is a "whiff": you get feedback but
# the ball keeps coming and you can swing again; only the ball getting past you ends the streak.
# Whiff feedback starts once the ball is this far along (0 = wall, 1 = you); earlier swings are
# ignored silently.
WHIFF_FROM_Z = 0.5

# Where the ball reaches you on screen follows the height of your wrist (so backing up or
# crouching doesn't make you swing early). It follows slowly, freezes while you can hit the ball,
# and stays within this range of the screen height (0 = top, 1 = bottom).
HIT_Y_DEFAULT = 0.80          # used when pose is off or the wrist hasn't been seen
HIT_Y_RANGE = (0.40, 0.85)
HIT_Y_FOLLOW_S = 0.4          # time constant: how quickly it catches up with the wrist

SERVE_DELAY_S = 1.0   # ball waits at the wall before each serve
MISS_PAUSE_S = 1.2    # time to show "MISS" before the next serve

# ── AprilTags (Phase 6) ──────────────────────────────────────────────────────
# One tag starts the game; where it's held picks the level, using the same three columns as the
# ball lanes (LANE_EDGES): left = easy, middle = medium, right = hard.
START_TAG_ID = 0                  # family 36h11, same as aprilTags/ (aprilTags/apriltag_0.png)
TAG_HOLD_S = 0.5                  # hold it in one column this long to start

# ── Paddle feedback (Phase 7) ────────────────────────────────────────────────
# A hit buzzes the paddle and flashes its light; a miss gives a longer, weaker buzz. The buzz is
# the Double Motor spinning its outputs briefly, so they must have NOTHING structural on them (not
# the axle your grip hangs on!). An off-centre LEGO piece on a spinning axle makes a stronger buzz,
# the same way a phone's vibration motor works.
#   "left" / "right": one output spins
#   "both":           both spin the same way
#   "opposite":       both spin at once in opposite directions (a stronger jolt)
HAPTIC_MOTOR = "opposite"
HAPTIC = {                     # event -> (buzz ms, motor speed %, light color); 0 ms = light only
    "hit": (90, 100, "green"),
    "miss": (350, 45, "red"),
    "whiff": (0, 0, "orange"),
}
LIGHT_FLASH_S = 0.4            # how long the light stays on the event color
LIGHT_IDLE_COLOR = "blue"      # the light's resting color while playing

# ── Live score ───────────────────────────────────────────────────────────────
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC = "ME193/Rogers/Jude"   # current streak as a float, e.g. "3.0"; "0.0" after a miss

# ── Pose (Phase 5) ───────────────────────────────────────────────────────────
HITTING_HAND = "right"        # pose_tracker.py follows this wrist
# Each ball comes down one of three lanes. Your wrist's lane is which third of the (mirrored)
# screen it's in: x < LANE_EDGES[0] is left, x > LANE_EDGES[1] is right, between is center.
LANE_EDGES = (1 / 3, 2 / 3)
# The wrist position used for a swing must be at most this old (the camera runs ~30 fps;
# a fast swing can blur the wrist, so the last good position before the swing is used).
WRIST_MAX_AGE_S = 0.35
# A hand this close (fraction of screen width) outside the ball's lane still counts as in it.
LANE_TOLERANCE = 0.05

# MediaPipe pose model: "lite" (fastest), "full", or "heavy" (most accurate, slowest).
# Downloaded into models/ on first use. Compare them with: python pose_tracker.py --model full
POSE_MODEL = "lite"
# How sure mediapipe must be (0-1). Lower tracking confidence keeps following a blurry,
# fast-moving arm instead of dropping it, at the cost of the odd wrong position.
POSE_MIN_DETECTION_CONFIDENCE = 0.5   # to find a person in a fresh frame
POSE_MIN_TRACKING_CONFIDENCE = 0.5    # to keep following them frame to frame
WRIST_MIN_VISIBILITY = 0.5            # ignore the wrist if mediapipe thinks it's hidden
# Smoothing: each new wrist position moves this fraction of the way from the old one.
# 1.0 = no smoothing (raw, jittery); lower = steadier but lags behind a moving hand.
WRIST_SMOOTHING = 0.6
