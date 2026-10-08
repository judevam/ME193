"""Ping pong against a wall: the webcam window with the ball drawn on top.

    python main.py --fake-imu        # space bar = swing (no paddle needed)
    python main.py                   # real Double Motor paddle
    python main.py --no-camera       # plain background instead of the webcam
    python main.py --camera 1        # pick another webcam (e.g. a phone via Camo)
    python main.py --no-mqtt         # don't publish the streak
    python main.py --fake-pose       # the mouse is your wrist (no pose tracking)
    python main.py --no-pose         # don't check where your hand is
    python main.py --pose-model full # lite / full / heavy (default: config.POSE_MODEL)
    python main.py --level 2         # skip the start screen and play level 2
    python main.py --no-haptics      # no buzz / light on the paddle

With the real paddle, a hit buzzes it and flashes its light green; a miss gives a longer, weaker
buzz and a red flash (paddle_feedback.py; settings in config.py).

The game opens on a start screen: hold up the start tag (python level_tags.py --make prints it)
in the left / middle / right of the screen to start easy / medium / hard. Keys 1/2/3 do the same.

The current streak is published to config.MQTT_TOPIC whenever it changes.
Each ball comes down the left, center or right lane (its column is highlighted): swing with your
wrist in that column as the shrinking ring closes on the ball. The dot shows where the game thinks
your wrist is: green = right lane, red = wrong lane.

Keys: space = swing (with --fake-imu), 1/2/3 = start a level, r = back to the start screen, q = quit.
"""

import argparse
import math
import time

import cv2
import numpy as np

import config
from game import Game, SERVE, INCOMING, RETURNING, MISSED, ANY_LANE
from level_tags import LEVEL_NAMES, LevelPicker
from pose_tracker import hand_lane
from scoreboard import Scoreboard
from swing import SwingDetector

WINDOW = "Ping Pong (q to quit)"
WIDTH, HEIGHT = 960, 720
TARGET_FPS = 60

# Perspective: the wall is a small rectangle near the top, you are at the bottom edge.
HORIZON_Y = 0.22      # fraction of the height where the wall is
BALL_R_FAR, BALL_R_NEAR = 6, 46
PERSPECTIVE = 1.25    # >1: the ball speeds up on screen as it gets close (1 = steady)

# Timing ring: shrinks onto the hit circle, landing exactly when the ball arrives.
APPROACH_S = 0.8          # shown for the last this-many seconds of the ball's flight
APPROACH_PX_PER_S = 220   # how fast it shrinks
BURST_S = 0.3             # length of the hit burst

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, ORANGE, YELLOW = (60, 220, 60), (60, 60, 230), (0, 140, 255), (0, 230, 255)

MESSAGES = {"hit": ("HIT!", GREEN), "serve": ("Serve!", WHITE),
            # whiffs: feedback only, the ball keeps coming
            "whiff_early": ("Too early!", ORANGE), "whiff_lane": ("Wrong spot!", ORANGE),
            "whiff_no_hand": ("Hand not seen!", ORANGE)}
MISS_MESSAGES = {"late": "MISS - too late", "early": "MISS - too early",
                 "lane": "MISS - wrong spot", "no_hand": "MISS - hand not seen"}
MESSAGE_S = 0.8
SWING_FLASH_S = 0.15


# ── Drawing (no camera or BLE, so it can be tested) ──────────────────────────

def lane_x(lane):
    """Screen x (0..1) of a lane's centre where the ball reaches you: the middle of that third."""
    edges = (0.0, *config.LANE_EDGES, 1.0)
    return (edges[lane] + edges[lane + 1]) / 2


class HitHeight:
    """Screen height (0..1) where the ball reaches you. Follows the wrist's height slowly,
    stays inside config.HIT_Y_RANGE, and holds still while frozen (once the ball is on its way in)."""

    def __init__(self, default=config.HIT_Y_DEFAULT, y_range=config.HIT_Y_RANGE,
                 follow_s=config.HIT_Y_FOLLOW_S):
        self.y = default
        self.range = y_range
        self.follow_s = follow_s

    def update(self, wrist_y, dt, frozen=False):
        if wrist_y is not None and not frozen:
            target = min(max(wrist_y, self.range[0]), self.range[1])
            k = 1 - math.exp(-dt / self.follow_s)   # frame-rate independent smoothing
            self.y += k * (target - self.y)
        return self.y


def ball_screen(z, w, h, lane=1, hit_y=config.HIT_Y_DEFAULT):
    """Ball centre and radius on screen for distance z (0 wall .. 1 you, over 1 when it gets past).
    It leaves the middle of the wall and spreads out to its lane, arriving at height hit_y."""
    p = z ** PERSPECTIVE
    x = w * (0.5 + (lane_x(lane) - 0.5) * p)
    y = h * (HORIZON_Y + (hit_y - HORIZON_Y) * p)
    r = BALL_R_FAR + (BALL_R_NEAR - BALL_R_FAR) * p
    return (int(x), int(y)), max(2, int(r))


def draw_table(img):
    """Translucent table that gives the scene depth."""
    h, w = img.shape[:2]
    overlay = img.copy()
    far_y, near_y = int(h * HORIZON_Y), h
    far_half, near_half = int(w * 0.10), int(w * 0.60)   # near edge runs off-screen
    table = np.array([[w // 2 - far_half, far_y], [w // 2 + far_half, far_y],
                      [w // 2 + near_half, near_y], [w // 2 - near_half, near_y]], np.int32)
    cv2.fillPoly(overlay, [table], (90, 50, 10))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.polylines(img, [table], True, WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(img, (w // 2 - far_half, far_y - int(h * 0.12)), (w // 2 + far_half, far_y), WHITE, 2)


def draw_lane_column(img, lane, hand_in_lane):
    """Highlight the screen third the incoming ball is heading for (where the wrist has to be)."""
    h, w = img.shape[:2]
    edges = (0.0, *config.LANE_EDGES, 1.0)
    x0, x1 = int(edges[lane] * w), int(edges[lane + 1] * w)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, int(h * HORIZON_Y)), (x1, h), GREEN if hand_in_lane else WHITE, -1)
    cv2.addWeighted(overlay, 0.14, img, 0.86, 0, img)


def draw_burst(img, center, age):
    """Hit effect: an expanding ring and rays where the ball was hit, fading over BURST_S."""
    f = age / BURST_S
    cx, cy = center
    radius = int(40 + 90 * f)
    thick = max(1, int(8 * (1 - f)))
    cv2.circle(img, center, radius, YELLOW, thick, cv2.LINE_AA)
    for i in range(10):
        a = i * 2 * math.pi / 10
        r0, r1 = radius * 0.6, radius * 1.15
        p0 = (int(cx + r0 * math.cos(a)), int(cy + r0 * math.sin(a)))
        p1 = (int(cx + r1 * math.cos(a)), int(cy + r1 * math.sin(a)))
        cv2.line(img, p0, p1, WHITE, thick, cv2.LINE_AA)


def draw_scene(img, game, now, message=None, swing_flash=False, wrist=None, hit_y=config.HIT_Y_DEFAULT,
               burst=None, fps=None):
    """Draw table, lane, timing ring, ball, wrist and HUD onto img (in place) and return it.
    wrist: a pose_tracker.Wrist in mirrored-screen coordinates, or None.
    hit_y: screen height (0..1) where the ball reaches you (see HitHeight).
    burst: (seconds since the hit, (x, y) where it happened), or None."""
    h, w = img.shape[:2]
    draw_table(img)
    in_lane = wrist is not None and hand_lane(wrist.x, game.lane) == game.lane

    if game.state == INCOMING:
        draw_lane_column(img, game.lane, in_lane)
        # hit circle: where the ball will arrive; lights up green while you can hit it
        (zx, zy), zr = ball_screen(1.0, w, h, game.lane, hit_y)
        in_window = game.in_hit_window(now)
        cv2.circle(img, (zx, zy), zr + 10, GREEN if in_window else (180, 180, 180), 4 if in_window else 2,
                   cv2.LINE_AA)
        # timing ring: closes onto the hit circle exactly when the ball gets there
        tta = game.time_to_arrival(now)
        if 0 < tta < APPROACH_S:
            cv2.circle(img, (zx, zy), int(zr + 10 + tta * APPROACH_PX_PER_S), WHITE, 2, cv2.LINE_AA)

    if game.state in (SERVE, INCOMING, RETURNING, MISSED):
        z = game.ball_z(now)
        (bx, by), br = ball_screen(z, w, h, game.lane, hit_y)
        shadow_y = by + int(br * 1.3)
        cv2.ellipse(img, (bx, shadow_y), (br, max(2, br // 4)), 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), br, ORANGE, -1, cv2.LINE_AA)
        cv2.circle(img, (bx - br // 3, by - br // 3), max(1, br // 4), (200, 230, 255), -1, cv2.LINE_AA)

    if burst is not None and burst[0] < BURST_S:
        draw_burst(img, burst[1], burst[0])

    if wrist is not None:
        color = (255, 255, 0)
        if game.state == INCOMING:
            color = GREEN if in_lane else RED
        center = (int(wrist.x * w), int(wrist.y * h))
        cv2.circle(img, center, 14, BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, center, 11, color, -1, cv2.LINE_AA)

    # HUD
    cv2.putText(img, f"Streak {game.streak}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, BLACK, 6, cv2.LINE_AA)
    cv2.putText(img, f"Streak {game.streak}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, YELLOW, 3, cv2.LINE_AA)
    cv2.putText(img, f"Best {game.best}   Level {game.level}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                WHITE, 2, cv2.LINE_AA)
    if message:
        text, color = message
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
        org = ((w - size[0]) // 2, int(h * 0.38))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, BLACK, 10, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5, cv2.LINE_AA)
    if game.last_offset is not None:
        cv2.putText(img, f"last swing {game.last_offset * 1000:+.0f} ms", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    if fps:
        cv2.putText(img, fps, (w - 230, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if swing_flash:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), YELLOW, 12)
    return img


def draw_start_screen(img, best, tags=(), held_level=None, progress=0.0):
    """The start screen: three columns (easy | medium | hard) and the start tag, if seen.
    tags: 4 corner points (img coordinates) of each start tag seen; held_level/progress: the
    column the tag is being held in and how far through the hold it is (0..1)."""
    h, w = img.shape[:2]
    cv2.addWeighted(img, 0.45, np.zeros_like(img), 0.55, 0, img)   # dim the camera image

    def centered(text, cx, y, scale, color, thick):
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (int(cx - size[0] / 2), y)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 4, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # the three columns, same split as the ball lanes; the one the tag is in lights up
    edges = (0.0, *config.LANE_EDGES, 1.0)
    for level, name in LEVEL_NAMES.items():
        x0, x1 = int(edges[level - 1] * w), int(edges[level] * w)
        if level == held_level:
            overlay = img.copy()
            cv2.rectangle(overlay, (x0, 0), (x1, h), GREEN, -1)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        if level > 1:
            cv2.line(img, (x0, int(h * 0.48)), (x0, h), (200, 200, 200), 2, cv2.LINE_AA)
        centered(name.upper(), (x0 + x1) / 2, int(h * 0.86), 1.2, GREEN if level == held_level else WHITE, 3)

    centered("PING PONG", w / 2, int(h * 0.17), 2.4, ORANGE, 6)
    centered("Hold up the tag in a column to pick the level", w / 2, int(h * 0.29), 0.9, WHITE, 2)
    centered("(or press 1 / 2 / 3)", w / 2, int(h * 0.36), 0.7, (200, 200, 200), 2)
    if best:
        centered(f"Best streak {best}", w / 2, int(h * 0.95), 0.9, YELLOW, 2)

    for corners in tags:
        pts = np.int32(corners)
        cv2.polylines(img, [pts], True, GREEN, 3, cv2.LINE_AA)
        if held_level is not None:
            cx, cy = pts.mean(axis=0).astype(int)
            radius = int(np.linalg.norm(pts[0] - pts[2]) / 2) + 15
            cv2.ellipse(img, (int(cx), int(cy)), (radius, radius), -90, 0, 360 * progress, GREEN, 8, cv2.LINE_AA)
    return img


def blank_background():
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img[:] = np.linspace(70, 15, HEIGHT, dtype=np.uint8)[:, None, None]   # dark gradient
    return img


# ── Inputs ───────────────────────────────────────────────────────────────────

def open_camera(index):
    # DirectShow: the default Windows backend often can't open virtual webcams like Camo
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {index}; using a plain background.")
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always hand over the newest frame, not a queued one
    return cap


class FpsCounter:
    def __init__(self):
        self._t, self._frames, self._cam, self.text = time.perf_counter(), 0, 0, ""

    def tick(self, now, cam_frames):
        self._frames += 1
        if now - self._t >= 1.0:
            cam_fps = (cam_frames - self._cam) / (now - self._t)
            self.text = f"{self._frames / (now - self._t):.0f} fps  (camera {cam_fps:.0f})"
            self._t, self._frames, self._cam = now, 0, cam_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake-imu", action="store_true", help="space bar swings instead of the paddle")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mqtt", action="store_true", help="don't publish the streak")
    parser.add_argument("--fake-pose", action="store_true", help="the mouse is your wrist")
    parser.add_argument("--no-pose", action="store_true", help="don't check where your hand is")
    parser.add_argument("--pose-model", choices=("lite", "full", "heavy"), default=config.POSE_MODEL)
    parser.add_argument("--no-haptics", action="store_true", help="no paddle buzz / light")
    parser.add_argument("--level", type=int, choices=config.LEVEL_TRAVEL_S, help="skip the start screen")
    args = parser.parse_args()

    paddle = detector = None
    from paddle_feedback import NoFeedback
    feedback = NoFeedback()
    if not args.fake_imu:
        from imu_paddle import DoubleMotorPaddle
        paddle = DoubleMotorPaddle()
        paddle.connect()
        detector = SwingDetector()
        if not args.no_haptics:
            from paddle_feedback import PaddleFeedback
            feedback = PaddleFeedback(paddle.motor)
    cap = None if args.no_camera else open_camera(args.camera)

    scoreboard = Scoreboard()
    if not args.no_mqtt:
        scoreboard.connect()

    cv2.namedWindow(WINDOW)
    pose = mouse = None
    if args.fake_pose:
        from pose_tracker import MousePose
        mouse = pose = MousePose(WINDOW, WIDTH, HEIGHT)
    elif not args.no_pose and cap is not None:
        from pose_tracker import PoseTracker
        pose = PoseTracker(model=args.pose_model)
    if pose is None:
        print("Pose is off: the hand's lane isn't checked.")

    camera = None
    if cap is not None:
        from camera_worker import CameraWorker
        from level_tags import TagReader
        camera = CameraWorker(cap, (WIDTH, HEIGHT), pose=None if mouse else pose, tag_reader=TagReader())

    picker = LevelPicker()
    game, best = None, 0     # game is None on the start screen

    def new_game(level, now):
        g = Game(level=level)
        g.best = best
        g.start(now)
        print(f"Start: level {level} ({LEVEL_NAMES[level]})")
        return g

    if args.level:
        game = new_game(args.level, time.perf_counter())
    message, message_until, flash_until = None, 0.0, 0.0
    burst_t, burst_at = -math.inf, (0, 0)
    hit_height = HitHeight()
    fps = FpsCounter()
    last_frame_t = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()
            frame, tags, cam_frames = camera.latest() if camera else (None, [], 0)
            if frame is None:
                frame = blank_background()
            if camera:
                camera.track_pose = game is not None
                camera.find_tags = game is None
            if mouse is not None:
                mouse.process(None, loop_start)

            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            fps.tick(now, cam_frames)
            if key == ord("q"):
                break
            if key == ord("r"):
                game = None
                picker.reset()

            # swings (always read, so old ones don't pile up while on the start screen)
            swing_times = []
            if paddle is not None:
                swing_times = [s.t for s in detector.update_many(paddle.read())]
            elif key == ord(" "):
                swing_times = [now]

            if game is None:
                level = picker.update([sum(x for x, _ in c) / 4 / WIDTH for c in tags], now)
                if key in (ord("1"), ord("2"), ord("3")):
                    level = key - ord("0")
                if level:
                    game = new_game(level, now)
                    picker.reset()
                else:
                    draw_start_screen(frame, best, tags, picker.level, picker.progress(now))
                    cv2.imshow(WINDOW, frame)
                    last_frame_t = now
                    continue

            events = []
            for t in swing_times:
                lane = ANY_LANE
                if pose is not None:
                    wrist = pose.history.at(t)
                    lane = hand_lane(wrist.x, game.lane) if wrist else None
                events += game.swing(t, lane)
                flash_until = now + SWING_FLASH_S
            events += game.update(now)
            scoreboard.update(game.streak)   # publishes only when it changed
            best = game.best

            for e in events:
                if e == "miss":
                    message = (MISS_MESSAGES[game.miss_reason], RED)
                else:
                    message = MESSAGES[e]
                message_until = now + MESSAGE_S
                feedback.send("whiff" if e.startswith("whiff") else e)   # buzz + light on the paddle
                if e == "hit":
                    burst_t, burst_at = now, ball_screen(1.0, WIDTH, HEIGHT, game.lane, hit_height.y)[0]
                judged = e == "hit" or e.startswith("whiff")
                timing = f"  swing {game.last_offset * 1000:+5.0f} ms" if judged else ""
                print(f"{e:13s} streak {game.streak}  best {game.best}{timing}")

            wrist = pose.history.latest() if pose is not None else None
            if wrist is not None and now - wrist.t > config.WRIST_MAX_AGE_S:
                wrist = None   # stale: the camera lost the wrist
            # the hit circle follows the wrist's height, but holds still once the ball is on its way in
            locked = game.state == INCOMING and game.ball_z(now) >= config.WHIFF_FROM_Z
            hit_y = hit_height.update(wrist.y if wrist else None, now - last_frame_t, locked)
            last_frame_t = now
            draw_scene(frame, game, now, message if now < message_until else None, now < flash_until, wrist,
                       hit_y, (now - burst_t, burst_at), fps.text)
            cv2.imshow(WINDOW, frame)

            spare = 1 / TARGET_FPS - (time.perf_counter() - loop_start)
            if spare > 0:
                time.sleep(spare)
    finally:
        feedback.stop()
        if camera is not None:
            camera.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if paddle is not None:
            paddle.disconnect()
        if pose is not None:
            pose.close()
        scoreboard.close()


if __name__ == "__main__":
    main()
