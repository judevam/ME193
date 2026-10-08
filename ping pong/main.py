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

The game opens on a start screen: hold up the start tag (python level_tags.py --make prints it)
in the left / middle / right of the screen to start easy / medium / hard. Keys 1/2/3 do the same.

The current streak is published to config.MQTT_TOPIC whenever it changes.
Each ball comes down the left, center or right lane: swing with your wrist in that lane (the dot
shows where the game thinks your wrist is: green = right lane, red = wrong lane).

Keys: space = swing (with --fake-imu), 1/2/3 = start a level, r = back to the start screen, q = quit.
"""

import argparse
import time

import cv2
import numpy as np

import config
from game import Game, SERVE, INCOMING, RETURNING, MISSED, ANY_LANE
from level_tags import LEVEL_NAMES, LevelPicker
from pose_tracker import lane_of
from scoreboard import Scoreboard
from swing import SwingDetector

WINDOW = "Ping Pong (q to quit)"
WIDTH, HEIGHT = 960, 720

# Perspective: the wall is a small rectangle near the top, you are at the bottom edge.
HORIZON_Y = 0.22      # fraction of the height where the wall is
NEAR_Y = 0.80         # where the ball is when it reaches you (hit zone)
BALL_R_FAR, BALL_R_NEAR = 6, 46

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, ORANGE, YELLOW = (60, 220, 60), (60, 60, 230), (0, 140, 255), (0, 230, 255)

MESSAGES = {"hit": ("HIT!", GREEN), "miss_early": ("MISS - too early", RED),
            "miss_late": ("MISS - too late", RED), "miss_lane": ("MISS - wrong spot", RED),
            "miss_no_hand": ("MISS - hand not seen", RED), "serve": ("Serve!", WHITE)}
MESSAGE_S = 0.8
SWING_FLASH_S = 0.15


# ── Drawing (no camera or BLE, so it can be tested) ──────────────────────────

def lane_x(lane):
    """Screen x (0..1) of a lane's centre where the ball reaches you: the middle of that third."""
    edges = (0.0, *config.LANE_EDGES, 1.0)
    return (edges[lane] + edges[lane + 1]) / 2


def ball_screen(z, w, h, lane=1):
    """Ball centre and radius on screen for distance z (0 wall .. 1 you, a bit over when missed).
    It leaves the middle of the wall and spreads out to its lane as it comes closer."""
    p = z ** 1.6   # perspective-ish: the ball speeds up on screen as it gets close
    x = w * (0.5 + (lane_x(lane) - 0.5) * p)
    y = h * (HORIZON_Y + (NEAR_Y - HORIZON_Y) * p)
    r = BALL_R_FAR + (BALL_R_NEAR - BALL_R_FAR) * p
    return (int(x), int(y)), max(2, int(r))


def draw_table(img):
    """Translucent table/court lines that give the scene depth."""
    h, w = img.shape[:2]
    overlay = img.copy()
    far_y, near_y = int(h * HORIZON_Y), h
    far_half, near_half = int(w * 0.10), int(w * 0.60)   # near edge runs off-screen
    table = np.array([[w // 2 - far_half, far_y], [w // 2 + far_half, far_y],
                      [w // 2 + near_half, near_y], [w // 2 - near_half, near_y]], np.int32)
    cv2.fillPoly(overlay, [table], (90, 50, 10))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.polylines(img, [table], True, WHITE, 2, cv2.LINE_AA)
    # lane dividers, converging on the wall
    for edge in config.LANE_EDGES:
        far_x = w // 2 + int((edge - 0.5) * 2 * far_half)
        cv2.line(img, (far_x, far_y), (int(edge * w), near_y), (200, 200, 200), 1, cv2.LINE_AA)
    # the wall
    cv2.rectangle(img, (w // 2 - far_half, far_y - int(h * 0.12)), (w // 2 + far_half, far_y), WHITE, 2)


def draw_scene(img, game, now, message=None, swing_flash=False, wrist=None):
    """Draw table, hit zone, ball, wrist and HUD onto img (in place) and return it.
    wrist: a pose_tracker.Wrist in mirrored-screen coordinates, or None."""
    h, w = img.shape[:2]
    draw_table(img)

    # hit zone: where the incoming ball will arrive; lights up green while you can hit it
    if game.state == INCOMING:
        (zx, zy), zr = ball_screen(1.0, w, h, game.lane)
        in_window = game.in_hit_window(now)
        cv2.circle(img, (zx, zy), zr + 10, GREEN if in_window else (180, 180, 180), 4 if in_window else 2,
                   cv2.LINE_AA)

    if game.state in (SERVE, INCOMING, RETURNING, MISSED):
        z = game.ball_z(now)
        (bx, by), br = ball_screen(z, w, h, game.lane)
        shadow_y = by + int(br * 1.3)
        cv2.ellipse(img, (bx, shadow_y), (br, max(2, br // 4)), 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), br, ORANGE, -1, cv2.LINE_AA)
        cv2.circle(img, (bx - br // 3, by - br // 3), max(1, br // 4), (200, 230, 255), -1, cv2.LINE_AA)

    if wrist is not None:
        color = (255, 255, 0)
        if game.state == INCOMING:
            color = GREEN if lane_of(wrist.x) == game.lane else RED
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
        org = ((w - size[0]) // 2, int(h * 0.55))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, BLACK, 10, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5, cv2.LINE_AA)
    if game.last_offset is not None:
        cv2.putText(img, f"last swing {game.last_offset * 1000:+.0f} ms", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
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
    return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake-imu", action="store_true", help="space bar swings instead of the paddle")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mqtt", action="store_true", help="don't publish the streak")
    parser.add_argument("--fake-pose", action="store_true", help="the mouse is your wrist")
    parser.add_argument("--no-pose", action="store_true", help="don't check where your hand is")
    parser.add_argument("--pose-model", choices=("lite", "full", "heavy"), default=config.POSE_MODEL)
    parser.add_argument("--level", type=int, choices=config.LEVEL_TRAVEL_S, help="skip the start screen")
    args = parser.parse_args()

    paddle = detector = None
    if not args.fake_imu:
        from imu_paddle import DoubleMotorPaddle
        paddle = DoubleMotorPaddle()
        paddle.connect()
        detector = SwingDetector()
    cap = None if args.no_camera else open_camera(args.camera)

    scoreboard = Scoreboard()
    if not args.no_mqtt:
        scoreboard.connect()

    tag_reader = None
    if cap is not None:
        from level_tags import TagReader
        tag_reader = TagReader()
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
    cv2.namedWindow(WINDOW)

    pose = None
    if args.fake_pose:
        from pose_tracker import MousePose
        pose = MousePose(WINDOW, WIDTH, HEIGHT)
    elif not args.no_pose and cap is not None:
        from pose_tracker import PoseTracker
        pose = PoseTracker(model=args.pose_model)
    if pose is None:
        print("Pose is off: the hand's lane isn't checked.")
    try:
        while True:
            frame, tags = None, []
            if cap is not None:
                ok, raw = cap.read()
                if ok:
                    if pose is not None and game is not None:
                        pose.process(raw, time.perf_counter())   # un-mirrored: keeps left/right right
                    if game is None:
                        rh, rw = raw.shape[:2]
                        for tag in tag_reader.detect(raw):        # mirror the corners to match the display
                            if tag.id == config.START_TAG_ID:
                                tags.append([((rw - x) * WIDTH / rw, y * HEIGHT / rh) for x, y in tag.corners])
                    frame = cv2.resize(cv2.flip(raw, 1), (WIDTH, HEIGHT))   # mirror, like a mirror
            if frame is None:
                frame = blank_background()
                if pose is not None and cap is None:
                    pose.process(None, time.perf_counter())   # mouse stand-in needs no frame

            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
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
                    continue
            events = []
            for t in swing_times:
                hand_lane = ANY_LANE
                if pose is not None:
                    wrist = pose.history.at(t)
                    hand_lane = lane_of(wrist.x) if wrist else None
                events += game.swing(t, hand_lane)
                flash_until = now + SWING_FLASH_S
            events += game.update(now)
            scoreboard.update(game.streak)   # publishes only when it changed
            best = game.best

            for e in events:
                message, message_until = MESSAGES[e], now + MESSAGE_S
                judged = e in ("hit", "miss_early", "miss_lane", "miss_no_hand")
                timing = f"  swing {game.last_offset * 1000:+5.0f} ms" if judged else ""
                print(f"{e:12s} streak {game.streak}  best {game.best}{timing}")

            wrist = pose.history.latest() if pose is not None else None
            draw_scene(frame, game, now, message if now < message_until else None, now < flash_until, wrist)
            cv2.imshow(WINDOW, frame)
    finally:
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
