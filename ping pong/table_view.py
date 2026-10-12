"""Wii Sports Resort-style 3D table view. Pure drawing: no camera, BLE or MQTT.

A virtual camera stands behind and above the player and looks down a regulation table. Everything
is placed in metres in a world frame and projected with a pinhole camera:

    x: right (+) / left (-) as the player sees it (lane 0 is left)
    y: up from the floor
    z: away from the player; the near table edge is z = 0, the far edge z = TABLE_L,
       and the wall the player rallies against is at z = WALL_Z

The game logic only knows the ball's progress `z` (0 = wall, 1 = you). `ball_world()` turns that
into a 3D arc: off the wall, one bounce on the player's half, up to the hit point. After a hit it
goes back the other way with a bounce on the far half. A missed ball keeps going and drops.
"""

import math

import cv2
import numpy as np

import config
from game import SERVE, INCOMING, RETURNING, MISSED

# ── The world (metres) ───────────────────────────────────────────────────────
TABLE_L, TABLE_W, TABLE_Y = 2.74, 1.525, 0.76      # regulation size and height
NET_Z, NET_H, NET_OVERHANG = TABLE_L / 2, 0.1525, 0.15
WALL_Z = 3.4                                       # the wall the ball rebounds from
BALL_R = 0.03                                      # drawn a bit bigger than a real 2 cm ball
HIT_Z, HIT_Y = -0.25, 1.0                          # where you meet the ball
LANE_X = (-0.5, 0.0, 0.5)                          # ball's hit point for lanes 0, 1, 2
WALL_HIT_Y = 0.95                                  # height the ball leaves / reaches the wall
BOUNCE_IN_Z = 0.55                                 # incoming ball bounces here (your half)
BOUNCE_OUT_Z = 2.1                                 # returned ball bounces here (far half)
ARC_DOWN, ARC_UP = 0.30, 0.22                      # arc heights of the two halves of a flight
PADDLE_R = 0.085
PADDLE_MAX_TILT = 28.0        # degrees of roll at the outer lanes (handle angles back toward you)
PADDLE_MAX_TURN = 0.30        # how edge-on the face turns at the outer lanes (0 = never)
SWING_ANIM_S = 0.28           # length of the swing animation
SWING_PUSH, SWING_LIFT = 0.35, 0.05     # metres the paddle lunges forward / rises mid-swing
SWING_SWEEP_X, SWING_SWEEP_DEG = 0.10, 40.0   # sideways sweep toward the middle, and the roll with it

# The virtual camera: behind and above the player, looking down the table
EYE = np.array([0.0, 1.55, -1.45])
LOOK_AT = np.array([0.0, 0.72, 1.65])
FOCAL = 0.95                                       # x image width

# ── Colours (BGR) ────────────────────────────────────────────────────────────
SKY_TOP, SKY_LOW = (235, 190, 120), (250, 230, 200)
FLOOR_FAR, FLOOR_NEAR = (120, 170, 200), (70, 115, 150)    # warm wooden court
WALL_FACE, WALL_TRIM = (215, 205, 190), (160, 90, 30)
TABLE_TOP, TABLE_EDGE, TABLE_LEG = (150, 85, 25), (95, 50, 15), (60, 60, 60)
LINE, NET_MESH, NET_BAND = (250, 250, 250), (50, 50, 50), (245, 245, 245)
BALL, BALL_SHINE, SHADOW = (20, 140, 255), (200, 235, 255), (60, 40, 15)   # orange ball
RUBBER, RUBBER_RIM, HANDLE = (40, 40, 200), (20, 20, 20), (70, 140, 200)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, ORANGE, YELLOW = (60, 220, 60), (60, 60, 230), (0, 140, 255), (0, 230, 255)

FONT = cv2.FONT_HERSHEY_DUPLEX


class Camera:
    """Pinhole camera looking from `eye` toward `target` (y is up)."""

    def __init__(self, eye, target, w, h, focal=FOCAL):
        self.eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - self.eye
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross([0.0, 1.0, 0.0], self.fwd)
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.fwd, self.right)
        self.f = focal * w
        self.cx, self.cy = w / 2, h / 2

    def depth(self, p):
        return float(np.dot(np.asarray(p, float) - self.eye, self.fwd))

    def project(self, p):
        """World point -> (u, v) pixel coordinates (floats)."""
        d = np.asarray(p, float) - self.eye
        zc = max(np.dot(d, self.fwd), 1e-3)
        return (self.cx + self.f * np.dot(d, self.right) / zc,
                self.cy - self.f * np.dot(d, self.up) / zc)

    def pts(self, points):
        """Several world points -> int32 polygon for cv2."""
        return np.array([self.project(p) for p in points], np.float32).round().astype(np.int32)

    def radius(self, p, r):
        """On-screen radius of a sphere of radius r at p."""
        return self.f * r / max(self.depth(p), 1e-3)


# ── Where things are ─────────────────────────────────────────────────────────

def paddle_x(wrist_x):
    """Mirrored-screen wrist x (0..1) -> paddle x in metres. Lane centres line up with LANE_X:
    the middle of the left third (1/6) maps to -0.5, the middle (1/2) to 0."""
    return (wrist_x - 0.5) * 3.0 * LANE_X[2]


def paddle_pose(x, swing_age=None):
    """Where the paddle is and how it's turned, for a paddle at x (metres) and the time since the
    last swing (None = no swing). Returns ((x, y, z) of the face centre, roll in degrees, face width
    factor). Roll > 0 leans the top to the right; the paddle rolls more the further out it is, so
    the handle always angles back toward the player. During a swing it lunges forward, rises, and
    sweeps toward the middle (forehand on the right, backhand on the left), then comes back."""
    reach = max(-1.0, min(1.0, x / LANE_X[2]))           # -1 left lane .. 0 middle .. +1 right lane
    roll = PADDLE_MAX_TILT * reach
    width = 1.0 - PADDLE_MAX_TURN * abs(reach)
    y, z = HIT_Y, HIT_Z - 0.02
    if swing_age is not None and 0 <= swing_age < SWING_ANIM_S:
        k = math.sin(math.pi * swing_age / SWING_ANIM_S)  # 0 -> 1 -> 0 over the swing
        side = 1.0 if x >= 0 else -1.0
        x -= side * SWING_SWEEP_X * k
        y += SWING_LIFT * k
        z += SWING_PUSH * k
        roll -= side * SWING_SWEEP_DEG * k
    return (x, y, z), roll, width


def _arc(a, b, height, u):
    """Height along a throw from a to b (u 0..1) with a hump of `height`; keeps falling past u=1."""
    return a + (b - a) * u + 4 * height * u * (1 - u)


def ball_world(game, now):
    """The ball's (x, y, z) in metres for the current game state, or None before the game starts."""
    if game.state is None:
        return None
    s = game.ball_z(now)                      # 0 = wall, 1 = you (over 1 = gone past you)
    lane_x = LANE_X[game.lane]
    table_top = TABLE_Y + BALL_R
    if game.state in (SERVE, INCOMING, MISSED):
        z = WALL_Z + (HIT_Z - WALL_Z) * s
        x = lane_x * s
        s_bounce = (WALL_Z - BOUNCE_IN_Z) / (WALL_Z - HIT_Z)
        if s < s_bounce:
            y = _arc(WALL_HIT_Y, table_top, ARC_DOWN, s / s_bounce)
        else:
            y = _arc(table_top, HIT_Y, ARC_UP, (s - s_bounce) / (1 - s_bounce))
        return x, max(y, BALL_R), z
    # RETURNING: q = 0 at your paddle, 1 at the wall
    q = 1.0 - s
    z = HIT_Z + (WALL_Z - HIT_Z) * q
    x = lane_x * (1 - q)
    q_bounce = (BOUNCE_OUT_Z - HIT_Z) / (WALL_Z - HIT_Z)
    if q < q_bounce:
        y = _arc(HIT_Y, table_top, ARC_UP, q / q_bounce)
    else:
        y = _arc(table_top, WALL_HIT_Y, ARC_DOWN, (q - q_bounce) / (1 - q_bounce))
    return x, y, z


def over_table(x, z):
    return abs(x) <= TABLE_W / 2 and 0 <= z <= TABLE_L


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _text(img, text, org, scale, color, thick, outline=BLACK):
    cv2.putText(img, text, org, FONT, scale, outline, thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _centered(img, text, cx, y, scale, color, thick):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    _text(img, text, (int(cx - tw / 2), int(y)), scale, color, thick)


def _blend_poly(img, poly, color, alpha):
    """Translucent filled polygon, blended only inside its bounding box (cheap)."""
    x, y, w, h = cv2.boundingRect(poly)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.fillPoly(layer, [poly - [x0, y0]], color, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def _rounded_panel(img, x0, y0, x1, y1, alpha=0.55):
    roi = img[y0:y1, x0:x1]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, roi)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)


class TableView:
    """Renders the 3D table scene. The background (sky, floor, wall, table) is drawn once and
    reused; each frame adds the ball, shadow, net, paddle and HUD on a copy of it."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cam = Camera(EYE, LOOK_AT, w, h)
        self.static = self._draw_static()

    # The parts that never move
    def _draw_static(self):
        w, h, cam = self.w, self.h, self.cam
        img = np.zeros((h, w, 3), np.uint8)
        horizon = int(cam.project((0, 0, 60))[1])
        for y in range(h):                                   # sky, then floor
            if y < horizon:
                t = y / max(horizon, 1)
                img[y] = np.add(np.multiply(SKY_TOP, 1 - t), np.multiply(SKY_LOW, t))
            else:
                t = (y - horizon) / max(h - horizon, 1)
                img[y] = np.add(np.multiply(FLOOR_FAR, 1 - t), np.multiply(FLOOR_NEAR, t))
        # floor boards, running away from the player
        for bx in np.arange(-4.0, 4.01, 0.5):
            cv2.line(img, tuple(cam.pts([(bx, 0, -0.5)])[0]), tuple(cam.pts([(bx, 0, 30)])[0]),
                     np.subtract(FLOOR_NEAR, 18).tolist(), 1, cv2.LINE_AA)
        # the wall: a backboard with a trim stripe and a target ring
        wall = cam.pts([(-1.8, 0, WALL_Z), (1.8, 0, WALL_Z), (1.8, 1.8, WALL_Z), (-1.8, 1.8, WALL_Z)])
        cv2.fillPoly(img, [wall], WALL_FACE, cv2.LINE_AA)
        trim = cam.pts([(-1.8, 0, WALL_Z), (1.8, 0, WALL_Z), (1.8, 0.35, WALL_Z), (-1.8, 0.35, WALL_Z)])
        cv2.fillPoly(img, [trim], WALL_TRIM, cv2.LINE_AA)
        cv2.polylines(img, [wall], True, WALL_TRIM, 3, cv2.LINE_AA)
        c = tuple(int(v) for v in cam.project((0, WALL_HIT_Y, WALL_Z)))
        rad = int(cam.radius((0, WALL_HIT_Y, WALL_Z), 0.35))
        for k, col in ((1.0, WALL_TRIM), (0.62, WALL_FACE), (0.3, WALL_TRIM)):
            cv2.circle(img, c, int(rad * k), col, -1, cv2.LINE_AA)
        # legs (dark posts under the four corners, slightly inset)
        for lx in (-0.6, 0.6):
            for lz in (0.25, TABLE_L - 0.25):
                leg = cam.pts([(lx - 0.03, 0, lz), (lx + 0.03, 0, lz), (lx + 0.03, TABLE_Y, lz),
                               (lx - 0.03, TABLE_Y, lz)])
                cv2.fillPoly(img, [leg], TABLE_LEG, cv2.LINE_AA)
        # table top, front edge, lines
        hw = TABLE_W / 2
        top = cam.pts([(-hw, TABLE_Y, 0), (hw, TABLE_Y, 0), (hw, TABLE_Y, TABLE_L), (-hw, TABLE_Y, TABLE_L)])
        edge = cam.pts([(-hw, TABLE_Y, 0), (hw, TABLE_Y, 0), (hw, TABLE_Y - 0.04, 0), (-hw, TABLE_Y - 0.04, 0)])
        cv2.fillPoly(img, [edge], TABLE_EDGE, cv2.LINE_AA)
        cv2.fillPoly(img, [top], TABLE_TOP, cv2.LINE_AA)
        inset = 0.02
        border = cam.pts([(-hw + inset, TABLE_Y, inset), (hw - inset, TABLE_Y, inset),
                          (hw - inset, TABLE_Y, TABLE_L - inset), (-hw + inset, TABLE_Y, TABLE_L - inset)])
        cv2.polylines(img, [border], True, LINE, 2, cv2.LINE_AA)
        cv2.line(img, *[tuple(p) for p in cam.pts([(0, TABLE_Y, inset), (0, TABLE_Y, TABLE_L - inset)])],
                 LINE, 1, cv2.LINE_AA)
        return img

    def _draw_net(self, img):
        cam, half = self.cam, TABLE_W / 2 + NET_OVERHANG
        bottom, top = TABLE_Y, TABLE_Y + NET_H
        for x in np.arange(-half, half + 1e-6, 0.06):          # mesh: vertical strands
            a, b = cam.pts([(x, bottom, NET_Z), (x, top, NET_Z)])
            cv2.line(img, tuple(a), tuple(b), NET_MESH, 1, cv2.LINE_AA)
        for y in np.arange(bottom, top, 0.03):                 # and horizontal ones
            a, b = cam.pts([(-half, y, NET_Z), (half, y, NET_Z)])
            cv2.line(img, tuple(a), tuple(b), NET_MESH, 1, cv2.LINE_AA)
        band = cam.pts([(-half, top - 0.015, NET_Z), (half, top - 0.015, NET_Z), (half, top, NET_Z),
                        (-half, top, NET_Z)])
        cv2.fillPoly(img, [band], NET_BAND, cv2.LINE_AA)
        for x in (-half, half):                                # posts
            a, b = cam.pts([(x, TABLE_Y - 0.02, NET_Z), (x, top, NET_Z)])
            cv2.line(img, tuple(a), tuple(b), NET_MESH, 4, cv2.LINE_AA)

    def _draw_ball(self, img, pos):
        x, y, z = pos
        cam = self.cam
        # shadow straight below, on the table if it's over it, else on the floor
        ground = TABLE_Y if over_table(x, z) and y >= TABLE_Y else 0.0
        sp = cam.project((x, ground, z))
        sr = cam.radius((x, ground, z), BALL_R * 1.2)
        fade = max(0.0, 1.0 - (y - ground) / 1.2)               # fainter the higher the ball is
        if sr >= 1 and fade > 0:
            ell = cv2.ellipse2Poly((int(sp[0]), int(sp[1])), (int(sr * 1.3), max(1, int(sr * 0.5))),
                                   0, 0, 360, 15)
            _blend_poly(img, ell, SHADOW, 0.55 * fade)
        c = cam.project(pos)
        r = max(2, int(round(cam.radius(pos, BALL_R))))
        center = (int(c[0]), int(c[1]))
        cv2.circle(img, center, r + 1, (0, 70, 140), -1, cv2.LINE_AA)
        cv2.circle(img, center, r, BALL, -1, cv2.LINE_AA)
        cv2.circle(img, (center[0] - r // 3, center[1] - r // 3), max(1, r // 3), BALL_SHINE, -1, cv2.LINE_AA)

    def _paddle_shape(self, x, swing_age):
        """Screen centre, axes (half-width, half-height) of the face, roll, and handle polygon."""
        face, roll, width = paddle_pose(x, swing_age)
        c = self.cam.project(face)
        r = self.cam.radius(face, PADDLE_R)
        axes = (max(2, int(r * width)), max(2, int(r * 0.92)))
        # handle in the paddle's own frame (pixels, y down), then rolled with the face
        hw0, hw1, top, bottom = r * 0.17, r * 0.22, r * 0.75, r * 2.25
        a = math.radians(roll)
        cos, sin = math.cos(a), math.sin(a)
        handle = np.array([[c[0] + px * cos - py * sin, c[1] + px * sin + py * cos]
                           for px, py in ((-hw0, top), (hw0, top), (hw1, bottom), (-hw1, bottom))]).astype(np.int32)
        return (int(c[0]), int(c[1])), axes, roll, handle

    def _draw_paddle(self, img, x, highlight, swing_age=None):
        # motion trail: faint copies a moment behind, during the forward stroke only
        if swing_age is not None and 0 <= swing_age < SWING_ANIM_S * 0.55:
            for lag, alpha in ((0.10, 0.18), (0.05, 0.30)):
                if swing_age - lag > 0:
                    center, axes, roll, _ = self._paddle_shape(x, swing_age - lag)
                    _blend_poly(img, cv2.ellipse2Poly(center, axes, int(roll), 0, 360, 10), RUBBER, alpha)
        center, (ax, ay), roll, handle = self._paddle_shape(x, swing_age)
        cv2.fillPoly(img, [handle], HANDLE, cv2.LINE_AA)
        cv2.polylines(img, [handle], True, BLACK, 1, cv2.LINE_AA)
        cv2.ellipse(img, center, (ax, ay), roll, 0, 360, RUBBER_RIM, -1, cv2.LINE_AA)
        cv2.ellipse(img, center, (max(1, ax - 3), max(1, ay - 3)), roll, 0, 360, RUBBER, -1, cv2.LINE_AA)
        # a soft highlight across the rubber so it reads as a surface, not a flat disc
        cv2.ellipse(img, center, (max(1, int(ax * 0.55)), max(1, int(ay * 0.35))), roll - 25, 200, 300,
                    (120, 120, 255), 2, cv2.LINE_AA)
        if highlight:
            cv2.ellipse(img, center, (ax + 5, ay + 5), roll, 0, 360, GREEN, 3, cv2.LINE_AA)

    def _draw_lane(self, img, lane, in_lane):
        """Highlight the strip of your half of the table the ball is heading down."""
        edges = (-TABLE_W / 2, (LANE_X[0] + LANE_X[1]) / 2, (LANE_X[1] + LANE_X[2]) / 2, TABLE_W / 2)
        x0, x1 = edges[lane], edges[lane + 1]
        strip = self.cam.pts([(x0, TABLE_Y, 0), (x1, TABLE_Y, 0), (x1, TABLE_Y, NET_Z), (x0, TABLE_Y, NET_Z)])
        _blend_poly(img, strip, GREEN if in_lane else WHITE, 0.22)

    def _draw_target(self, img, game, now):
        """Ring where you meet the ball: green in the hit window; a timing ring closes onto it."""
        p = (LANE_X[game.lane], HIT_Y, HIT_Z)
        c = self.cam.project(p)
        center = (int(c[0]), int(c[1]))
        r = int(self.cam.radius(p, PADDLE_R * 1.25))
        in_window = game.in_hit_window(now)
        cv2.circle(img, center, r, GREEN if in_window else (230, 230, 230), 4 if in_window else 2, cv2.LINE_AA)
        tta = game.time_to_arrival(now)
        if tta is not None and 0 < tta < 0.8:
            cv2.circle(img, center, int(r + tta * 260), WHITE, 2, cv2.LINE_AA)

    def _draw_burst(self, img, lane, age, length=0.3):
        if age >= length:
            return
        c = self.cam.project((LANE_X[lane], HIT_Y, HIT_Z))
        cx, cy = int(c[0]), int(c[1])
        f = age / length
        radius = int(45 + 110 * f)
        thick = max(1, int(8 * (1 - f)))
        cv2.circle(img, (cx, cy), radius, YELLOW, thick, cv2.LINE_AA)
        for i in range(12):
            a = i * 2 * math.pi / 12
            p0 = (int(cx + radius * 0.6 * math.cos(a)), int(cy + radius * 0.6 * math.sin(a)))
            p1 = (int(cx + radius * 1.2 * math.cos(a)), int(cy + radius * 1.2 * math.sin(a)))
            cv2.line(img, p0, p1, WHITE, thick, cv2.LINE_AA)

    def _draw_inset(self, img, cam_frame, wrist, lane=None, in_lane=False, aids=True):
        """The webcam, small, top-right: so you can still see yourself and where your wrist is."""
        iw, ih, m = self.w // 4, self.h // 4, 12
        x0, y0 = self.w - iw - m, m
        if cam_frame is None:
            small = np.full((ih, iw, 3), 40, np.uint8)
            _centered(small, "no camera", iw / 2, ih / 2, 0.5, WHITE, 1)
        else:
            small = cv2.resize(cam_frame, (iw, ih), interpolation=cv2.INTER_AREA)
        if lane is not None:
            edges = (0.0, *config.LANE_EDGES, 1.0)
            _blend_poly(small, np.array([[edges[lane] * iw, 0], [edges[lane + 1] * iw, 0],
                                         [edges[lane + 1] * iw, ih], [edges[lane] * iw, ih]], np.int32),
                        GREEN if in_lane else WHITE, 0.2)
        for e in config.LANE_EDGES:
            cv2.line(small, (int(e * iw), 0), (int(e * iw), ih), WHITE, 1, cv2.LINE_AA)
        if wrist is not None:
            pt = (int(wrist.x * iw), int(wrist.y * ih))
            cv2.circle(small, pt, 7, BLACK, -1, cv2.LINE_AA)
            cv2.circle(small, pt, 5, GREEN if in_lane and aids else (255, 255, 0), -1, cv2.LINE_AA)
        img[y0:y0 + ih, x0:x0 + iw] = small
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + iw + 1, y0 + ih + 1), WHITE, 3, cv2.LINE_AA)
        _text(img, "YOU", (x0 + 8, y0 + 22), 0.6, WHITE, 1)

    def _draw_hud(self, img, game, message, fps):
        _rounded_panel(img, 14, 14, 214, 132)
        _text(img, "STREAK", (28, 42), 0.6, (220, 220, 220), 1)
        _text(img, str(game.streak), (28, 98), 2.0, YELLOW, 4)
        _text(img, f"Best {game.best}   Lv {game.level}", (28, 122), 0.55, WHITE, 1)
        if game.speed > 1.005:
            _text(img, f"Speed {game.speed * 100:.0f}%", (224, 122), 0.6, ORANGE, 2)
        if game.last_offset is not None:
            _text(img, f"last swing {game.last_offset * 1000:+.0f} ms", (18, 158), 0.5, WHITE, 1)
        if message:
            text, color = message
            _centered(img, text, self.w / 2, self.h * 0.40, 1.6, color, 4)
        if fps:
            _text(img, fps, (self.w - 230, self.h - 14), 0.5, (230, 230, 230), 1)

    # The frame
    def render(self, game, now, *, wrist=None, wrist_lane=None, cam_frame=None, message=None,
               swing_flash=False, burst=None, fps=None, swing_age=None, aids=True):
        """Draw one frame. wrist: pose_tracker.Wrist (mirrored screen coords) or None;
        wrist_lane: the lane the wrist counts as being in (with the edge tolerance), or None;
        burst: (seconds since the hit, lane) or None; swing_age: seconds since the last swing;
        aids: draw the helpers (target + timing ring, lane strip, in-lane glow) - off on hard."""
        img = self.static.copy()
        in_lane = wrist_lane is not None and wrist_lane == game.lane
        incoming = game.state == INCOMING
        helped = incoming and aids

        if helped:
            self._draw_lane(img, game.lane, in_lane)
        pos = ball_world(game, now)
        if pos is not None and self.cam.depth(pos) < 0.3:
            pos = None                     # a missed ball that has flown past the camera
        behind_net = pos is not None and pos[2] > NET_Z
        if pos is not None and behind_net:
            self._draw_ball(img, pos)
        self._draw_net(img)
        if helped:
            self._draw_target(img, game, now)
        if wrist is not None:
            self._draw_paddle(img, paddle_x(wrist.x), helped and in_lane, swing_age)
        if pos is not None and not behind_net:
            self._draw_ball(img, pos)
        if burst is not None:
            self._draw_burst(img, burst[1], burst[0])

        self._draw_inset(img, cam_frame, wrist, game.lane if helped else None, in_lane, aids)
        self._draw_hud(img, game, message, fps)
        if swing_flash:
            cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), YELLOW, 10)
        return img
