"""Where is the hitting-hand wrist? (mediapipe Pose, same setup as poserace/)

    python pose_tracker.py              # webcam with the wrist dot and its lane
    python pose_tracker.py --camera 1
    python pose_tracker.py --model full # compare models: lite / full / heavy
    python pose_tracker.py --raw        # turn smoothing off (see how much it helps)

The top line shows ms per frame and how often the wrist was seen over the last ~2 s: use
these to pick POSE_MODEL / WRIST_SMOOTHING in config.py.

Positions are in the MIRRORED screen's coordinates (x 0 = left edge, 1 = right edge), because
that's what the game shows. Detection runs on the un-mirrored frame so mediapipe's left/right
labels stay correct, then x is flipped.
"""

import argparse
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import config

LEFT, CENTER, RIGHT = 0, 1, 2
LANE_NAMES = ("left", "center", "right")

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{0}/float16/latest/pose_landmarker_{0}.task")
POSE_MODELS = ("lite", "full", "heavy")
WRIST_INDEX = {"left": 15, "right": 16}


@dataclass
class Wrist:
    t: float
    x: float   # 0..1 across the mirrored screen
    y: float   # 0..1 down the screen


def lane_of(x):
    """Lane for a mirrored-screen x position."""
    lo, hi = config.LANE_EDGES
    return LEFT if x < lo else RIGHT if x > hi else CENTER


def hand_lane(x, ball_lane, tolerance=config.LANE_TOLERANCE):
    """The hand's lane, giving the benefit of the doubt near an edge: a hand within `tolerance`
    of the ball's lane counts as in it. Stops a near-the-line swing from being "wrong spot"."""
    edges = (0.0, *config.LANE_EDGES, 1.0)
    if edges[ball_lane] - tolerance <= x <= edges[ball_lane + 1] + tolerance:
        return ball_lane
    return lane_of(x)


class Smoother:
    """Exponential smoothing of the wrist position. Starts fresh after the wrist is lost."""

    def __init__(self, amount=config.WRIST_SMOOTHING, reset_after=config.WRIST_MAX_AGE_S):
        self.amount = amount              # fraction of the way to move toward each new position
        self.reset_after = reset_after
        self._last = None

    def __call__(self, wrist):
        last = self._last
        if last is None or wrist.t - last.t > self.reset_after:
            smoothed = wrist
        else:
            a = self.amount
            smoothed = Wrist(wrist.t, last.x + a * (wrist.x - last.x), last.y + a * (wrist.y - last.y))
        self._last = smoothed
        return smoothed


class WristHistory:
    """Recent wrist positions, so a swing can be matched with where the hand was at that moment."""

    def __init__(self, max_age=config.WRIST_MAX_AGE_S):
        self.max_age = max_age
        self._samples = deque(maxlen=120)
        self._lock = threading.Lock()   # the camera thread adds while the game loop reads

    def add(self, wrist):
        with self._lock:
            self._samples.append(wrist)

    def latest(self):
        with self._lock:
            return self._samples[-1] if self._samples else None

    def at(self, t):
        """The last position seen at or before time t, if it's recent enough; else None."""
        with self._lock:
            samples = list(self._samples)
        for w in reversed(samples):
            if w.t <= t:
                return w if t - w.t <= self.max_age else None
        return None


class PoseTracker:
    """Runs mediapipe on camera frames and records the wrist."""

    def __init__(self, model=config.POSE_MODEL, hand=config.HITTING_HAND, smoothing=config.WRIST_SMOOTHING):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = MODELS_DIR / f"pose_landmarker_{model}.task"
        if not path.exists():
            print(f"Downloading the {model} pose model (first run only)...")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL.format(model), path)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.model = model
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._index = WRIST_INDEX[hand]
        self._last_ms = -1
        self._smooth = Smoother(smoothing)
        self.history = WristHistory()

    def process(self, frame_bgr, t):
        """Detect the wrist in an UN-mirrored BGR frame taken at time t. Returns a Wrist or None."""
        import cv2

        ms = max(int(t * 1000), self._last_ms + 1)   # mediapipe needs increasing timestamps
        self._last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0][self._index]
        if lm.visibility is not None and lm.visibility < config.WRIST_MIN_VISIBILITY:
            return None
        wrist = self._smooth(Wrist(t, 1.0 - lm.x, lm.y))   # flip x to match the mirrored display
        self.history.add(wrist)
        return wrist

    def close(self):
        self._landmarker.close()


class MousePose:
    """Stand-in for PoseTracker: the mouse is the wrist (main.py --fake-pose)."""

    def __init__(self, window, width, height):
        import cv2

        self.history = WristHistory()
        self._size = (width, height)
        self._mouse = None
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, _event, x, y, _flags, _param):
        self._mouse = (x / self._size[0], y / self._size[1])

    def process(self, _frame, t):
        if self._mouse is None:
            return None
        wrist = Wrist(t, *self._mouse)
        self.history.add(wrist)
        return wrist

    def close(self):
        pass


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=POSE_MODELS, default=config.POSE_MODEL)
    parser.add_argument("--raw", action="store_true", help="no smoothing")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    tracker = PoseTracker(model=args.model, smoothing=1.0 if args.raw else config.WRIST_SMOOTHING)
    window = "Pose tracker (q to quit)"
    seen = deque(maxlen=60)      # was the wrist found in each of the last 60 frames?
    times = deque(maxlen=60)     # ms per frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            wrist = tracker.process(frame, t0)
            times.append((time.perf_counter() - t0) * 1000)
            seen.append(wrist is not None)
            shown = cv2.flip(frame, 1)
            h, w = shown.shape[:2]
            for edge in config.LANE_EDGES:
                cv2.line(shown, (int(edge * w), 0), (int(edge * w), h), (255, 255, 255), 1)
            text = "no wrist"
            if wrist:
                cv2.circle(shown, (int(wrist.x * w), int(wrist.y * h)), 12, (255, 255, 0), -1)
                text = f"{config.HITTING_HAND} wrist: {LANE_NAMES[lane_of(wrist.x)]}"
            stats = (f"{tracker.model}{' raw' if args.raw else ''}: {sum(times) / len(times):.0f} ms/frame, "
                     f"wrist seen {100 * sum(seen) / len(seen):.0f}%")
            cv2.putText(shown, stats, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(shown, text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
