"""One AprilTag starts the game; WHERE you hold it picks the level.

    python level_tags.py           # webcam: shows the tag, the three zones and the level it picks
    python level_tags.py --make    # save the printable tag to tags/start_tag.png

The (mirrored) screen is split into the same three columns as the ball lanes:
left = easy, middle = medium, right = hard. Hold the tag (config.START_TAG_ID, family 36h11 like
aprilTags/) still in one column for config.TAG_HOLD_S to start; moving it to another column
restarts the hold, so a tag passing through doesn't start the wrong level.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import config
from pose_tracker import lane_of

LEVEL_NAMES = {1: "easy", 2: "medium", 3: "hard"}
TAGS_DIR = Path(__file__).resolve().parent / "tags"


@dataclass
class Tag:
    id: int
    corners: list   # 4 (x, y) points in the frame it was found in


def level_at(x):
    """Level for a tag centred at mirrored-screen x (0..1): left third easy .. right third hard."""
    return lane_of(x) + 1


def mirrored_center_x(tag, frame_width):
    """The tag's centre as a 0..1 x position on the MIRRORED screen the player sees."""
    cx = sum(x for x, _ in tag.corners) / 4
    return 1.0 - cx / frame_width


class LevelPicker:
    """Pure logic: which column is the start tag held in, and has it been held long enough?"""

    def __init__(self, hold=config.TAG_HOLD_S):
        self.hold = hold
        self.level = None        # level of the column the tag is in right now
        self._since = None       # when the tag arrived in that column

    def update(self, tag_xs, t):
        """Feed the mirrored-screen x of each start tag seen in one frame (usually 0 or 1 of them).
        Returns a level once the tag has been held in one column long enough."""
        level = level_at(tag_xs[0]) if tag_xs else None
        if level != self.level:
            self.level, self._since = level, t
        if level is not None and t - self._since >= self.hold:
            self._since = float("inf")         # report once until the tag is put away or moved
            return level
        return None

    def progress(self, t):
        """0..1: how far through the hold the tag is (for drawing a ring)."""
        if self.level is None or self._since == float("inf"):
            return 0.0
        return min(1.0, (t - self._since) / self.hold)

    def reset(self):
        self.level, self._since = None, None


class TagReader:
    """Finds AprilTags in a camera frame with OpenCV's aruco module."""

    def __init__(self):
        import cv2

        self._cv2 = cv2
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(tag_dict, cv2.aruco.DetectorParameters())

    def detect(self, frame_bgr):
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        return [Tag(int(i), c.reshape(4, 2).tolist()) for c, i in zip(corners_list, ids.flatten())]


def make_tag(out_dir=TAGS_DIR):
    """Printable start tag with a white border (detection needs it) and a label underneath."""
    import cv2
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(tag_dict, config.START_TAG_ID, 600)
    page = np.full((900, 800), 255, np.uint8)
    page[100:700, 100:700] = tag
    label = f"Ping pong start tag (id {config.START_TAG_ID})"
    cv2.putText(page, label, (100, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3, cv2.LINE_AA)
    path = out_dir / "start_tag.png"
    cv2.imwrite(str(path), page)
    print(f"Saved {path}")
    return path


def main():
    import time
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make", action="store_true", help="save the printable tag to tags/")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.make:
        make_tag()
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    reader, picker = TagReader(), LevelPicker()
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            h, w = raw.shape[:2]
            tags = [tag for tag in reader.detect(raw) if tag.id == config.START_TAG_ID]
            xs = [mirrored_center_x(tag, w) for tag in tags]
            picked = picker.update(xs, time.perf_counter())
            if picked:
                print(f"START level {picked} ({LEVEL_NAMES[picked]})")

            frame = cv2.flip(raw, 1)   # mirrored, like the game
            for edge in config.LANE_EDGES:
                cv2.line(frame, (int(edge * w), 0), (int(edge * w), h), (255, 255, 255), 1)
            for tag, x in zip(tags, xs):
                pts = [(int(w - px), int(py)) for px, py in tag.corners]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    cv2.line(frame, a, b, (0, 255, 0), 2)
                cv2.putText(frame, f"level {level_at(x)} ({LEVEL_NAMES[level_at(x)]})", (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Level tag (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
