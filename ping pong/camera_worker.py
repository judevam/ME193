"""Reads the webcam on a background thread so the game can draw at full speed.

The camera only delivers ~30 frames/s and pose tracking takes ~20 ms a frame. Doing both in the
game loop held drawing to ~20 fps, so the ball moved in visible jumps. Here they run on their own
thread; the game loop just picks up the latest frame and draws the ball for the current moment.

Each camera frame is:
  - run through pose tracking (if `track_pose` is on), on the UN-mirrored image
  - checked for the start tag (if `find_tags` is on), corners converted to display coordinates
  - mirrored and resized for display
"""

import threading
import time

import cv2

import config


class CameraWorker:
    def __init__(self, cap, size, pose=None, tag_reader=None):
        self._cap = cap
        self._size = size                 # (width, height) of the game window
        self._pose = pose
        self._tag_reader = tag_reader
        self.track_pose = False           # set by the game: on while playing
        self.find_tags = False            # set by the game: on at the start screen
        self._lock = threading.Lock()
        self._frame = None                # latest mirrored, resized frame
        self._tags = []                   # latest start-tag corners, display coordinates
        self._frame_count = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self):
        """(frame copy or None, start-tag corner lists, frames read so far)."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, list(self._tags), self._frame_count

    def _run(self):
        w, h = self._size
        while self._running:
            ok, raw = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.perf_counter()
            if self.track_pose and self._pose is not None:
                self._pose.process(raw, t)   # un-mirrored: keeps mediapipe's left/right right
            tags = []
            if self.find_tags and self._tag_reader is not None:
                rh, rw = raw.shape[:2]
                for tag in self._tag_reader.detect(raw):
                    if tag.id == config.START_TAG_ID:   # mirror the corners to match the display
                        tags.append([((rw - x) * w / rw, y * h / rh) for x, y in tag.corners])
            frame = cv2.resize(cv2.flip(raw, 1), (w, h))
            with self._lock:
                self._frame, self._tags = frame, tags
                self._frame_count += 1

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
