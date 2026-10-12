"""CameraWorker with a fake camera and fake pose: frames arrive, pose/tags only when asked."""

import time

import numpy as np

from camera_worker import CameraWorker


class FakeCap:
    def read(self):
        time.sleep(0.005)
        return True, np.zeros((48, 64, 3), np.uint8)


class FakePose:
    def __init__(self):
        self.calls = 0

    def process(self, frame, t):
        assert frame.shape == (48, 64, 3)   # the un-mirrored, un-resized camera frame
        self.calls += 1


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_frames_arrive_resized():
    worker = CameraWorker(FakeCap(), (96, 72))
    try:
        assert wait_for(lambda: worker.latest()[0] is not None)
        frame, tags, count = worker.latest()
        assert frame.shape == (72, 96, 3)
        assert tags == [] and count > 0
    finally:
        worker.stop()


def test_pose_only_runs_when_tracking():
    pose = FakePose()
    worker = CameraWorker(FakeCap(), (96, 72), pose=pose)
    try:
        assert wait_for(lambda: worker.latest()[2] > 5)
        assert pose.calls == 0
        worker.track_pose = True
        assert wait_for(lambda: pose.calls > 3)
    finally:
        worker.stop()
