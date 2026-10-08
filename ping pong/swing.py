"""Turn IMU samples into "swing!" events. Pure logic, no hardware.

A swing needs two things within SWING_PAIR_WINDOW_S of each other:
  - rotation: gyro size sqrt(gx² + gy² + gz²) over SWING_THRESHOLD (either direction, so
    forehand and backhand both count)
  - an arc: accel size over SWING_ACCEL_THRESHOLD. Swinging your arm whips the paddle around
    (5-12 g in the recordings); twisting it in place, walking or wiggling stays under ~1.5 g.
Requiring both also ignores a knock or tap (accel but no rotation). After a swing, a cooldown
stops the follow-through bounce from counting as a second swing.

On a big backswing the gyro crosses first and the accel ~60 ms later, during the forward stroke,
so the swing is reported on the forward stroke. The direction isn't reported: the backswing
makes it unreliable.

    python swing.py      # live demo with the real paddle: prints SWING once per swing
"""

import math
from dataclasses import dataclass

import config


@dataclass
class Swing:
    t: float          # time of the sample that completed the swing
    speed: float      # gyro size at that sample


def gyro_size(sample):
    return math.sqrt(sample.gx ** 2 + sample.gy ** 2 + sample.gz ** 2)


def accel_size(sample):
    return math.sqrt(sample.ax ** 2 + sample.ay ** 2 + sample.az ** 2)


class SwingDetector:
    def __init__(self, threshold=config.SWING_THRESHOLD, accel_threshold=config.SWING_ACCEL_THRESHOLD,
                 pair_window=config.SWING_PAIR_WINDOW_S, cooldown=config.SWING_COOLDOWN_S):
        self.threshold = threshold
        self.accel_threshold = accel_threshold
        self.pair_window = pair_window
        self.cooldown = cooldown
        self._last_gyro_t = -math.inf    # last time each signal was over its threshold
        self._last_accel_t = -math.inf
        self._last_swing_t = -math.inf

    def update(self, sample):
        """Feed one ImuSample. Returns a Swing the first time a swing is seen, else None."""
        size = gyro_size(sample)
        if size >= self.threshold:
            self._last_gyro_t = sample.t
        if accel_size(sample) >= self.accel_threshold:
            self._last_accel_t = sample.t

        both = (sample.t - self._last_gyro_t <= self.pair_window
                and sample.t - self._last_accel_t <= self.pair_window)
        if not both or sample.t - self._last_swing_t < self.cooldown:
            return None
        self._last_swing_t = sample.t
        return Swing(sample.t, size)

    def update_many(self, samples):
        """Feed a batch (e.g. paddle.read()). Returns the list of swings found in it."""
        return [s for s in map(self.update, samples) if s]


def main():
    import time
    from imu_paddle import DoubleMotorPaddle

    paddle = DoubleMotorPaddle()
    paddle.connect()
    detector = SwingDetector()
    count = 0
    rotation = []   # samples since the gyro last went over threshold without making a swing
    print(f"Swing away (gyro > {detector.threshold}, accel > {detector.accel_threshold / 1000:g} g, "
          f"cooldown {detector.cooldown} s). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.01)
            for sample in paddle.read():
                swing = detector.update(sample)
                if swing:
                    count += 1
                    print(f"SWING #{count}  speed {swing.speed:5.0f}  "
                          f"peak accel {max(map(accel_size, rotation + [sample])) / 1000:4.1f} g")
                    rotation = []
                elif sample.t - detector._last_swing_t < detector.cooldown:
                    pass   # the rest of the swing just reported
                elif gyro_size(sample) >= detector.threshold:
                    rotation.append(sample)
                elif rotation and sample.t - rotation[-1].t > detector.cooldown:
                    # rotation ended without a swing: e.g. twisting in place
                    print(f"  (ignored: rotation {max(map(gyro_size, rotation)):5.0f}, "
                          f"peak accel only {max(map(accel_size, rotation)) / 1000:4.1f} g)")
                    rotation = []
    except KeyboardInterrupt:
        print()
    finally:
        paddle.disconnect()


if __name__ == "__main__":
    main()
