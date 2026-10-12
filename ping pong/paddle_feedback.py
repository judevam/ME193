"""Haptic buzz + light flash on the Double Motor paddle (Phase 7).

    python paddle_feedback.py              # Bluetooth check: buzzes while recording the IMU
    python paddle_feedback.py --seconds 30

The check alternates hit / miss feedback every 1.5 s while recording IMU samples. HOLD THE PADDLE
STILL. At the end it reports the sample rate (should stay ~63 Hz), the biggest gap, how long the
commands took to send, and whether the buzzing set off the swing detector (should be 0).

In the game, `send()` only puts the event on a queue; a background thread talks to the motor, so
the game loop never waits on Bluetooth. Motor pulse and light change go out in one BLE packet.
"""

import argparse
import queue
import threading
import time

import config


class PaddleFeedback:
    def __init__(self, motor, haptic=config.HAPTIC, side=config.HAPTIC_MOTOR,
                 flash_s=config.LIGHT_FLASH_S, idle_color=config.LIGHT_IDLE_COLOR):
        import legoeducation as le

        self._le = le
        self._motor = motor          # a connected lelib.doubleMotor
        self._haptic = haptic
        cw, ccw = le.MOTOR_MOVE_DIRECTION_CLOCKWISE, le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
        # (motor, direction) for each output that spins; separate motors can share one batch
        self._spins = {"left": [(le.MOTOR_LEFT, cw)], "right": [(le.MOTOR_RIGHT, cw)],
                       "both": [(le.MOTOR_BOTH, cw)],
                       "opposite": [(le.MOTOR_LEFT, cw), (le.MOTOR_RIGHT, ccw)]}[side]
        self._flash_s = flash_s
        self._idle = self._color(idle_color)
        self._queue = queue.Queue()
        self.send_ms = []            # how long each send took (for the check)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, event):
        """Queue feedback for a game event ("hit", "miss", "whiff"). Returns immediately."""
        if event in self._haptic:
            self._queue.put(event)

    def stop(self):
        self._queue.put(None)
        self._thread.join(timeout=1.0)

    def _color(self, name):
        return getattr(self._le, f"LEGO_COLOR_{name.upper()}")

    def _run(self):
        lit = False   # is the light showing an event color that needs to go back to idle?
        while True:
            try:
                event = self._queue.get(timeout=self._flash_s if lit else None)
            except queue.Empty:
                self._safe(lambda: self._motor.light_color(self._idle, blocking=False))
                lit = False
                continue
            if event is None:
                return
            buzz_ms, speed, color = self._haptic[event]
            t0 = time.perf_counter()

            def fire():
                with self._motor.batch(blocking=False):   # one BLE packet: motor(s) + light together
                    if buzz_ms > 0:
                        for motor, direction in self._spins:
                            self._motor.motor_run_for_time(buzz_ms, direction=direction, motor=motor,
                                                           speed=speed, blocking=False)
                    self._motor.light_color(self._color(color), blocking=False)

            self._safe(fire)
            self.send_ms.append((time.perf_counter() - t0) * 1000)
            lit = True

    def _safe(self, action):
        try:
            action()
        except Exception as e:   # never let feedback take the game down
            print(f"Paddle feedback failed: {e}")


class NoFeedback:
    """Stand-in when there's no paddle (--fake-imu)."""

    send_ms = []

    def send(self, event):
        pass

    def stop(self):
        pass


def main():
    from imu_paddle import DoubleMotorPaddle, stats
    from swing import SwingDetector

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()

    paddle = DoubleMotorPaddle()
    paddle.connect()
    feedback = PaddleFeedback(paddle.motor)
    detector = SwingDetector()
    print(f"Hold the paddle STILL. Buzzing ({config.HAPTIC_MOTOR} side) every 1.5 s for {args.seconds:.0f} s...")
    samples, false_swings, n = [], 0, 0
    try:
        paddle.read()
        end = time.perf_counter() + args.seconds
        next_buzz = time.perf_counter() + 1.0
        while time.perf_counter() < end:
            now = time.perf_counter()
            if now >= next_buzz:
                event = "hit" if n % 2 == 0 else "miss"
                feedback.send(event)
                print(f"  {event}")
                n, next_buzz = n + 1, now + 1.5
            new = paddle.read()
            samples += new
            false_swings += len(detector.update_many(new))
            time.sleep(0.01)
    finally:
        feedback.stop()
        paddle.disconnect()

    st = stats(samples)
    sends = feedback.send_ms
    print(f"\nIMU: {st['hz']:.1f} Hz (no buzz: ~63), biggest gap {st['dt_max_ms']:.0f} ms, "
          f"peak gyro {st['peak_gmag']:.0f}")
    if sends:
        print(f"Commands: {len(sends)} sent, {sum(sends) / len(sends):.0f} ms average, {max(sends):.0f} ms max")
    print(f"False swings from the buzzing: {false_swings}  ({'OK' if false_swings == 0 else 'PROBLEM'})")


if __name__ == "__main__":
    main()
