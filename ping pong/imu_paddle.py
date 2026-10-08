"""Read gyro/accel from the Double Motor paddle.

    python imu_paddle.py live                      # live readout + measured sample rate
    python imu_paddle.py record swing --seconds 20 # save data/swing_<time>.csv
    python imu_paddle.py plot data/swing_*.csv     # graph recordings + print stats (no hardware)

Record one file per label: still, walk, wiggle, swing (5-10 swings in the swing file).
"""

import argparse
import csv
import glob
import threading
import time
from collections import deque
from dataclasses import dataclass, astuple, fields
from datetime import datetime
from pathlib import Path

import config

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class ImuSample:
    t: float    # seconds (perf_counter on arrival at the laptop)
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float
    gesture: int  # -1 none, 0 tap, 1 double tap, 2 collision, 3 shake, 4 freefall


CSV_HEADER = [f.name for f in fields(ImuSample)]


class DoubleMotorPaddle:
    """The real paddle. Every BLE notification becomes one ImuSample in a queue."""

    def __init__(self, card_color=config.PADDLE_CARD_COLOR, card_serial=config.PADDLE_CARD_SERIAL,
                 notify_ms=config.IMU_NOTIFY_MS):
        self.card_color = card_color
        self.card_serial = card_serial
        self.notify_ms = notify_ms
        self._queue = deque(maxlen=10_000)
        self._lock = threading.Lock()
        self.motor = None

    def connect(self):
        import legoeducation as le   # imported here so tests don't need BLE
        from lelib import doubleMotor

        color = getattr(le, f"LEGO_COLOR_{self.card_color.upper()}")
        self.motor = doubleMotor()
        print(f"Connecting to Double Motor ({self.card_color} {self.card_serial})...")
        self.motor.connect(card_serial=self.card_serial, card_color=color)
        self.motor.device_notification_request(self.notify_ms)
        self.motor.set_notification_callback(self._on_notification)
        print("Connected.")

    def _on_notification(self, _raw):
        # Runs on the BLE thread; lelib has already copied the new values into imu_device.
        imu = self.motor.imu_device
        gesture = self.motor.imu_gesture.gesture
        sample = ImuSample(time.perf_counter(),
                           float(imu.gyroscopeX), float(imu.gyroscopeY), float(imu.gyroscopeZ),
                           float(imu.accelerometerX), float(imu.accelerometerY), float(imu.accelerometerZ),
                           int(gesture) if gesture == gesture else -1)  # NaN before the first gesture
        with self._lock:
            self._queue.append(sample)

    def read(self):
        """Return every sample received since the last call (oldest first)."""
        with self._lock:
            samples = list(self._queue)
            self._queue.clear()
        return samples

    def disconnect(self):
        if self.motor is not None:
            self.motor.set_notification_callback(lambda _raw: None)  # None isn't accepted
            self.motor.disconnect()


# ── CSV helpers (hardware-free, reused by tests) ─────────────────────────────

def save_csv(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        t0 = samples[0].t if samples else 0.0
        for s in samples:
            row = list(astuple(s))
            row[0] = round(s.t - t0, 4)   # store time relative to the first sample
            w.writerow(row)


def load_csv(path):
    with open(path, newline="") as f:
        return [ImuSample(float(r["t"]), float(r["gx"]), float(r["gy"]), float(r["gz"]),
                          float(r["ax"]), float(r["ay"]), float(r["az"]), int(r["gesture"]))
                for r in csv.DictReader(f)]


def stats(samples):
    """Sample rate, timing gaps and per-axis gyro peaks for one recording."""
    dts = [b.t - a.t for a, b in zip(samples, samples[1:])]
    duration = samples[-1].t - samples[0].t if len(samples) > 1 else 0.0
    return {
        "n": len(samples),
        "seconds": duration,
        "hz": (len(samples) - 1) / duration if duration > 0 else 0.0,
        "dt_mean_ms": 1000 * sum(dts) / len(dts) if dts else 0.0,
        "dt_max_ms": 1000 * max(dts) if dts else 0.0,
        "peak_gx": max((abs(s.gx) for s in samples), default=0.0),
        "peak_gy": max((abs(s.gy) for s in samples), default=0.0),
        "peak_gz": max((abs(s.gz) for s in samples), default=0.0),
        "peak_gmag": max(((s.gx**2 + s.gy**2 + s.gz**2) ** 0.5 for s in samples), default=0.0),
    }


# ── Command-line demos ───────────────────────────────────────────────────────

def run_live(paddle):
    print("Live IMU (Ctrl+C to stop). Rate is measured over the last second.")
    recent = deque()
    try:
        while True:
            time.sleep(0.1)
            new = paddle.read()
            now = time.perf_counter()
            recent.extend(s.t for s in new)
            while recent and recent[0] < now - 1.0:
                recent.popleft()
            if not new:
                print(f"{len(recent):3d} Hz | (no samples)")
                continue
            s = new[-1]
            gmag = max((x.gx**2 + x.gy**2 + x.gz**2) ** 0.5 for x in new)
            bar = "#" * min(40, int(gmag / 500))
            print(f"{len(recent):3d} Hz | gyro {s.gx:7.0f} {s.gy:7.0f} {s.gz:7.0f} | "
                  f"acc {s.ax:7.0f} {s.ay:7.0f} {s.az:7.0f} | g{s.gesture:2d} | {bar}")
    except KeyboardInterrupt:
        print()


def run_record(paddle, label, seconds):
    for n in (3, 2, 1):
        print(f"Recording '{label}' in {n}...")
        time.sleep(1)
    paddle.read()   # drop anything from before the countdown
    print(f"GO - recording for {seconds} s")
    samples = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        time.sleep(0.05)
        samples.extend(paddle.read())
    print("STOP")
    if not samples:
        print("No samples received - is the motor connected?")
        return
    path = DATA_DIR / f"{label}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    save_csv(path, samples)
    st = stats(load_csv(path))
    print(f"Saved {path.name}: {st['n']} samples, {st['hz']:.1f} Hz, "
          f"max gap {st['dt_max_ms']:.0f} ms, peak |gyro| {st['peak_gmag']:.0f}")


def run_plot(patterns, show=True):
    import matplotlib.pyplot as plt

    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("No files matched.")
        return
    print(f"{'file':38s} {'n':>5s} {'Hz':>5s} {'dt_ms':>6s} {'gap_ms':>6s} "
          f"{'|gx|':>7s} {'|gy|':>7s} {'|gz|':>7s} {'|g|':>7s}")
    fig, axes = plt.subplots(len(paths), 1, sharex=True, squeeze=False,
                             figsize=(11, 2.4 * len(paths)))
    for ax, path in zip(axes[:, 0], paths):
        samples = load_csv(path)
        st = stats(samples)
        print(f"{Path(path).name:38s} {st['n']:5d} {st['hz']:5.1f} {st['dt_mean_ms']:6.1f} "
              f"{st['dt_max_ms']:6.0f} {st['peak_gx']:7.0f} {st['peak_gy']:7.0f} "
              f"{st['peak_gz']:7.0f} {st['peak_gmag']:7.0f}")
        t = [s.t for s in samples]
        ax.plot(t, [s.gx for s in samples], label="gx")
        ax.plot(t, [s.gy for s in samples], label="gy")
        ax.plot(t, [s.gz for s in samples], label="gz")
        ax.set_title(Path(path).name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("live")
    rec = sub.add_parser("record")
    rec.add_argument("label", help="still / walk / wiggle / swing")
    rec.add_argument("--seconds", type=float, default=20)
    plot = sub.add_parser("plot")
    plot.add_argument("files", nargs="+")
    args = parser.parse_args()

    if args.mode == "plot":
        run_plot(args.files)
        return

    paddle = DoubleMotorPaddle()
    paddle.connect()
    try:
        if args.mode == "live":
            run_live(paddle)
        else:
            run_record(paddle, args.label, args.seconds)
    finally:
        paddle.disconnect()


if __name__ == "__main__":
    main()
