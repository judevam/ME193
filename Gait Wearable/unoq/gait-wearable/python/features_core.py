"""The feature math for one window of IMU samples. numpy only, no project settings, so the
exact same file runs on the laptop and on the UNO Q (push_models.py copies it to the board).
"""
import numpy as np

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]


def peak(w, axis_index=0, sign=1):
    """Model 1's only input: the peak acceleration along one axis, in g."""
    return float(np.max(sign * w[:, axis_index]))


def dominant_freq(x, rate_hz, lo=0.3, hi=5.0):
    """(frequency in Hz, share of the signal's power at that frequency) within lo..hi.
    This is the step or pedal cadence."""
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate_hz)
    band = (freqs >= lo) & (freqs <= hi)
    total = power[1:].sum()
    if not band.any() or total <= 0:
        return 0.0, 0.0
    i = np.argmax(np.where(band, power, 0))
    return float(freqs[i]), float(power[i] / total)


def _corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def all_features(w, rate_hz):
    """Model 2's inputs: ~50 numbers that summarize one window of all six IMU outputs."""
    ax, ay, az, gx, gy, gz = w.T
    signals = {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz,
               "amag": np.sqrt(ax**2 + ay**2 + az**2),
               "gmag": np.sqrt(gx**2 + gy**2 + gz**2)}
    f = {}
    for name, x in signals.items():
        f[f"{name}_mean"] = float(x.mean())
        f[f"{name}_std"] = float(x.std())
        f[f"{name}_min"] = float(x.min())
        f[f"{name}_max"] = float(x.max())
        f[f"{name}_range"] = float(np.ptp(x))
    for name in ("ax", "amag", "gmag"):
        f[f"{name}_freq"], f[f"{name}_freq_power"] = dominant_freq(signals[name], rate_hz)
    # Orientation from gravity (the window's mean acceleration). With X along the thigh:
    # standing -> pitch near +/-90 deg, biking or lying -> much flatter.
    mx, my, mz = ax.mean(), ay.mean(), az.mean()
    f["pitch_deg"] = float(np.degrees(np.arctan2(mx, np.hypot(my, mz))))
    f["roll_deg"] = float(np.degrees(np.arctan2(my, mz)))
    f["corr_xy"], f["corr_xz"], f["corr_yz"] = _corr(ax, ay), _corr(ax, az), _corr(ay, az)
    return f
