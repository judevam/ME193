"""Hardware-free checks for the imu_paddle CSV helpers and plot."""

import matplotlib
matplotlib.use("Agg")

from imu_paddle import ImuSample, save_csv, load_csv, stats, run_plot


def fake_samples(n=100, dt=0.015, spike_at=50):
    out = []
    for i in range(n):
        gz = 20000.0 if i == spike_at else 10.0
        out.append(ImuSample(100.0 + i * dt, 1.0, -2.0, gz, 0.0, 0.0, 1000.0, -1))
    return out


def test_csv_round_trip_makes_time_relative(tmp_path):
    path = tmp_path / "swing_test.csv"
    save_csv(path, fake_samples())
    back = load_csv(path)
    assert len(back) == 100
    assert back[0].t == 0.0
    assert abs(back[-1].t - 99 * 0.015) < 1e-3
    assert back[50].gz == 20000.0
    assert back[0].gesture == -1


def test_stats_rate_and_peaks():
    st = stats(fake_samples(dt=0.015))
    assert abs(st["hz"] - 1 / 0.015) < 0.5
    assert abs(st["dt_max_ms"] - 15) < 0.5
    assert st["peak_gz"] == 20000.0
    assert st["peak_gx"] == 1.0


def test_plot_runs_without_display(tmp_path):
    save_csv(tmp_path / "a.csv", fake_samples())
    save_csv(tmp_path / "b.csv", fake_samples(spike_at=10))
    fig = run_plot([str(tmp_path / "*.csv")], show=False)
    assert len(fig.axes) == 2
