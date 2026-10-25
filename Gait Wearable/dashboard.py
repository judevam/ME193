"""Live dashboard for the gait wearable.

  python dashboard.py                          # live from the board (over MQTT)
  python dashboard.py --fake                   # no board: synthetic data cycling through activities
  python dashboard.py --models data_synthetic/models --fake   # with the synthetic-data models

Shows the current gait (both models), live accel and gyro graphs, model 1's peak per window
against its thresholds, a gait timeline, and the board's status. The buttons start and stop
recordings on the board, like `gait.py record`.

Classification runs here with the same code and models (models/gait_models.json) as the
board, so the two should agree. "It says" in the status panel is the board's own answer.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, RadioButtons

import config
from live import LiveGait, parse_imu

COLORS = {"standing": "#5b8def", "walking": "#3fb950", "running": "#f0883e",
          "biking": "#bc8cff", None: "#6e7681"}
AXIS_COLORS = ["#ff7b72", "#3fb950", "#58a6ff"]   # X, Y, Z
GRAPH_S = 10        # seconds shown in the accel and gyro graphs
TIMELINE_S = 120    # seconds shown in the peak and timeline graphs
# Fixed ranges (the pilot's max was 4.9 g and 503 deg/s). Fixed axes let matplotlib redraw
# only the lines and text each frame ("blitting") instead of the whole window, which was
# too slow (~340 ms a frame) and made the dashboard lag.
ACCEL_RANGE = (-2.0, 5.0)
GYRO_RANGE = (-600.0, 600.0)
PEAK_RANGE = (-1.5, 4.5)


class Dashboard:
    def __init__(self, live: LiveGait, client=None):
        self.live = live
        self.client = client
        self.lock = threading.Lock()
        self.online = None          # from TOPIC/status
        self.board: dict = {}       # last message on TOPIC
        self.last_imu = 0.0         # time.time() of the last sample batch
        self._build()

    # ---- incoming data (MQTT thread) ----
    def on_message(self, client, userdata, msg) -> None:
        with self.lock:
            if msg.topic.endswith("/imu"):
                seq, samples = parse_imu(msg.payload)
                before = len(self.live.history) and self.live.history[-1][0]
                self.live.add(samples, seq)
                self.last_imu = time.time()
                if (self.live.history and self.live.history[-1][0] != before
                        and "gait" not in self.board):
                    # The board has no models of its own: show our answer on its LED matrix.
                    # Plain publish (no waiting): we're inside paho's callback thread.
                    client.publish(config.TOPIC + "/cmd", json.dumps(
                        {"cmd": "show", "gait": self.live.history[-1][1], "seconds": 5}))
            elif msg.topic.endswith("/status"):
                self.online = msg.payload.decode() == "online"
            else:
                try:
                    self.board = json.loads(msg.payload)
                except ValueError:
                    pass

    def add_fake(self, samples) -> None:
        with self.lock:
            self.live.add(samples)
            self.last_imu = time.time()
            self.online = True

    # ---- layout ----
    def _build(self) -> None:
        plt.style.use("dark_background")
        self.fig = fig = plt.figure(figsize=(15, 9))
        fig.canvas.manager.set_window_title("Gait Wearable")
        gs = fig.add_gridspec(4, 3, height_ratios=[1.15, 1, 1, 1], width_ratios=[1, 1, 0.75],
                              hspace=0.55, wspace=0.25, left=0.05, right=0.98, top=0.95, bottom=0.06)

        # Big gait readout
        self.ax_gait = fig.add_subplot(gs[0, 0])
        self.ax_gait.axis("off")
        self.txt_gait = self.ax_gait.text(0.0, 0.62, "-", fontsize=44, weight="bold", va="center")
        self.txt_conf = self.ax_gait.text(0.0, 0.22, "", fontsize=13, va="center", color="#c9d1d9")
        self.txt_models = self.ax_gait.text(0.0, -0.05, "", fontsize=11, va="center", color="#8b949e")

        # Gait timeline: one image, one pixel per window (redrawing a bar per window was slow)
        self.ax_tl = fig.add_subplot(gs[0, 1:])
        self.tl_labels = [None] + config.LABELS
        self.tl_slots = int(TIMELINE_S / config.HOP_S)
        self.tl_image = self.ax_tl.imshow(
            np.zeros((1, self.tl_slots)), aspect="auto", interpolation="nearest",
            cmap=ListedColormap(["#000000"] + [COLORS[l] for l in config.LABELS]),
            vmin=0, vmax=len(config.LABELS), extent=[-TIMELINE_S, 0, -0.5, 0.5])
        self.ax_tl.set_yticks([])
        self.ax_tl.set_title("Gait timeline (last 2 min)", loc="left", fontsize=10)
        for label in config.LABELS:
            self.ax_tl.barh(0, 0, color=COLORS[label], label=label)
        self.ax_tl.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4, fontsize=8, frameon=False)

        # Accel and gyro
        self.ax_acc = fig.add_subplot(gs[1, :2])
        self.ax_gyr = fig.add_subplot(gs[2, :2], sharex=self.ax_acc)
        self.acc_lines = [self.ax_acc.plot([], [], lw=1, color=c, label=l)[0]
                          for c, l in zip(AXIS_COLORS, ["ax", "ay", "az"])]
        self.acc_lines.append(self.ax_acc.plot([], [], lw=1.4, color="w", alpha=0.8, label="|a|")[0])
        self.gyr_lines = [self.ax_gyr.plot([], [], lw=1, color=c, label=l)[0]
                          for c, l in zip(AXIS_COLORS, ["gx", "gy", "gz"])]
        self.ax_acc.set_title("Acceleration (g)", loc="left", fontsize=10)
        self.ax_gyr.set_title("Rotation (deg/s)", loc="left", fontsize=10)
        self.ax_gyr.set_xlabel("seconds ago")
        self.ax_acc.set_xlim(-GRAPH_S, 0)
        self.ax_acc.set_ylim(*ACCEL_RANGE)
        self.ax_gyr.set_ylim(*GYRO_RANGE)
        for ax in (self.ax_acc, self.ax_gyr):
            ax.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.3)
            ax.grid(alpha=0.15)

        # Model 1: peak per window vs thresholds (bands drawn once, the line just updates)
        self.ax_peak = fig.add_subplot(gs[3, :2])
        self.peak_line = self.ax_peak.plot([], [], "o-", ms=3, lw=1, color="w")[0]
        self.ax_peak.set_title(f"Model 1 input: peak {config.PEAK_AXIS} per {config.WINDOW_S:g} s window (g), "
                               "bands = its thresholds", loc="left", fontsize=10)
        self.ax_peak.grid(alpha=0.15)
        self.ax_peak.set_xlim(-TIMELINE_S, 0)
        self.ax_peak.set_ylim(*PEAK_RANGE)
        self.ax_tl.set_xlim(-TIMELINE_S, 0)
        m1 = self.live.models
        if m1.order is not None:
            edges = [-5.0] + list(m1.thresholds) + [10.0]
            for (lo, up), label in zip(zip(edges, edges[1:]), m1.order):
                self.ax_peak.axhspan(lo, up, color=COLORS.get(label, COLORS[None]), alpha=0.12)
                # x in axes coordinates (stays at the left edge), y in g
                self.ax_peak.text(0.005, (max(lo, -0.5) + min(up, 4.0)) / 2, label, fontsize=8, va="center",
                                  color=COLORS.get(label), transform=self.ax_peak.get_yaxis_transform())

        # Status and controls
        self.ax_status = fig.add_subplot(gs[1:3, 2])
        self.ax_status.axis("off")
        self.txt_status = self.ax_status.text(0.02, 0.98, "", fontsize=11, va="top", family="monospace")

        cell = gs[3, 2].get_position(fig)
        ax_radio = fig.add_axes([cell.x0, cell.y0, cell.width * 0.45, cell.height])
        ax_radio.set_title("Record", fontsize=10, loc="left")
        self.radio = RadioButtons(ax_radio, config.LABELS, activecolor="#3fb950")
        ax_start = fig.add_axes([cell.x0 + cell.width * 0.52, cell.y0 + cell.height * 0.55,
                                 cell.width * 0.45, cell.height * 0.38])
        ax_stop = fig.add_axes([cell.x0 + cell.width * 0.52, cell.y0 + cell.height * 0.05,
                                cell.width * 0.45, cell.height * 0.38])
        self.btn_start = Button(ax_start, "Start", color="#238636", hovercolor="#2ea043")
        self.btn_stop = Button(ax_stop, "Stop", color="#da3633", hovercolor="#f85149")
        self.btn_start.on_clicked(lambda _: self._send({"cmd": "start", "label": self.radio.value_selected}))
        self.btn_stop.on_clicked(lambda _: self._send({"cmd": "stop"}))

        # Everything that changes each frame; only these get redrawn (see ACCEL_RANGE)
        self.animated = [*self.acc_lines, *self.gyr_lines, self.peak_line, self.tl_image,
                         self.txt_gait, self.txt_conf, self.txt_models, self.txt_status]
        for artist in self.animated:
            artist.set_animated(True)

    def _send(self, cmd: dict) -> None:
        if self.client is None:
            print(f"(fake mode) would send {cmd}")
            return
        from gait import send
        threading.Thread(target=send, args=(self.client, cmd), daemon=True).start()

    # ---- drawing (UI thread) ----
    def update(self, _frame=None) -> list:
        """Refresh the changing parts; returns them for blitting."""
        with self.lock:
            live = self.live
            t, data = live.recent(GRAPH_S)
            now, current = live.now, dict(live.current)
            peaks, history = list(live.peaks), list(live.history)
            online, board, last_imu, lost = self.online, dict(self.board), self.last_imu, live.lost

        # graphs, with time as "seconds ago"
        if len(data):
            ago = t - now
            amag = np.linalg.norm(data[:, :3], axis=1)
            for i, line in enumerate(self.acc_lines[:3]):
                line.set_data(ago, data[:, i])
            self.acc_lines[3].set_data(ago, amag)
            for i, line in enumerate(self.gyr_lines):
                line.set_data(ago, data[:, i + 3])

        # gait readout
        gait = current.get("gait")
        if gait:
            self.txt_gait.set_text(gait.upper())
            self.txt_gait.set_color(COLORS.get(gait, COLORS[None]))
            conf = current.get("confidence")
            self.txt_conf.set_text(f"confidence {conf:.0%}" if conf is not None else "")
            self.txt_models.set_text(f"model 1 (peak): {current.get('model1') or '-'}    "
                                     f"model 2 (all): {current.get('model2') or '-'}")
        elif not live.has_models:
            self.txt_gait.set_text("NO MODEL")
            self.txt_gait.set_color(COLORS[None])
            self.txt_conf.set_text("run train.py first (or --models data_synthetic/models)")
        else:
            self.txt_gait.set_text("...")
            self.txt_conf.set_text("collecting the first window")

        self._draw_timeline(history, now)
        self._draw_peaks(peaks, now)
        self._draw_status(online, board, last_imu, lost, current)
        return self.animated

    def _draw_timeline(self, history, now) -> None:
        codes = np.zeros(self.tl_slots)
        start = now - TIMELINE_S
        for t, gait in history:
            slot = int((t - start) / config.HOP_S) - 1
            if 0 <= slot < self.tl_slots and gait in self.tl_labels:
                codes[slot] = self.tl_labels.index(gait)
        self.tl_image.set_data(codes[None, :])

    def _draw_peaks(self, peaks, now) -> None:
        if peaks:
            ts, ps = zip(*peaks)
            self.peak_line.set_data(np.asarray(ts) - now, np.clip(ps, *PEAK_RANGE))

    def _draw_status(self, online, board, last_imu, lost, current) -> None:
        age = time.time() - last_imu if last_imu else None
        if online is False or (age is not None and age > 3):
            state = "OFFLINE" if online is False else f"no data for {age:.0f} s"
        elif age is None:
            state = "waiting for data..."
        else:
            state = "online"
        lines = [f"Wearable   {state}"]
        if board:
            lines.append(f"Rate       {board.get('rate_hz', '?')} Hz")
            if board.get("gait"):
                lines.append(f"It says    {board['gait']}")  # the board's own classification
            elif "rate_hz" in board:
                lines.append("It says    (no models on the board)")
            if board.get("mode") == "record":
                lines += ["", f"RECORDING  {board.get('label')}",
                          f"           {board.get('seconds', 0):.0f} s, {board.get('samples', 0)} samples",
                          f"           {board.get('file', '').removesuffix('.csv')}"]
                if board.get("lost"):
                    lines.append(f"           LOST {board['lost']}")
            else:
                lines += ["", "Not recording"]
        lines += ["", f"Lost here  {lost} samples"]
        if current.get("peak") is not None:
            lines.append(f"Peak {config.PEAK_AXIS[1]}     {current['peak']:.2f} g")
        self.txt_status.set_text("\n".join(lines))
        self.txt_status.set_color("#f85149" if board.get("mode") == "record" else "#c9d1d9")


def fake_source(dash: Dashboard, stop: threading.Event, segment_s: float = 15) -> None:
    """Stream synthetic data like the board would, cycling through the activities."""
    from synthetic import generate
    rng = np.random.default_rng()
    batch = int(0.2 * config.RATE_HZ)
    while not stop.is_set():
        for label in config.LABELS:
            data = generate(label, segment_s, rng)[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()
            for i in range(0, len(data), batch):
                if stop.is_set():
                    return
                dash.add_fake(data[i:i + batch].tolist())
                time.sleep(0.2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="synthetic data instead of the board")
    p.add_argument("--models", type=Path, default=config.MODEL_DIR)
    args = p.parse_args()

    live = LiveGait(args.models)
    stop = threading.Event()
    if args.fake:
        dash = Dashboard(live)
        threading.Thread(target=fake_source, args=(dash, stop), daemon=True).start()
    else:
        from gait import connect
        dash = Dashboard(live)
        dash.client = connect(dash.on_message, imu=True)
    dash._anim = FuncAnimation(dash.fig, dash.update, interval=200, blit=True, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
        if dash.client:
            dash.client.disconnect()


if __name__ == "__main__":
    main()
