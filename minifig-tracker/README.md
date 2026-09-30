# Green Minifig Tracker

Two-part project on an Arduino UNO Q ("JudeUnoQ" on the `Tufts_Wireless` network):

- **Part 1** — laptop webcam finds a green LEGO minifigure by color, publishes its
  position over MQTT, and the UNO Q lights a scaled dot on its LED matrix.
- **Part 2** — laptop runs a custom-trained YOLO model (data annotated in Roboflow,
  trained locally with `ultralytics`) to find the minifig, publishes the same MQTT
  message, and the UNO Q drives two DC motors (SparkFun Maker Drive) forward/back,
  stopping once the minifig is centered in the laptop's camera frame.

Both parts share one MQTT message shape published to topic `minifig/pos`:

```json
{"x": 0.0, "y": 0.0, "found": true}
```

`x`/`y` are the detected centroid normalized to `[0, 1]` across the frame
(`0,0` = top-left, `1,1` = bottom-right); `found` is `false` (with `x`/`y` omitted)
on frames where nothing was detected.

## MQTT broker

Mosquitto runs on your laptop (the board's apt/TLS is currently broken by a clock
issue, see below). The board apps connect out to your laptop's IP:1883 — set
`MQTT_HOST` in each board app's `python/main.py` to your laptop's IPv4 address
(`ipconfig` on Windows).

## Laptop setup

```
cd laptop
pip install -r requirements.txt
python part1_color_track.py
```

A debug window shows the camera feed with the detected bounding box, centroid
marker, and the current mask — use it to tune the `GREEN_LOWER`/`GREEN_UPPER`
HSV constants at the top of the file for your lighting/minifig.

Part 2 (after training a model — see `training/`):

```
cd laptop
python part2_yolo_track.py
```

## Part 2 model training

See `training/README.md` for the step-by-step: capture images, annotate in
Roboflow, export, download, and train locally with `ultralytics`.

## Board apps

Two Arduino Apps live on the board (mirrored locally under `boards/` for version
control — the live copies are managed by App Lab, edit through it or keep both in
sync by hand):

- `Minifig LED Tracker` — Part 1 receiver, lights a pixel on the LED matrix.
  Note: the matrix is monochrome (3-bit grayscale), so the "blue dot" is
  represented as the brightest lit pixel — there's no physical blue LED.
- `Minifig Motor Tracker` — Part 2 receiver, drives the Maker Drive's two DC
  motors forward/back with a deadband around center, stopping when centered.

## Known blocker

The UNO Q's system clock is currently wrong (stuck around Dec 2025, RTC reset to
1970, NTP not syncing), which breaks TLS/HTTPS on the board — this blocks sketch
compilation (`apps_start`) and any apt/pip install on the board until it's fixed
with real terminal + sudo access to the board.
