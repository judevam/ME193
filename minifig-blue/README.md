# Minifig tracker

A YOLOv8 model on the laptop finds a minifig (green or blue) in the webcam
feed and publishes its position over MQTT. An Arduino UNO Q receives it.

```
webcam -> YOLOv8 (laptop/detect_publish.py) -> MQTT minifig/pos -> UNO Q python/main.py -> Bridge -> sketch.ino
```

Payload (x, y normalized 0..1, origin top-left):
`{"color": "blue", "found": true, "x": 0.42, "y": 0.61, "conf": 0.88}` or `{"color": "blue", "found": false}`

## Folders
- `laptop/` - training (`train.py --color blue`) and detection (`detect_publish.py --color blue`). Datasets in `laptop/datasets/<color>/`.
- `unoq-led-app/` - UNO Q app that lights the LED matrix dot at the minifig's position.
- `unoq-drive-app/` - UNO Q app that drives a car (Cytron Maker Drive) so the minifig ends up centered in the camera view.

## Running the drive app
1. **Wire it** (UNO Q pin -> Maker Drive input; pins are set at the top of `sketch/sketch.ino`):

   | Motor | Maker Drive | UNO Q |
   |---|---|---|
   | left  | M1A / M1B | D3 / D5 |
   | right | M2A / M2B | D6 / D9 |

   Also connect **UNO Q GND to Maker Drive GND**. Only PWM pins work: 3, 5, 6, 9, 10, 11.
   The Maker Drive runs on **2.5-9.5 V and 1 A per motor** (VB+ / VB- terminals).
   Do not connect more than 9.5 V to it. The UNO Q needs 5 V / 3 A on USB-C, or
   7-24 V on VIN, so one battery can only feed both if it stays between 7 and 9.5 V.
2. **Laptop:** `mosquitto -c laptop/mosquitto.conf -v` (allow TCP 1883 in the firewall).
3. **UNO Q:** set `MQTT_HOST` in `unoq-drive-app/python/main.py` to the laptop's IP, then run the app from Arduino App Lab.
4. **Check the motors first, wheels off the ground:** set `SELF_TEST = True`, run, and confirm left then right spin forward. Fix a wrong one with `INVERT_LEFT` / `INVERT_RIGHT` in the sketch. Set `SELF_TEST` back to `False`.
5. **Laptop:** `python laptop/detect_publish.py --color blue --broker <laptop-ip> --rate 15`

## Tuning (top of `python/main.py`)
- `MODE`: `"drive"` = camera fixed, car drives forward/back; `"turn"` = camera on the car, car spins in place.
- Car runs away from center -> set `INVERT = True`.
- Stalls short of center -> raise `MIN_SPEED`. Overshoots or hunts -> lower `MIN_SPEED`/`MAX_SPEED`, widen `START_ZONE`.
- The sketch stops the motors if no command arrives for 0.5 s.
