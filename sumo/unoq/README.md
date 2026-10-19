# UNO Q code

`sumo-ultrasonic/` is a **snapshot** of the UNO Q app. The working copy lives in a
separate (private) App Lab repo, `ArduinoApps/`, where `tools/deploy.py` copies it to
the board. This copy is synced by hand, so it may lag behind.

- `sumo-ultrasonic/python/main.py`: MQTT publisher (`sensor/ultrasonic`) and buzzer player (`cmd/sound`)
- `sumo-ultrasonic/sketch/sketch.ino`: HC-SR04 reading and buzzer tones on the microcontroller (pins are `TODO(hardware)`)

Deploy from the `ArduinoApps` folder:

    python tools/deploy.py sumo-ultrasonic

To refresh this snapshot, copy `ArduinoApps/sumo-ultrasonic/` over `sumo/unoq/sumo-ultrasonic/`
(skip `__pycache__/`).

Before the sensor is wired, `python -m sensors.mock_ultrasonic` (run from `sumo/`)
publishes fake readings on the same topic.
