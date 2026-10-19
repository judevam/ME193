# UNO Q code

The UNO Q app lives in the App Lab folder so `tools/deploy.py` can copy it to the board:

- `ArduinoApps/sumo-ultrasonic/python/main.py`: MQTT publisher (`sensor/ultrasonic`) and buzzer player (`cmd/sound`)
- `ArduinoApps/sumo-ultrasonic/sketch/sketch.ino`: HC-SR04 reading and buzzer tones on the microcontroller (pins are `TODO(hardware)`)

Deploy from the `ArduinoApps` folder:

    python tools/deploy.py sumo-ultrasonic

Before the sensor is wired, `python -m sensors.mock_ultrasonic` (run from `sumo/`)
publishes fake readings on the same topic.
