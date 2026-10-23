# UNO Q code

`gait-wearable/` is a **snapshot** of the UNO Q app. The working copy lives in a separate
(private) App Lab repo, `ArduinoApps/`, where `tools/deploy.py` copies it to the board.
This copy is synced by hand, so it may lag behind.

- `gait-wearable/sketch/MPU6050.h`, `.cpp`: MPU-6050 driver (I2C, FIFO, gyro calibration)
- `gait-wearable/sketch/sketch.ino`: IMU at 100 Hz, `read_samples()` and `draw()` for Python over the Bridge
- `gait-wearable/python/main.py`: MQTT-controlled recording to CSV on the board, live sample stream, status
- `gait-wearable/python/matrix_art.py`: LED matrix animations for each activity

Deploy from the `ArduinoApps` folder:

    python tools/deploy.py gait-wearable

To refresh this snapshot, copy `ArduinoApps/gait-wearable/` over `Gait Wearable/unoq/gait-wearable/`
(skip `__pycache__/`, `.cache/` and `python/data/`).
