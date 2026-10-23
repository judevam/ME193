"""Every tunable number for the laptop side in one place."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"          # recordings copied off the board (pull_data.py)
MODEL_DIR = HERE / "models"       # trained models and plots (train.py)

# MQTT: must match python/main.py on the board
BROKER_HOST = "test.mosquitto.org"
BROKER_PORT = 1883
TOPIC = "ME193/Jude/gait"

# Where recordings live on the board (the app's python folder)
BOARD_DATA_DIR = "/home/arduino/ArduinoApps/gait-wearable/python/data"
ARDUINO_APPS = HERE.parent / "ArduinoApps"   # local copy of the private App Lab repo

LABELS = ["standing", "walking", "running", "biking"]

# Windows
RATE_HZ = 100
WINDOW_S = 2.0
HOP_S = 1.0        # 50 % overlap: one new prediction per second
TRIM_S = 3.0       # drop this much from the start and end of every recording

# Model 1: peak acceleration along one axis. X points along the thigh bone toward
# the hip. If the IMU ended up mounted the other way round, set PEAK_SIGN = -1.
PEAK_AXIS = "ax"
PEAK_SIGN = 1
