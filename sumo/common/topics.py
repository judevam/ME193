"""Every MQTT topic and its QoS/retain setting, in one place.

Code uses the short topic names below (e.g. SENSOR_COLOR); the bus adds the
"sumo/<robot_id>/" prefix.
"""

SENSOR_COLOR = "sensor/color"            # {"name": "blue", "rgb": [r, g, b], "t": epoch_s}
SENSOR_ULTRASONIC = "sensor/ultrasonic"  # {"cm": 34.5, "t": ...}

TELEOP_JOYSTICK = "teleop/joystick"      # {"left": -1..1, "right": -1..1, "t": ...}  (tank drive)
TELEOP_FLIPPER_MODE = "teleop/flipper_mode"  # "auto" | "manual", retained
TELEOP_FLIPPER_FIRE = "teleop/flipper_fire"  # {"t": ...}
TELEOP_DEFEAT = "teleop/defeat"          # {"t": ...}  manual defeat key

CMD_MOTORS = "cmd/motors"                # {"left": -1..1, "right": -1..1, "t": ..., "src_t": ... | null}
CMD_FLIPPER = "cmd/flipper"              # {"action": "fire" | "home", "t": ...}
CMD_SOUND = "cmd/sound"                  # {"song": "sad", "t": ...}

STATE = "state"                          # {"node", "flipper_node", "defeated", "flipper_mode", "t"}, retained
CONTROL_RESET = "control/reset"          # {"t": ...} clears the defeated latch

STATUS = "status"                        # status/<node>: "online" | "offline" (Last Will), retained
                                         # status/<node>/ble: "connected" | "disconnected", retained

QOS = {
    TELEOP_FLIPPER_MODE: 1,
    TELEOP_FLIPPER_FIRE: 1,
    TELEOP_DEFEAT: 1,
    CMD_FLIPPER: 1,
    CMD_SOUND: 1,
    CONTROL_RESET: 1,
}
RETAIN = {TELEOP_FLIPPER_MODE, STATE}


def status_topic(node: str) -> str:
    """Topic where a node reports online/offline."""
    return f"{STATUS}/{node}"


def ble_status_topic(node: str) -> str:
    """Topic where a bridge reports its LEGO device's BLE connection state."""
    return f"{STATUS}/{node}/ble"


def qos_for(topic: str) -> int:
    """QoS for a short topic name (status topics are QoS 1)."""
    if topic.startswith(STATUS + "/"):
        return 1
    return QOS.get(topic, 0)


def retain_for(topic: str) -> bool:
    """Whether a short topic name is published retained."""
    return topic in RETAIN or topic.startswith(STATUS + "/")


def prefix(robot_id: str) -> str:
    """Full topic prefix for a robot."""
    return f"sumo/{robot_id}/"
