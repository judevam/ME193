"""Steer a LEGO Education Double Motor car with your arms, tracked via mediapipe Pose.

Raise both hands above your shoulders to drive forward - the higher they go
(up to MAX_RAISE), the faster the car goes. Bring your hands together (close
to each other) to drive backward. Raise one hand higher than the other to
steer toward that side: e.g. raising your right hand more turns the car
right, raising your left hand more turns it left.

A separate Single Motor connects independently and spins at a constant speed
(SINGLE_MOTOR_SPEED) for the entire time this script is running, regardless
of arm position.

See README.md for setup and a discussion of the sync/async model and the
mediapipe model's training and limitations.
"""

import math
import time
import urllib.request
from pathlib import Path

import cv2
import legoeducation as le
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# --- Configuration ---------------------------------------------------------

# Update these to match the Connection Card plugged into the car's Double Motor
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "7572"

# Update these to match the Connection Card of the SEPARATE Single Motor hub
# (runs at a constant speed the whole time this script is running). This must
# be a different card than CARD_COLOR/CARD_SERIAL above if it's a different
# physical hub - run find_devices.py to read off its real values.
SINGLE_MOTOR_CARD_COLOR = le.LEGO_COLOR_AZURE
SINGLE_MOTOR_CARD_SERIAL = "3683"  # placeholder - replace with your Single Motor's actual card
SINGLE_MOTOR_SPEED = 50  # -100..100, constant while the script runs

CAMERA_INDEX = 0

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

MAX_RAISE = 0.35        # fraction of image height above the shoulder line for 100% forward throttle
STEER_GAIN = 140        # scales left/right wrist-height difference into a steering split
TOGETHER_DISTANCE = 0.15  # wrist-to-wrist distance (fraction of frame) below which hands count as "together"
BACKWARD_SPEED = 50     # constant reverse speed (%) while hands are together
SEND_THRESHOLD = 3      # only send a new BLE motor command if speed changed by more than this (%)

POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16


def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading pose landmarker model (first run only)...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def try_connect(device, card_color, card_serial, label):
    """Connect a LEGO device, treating any connection-time error (e.g. a
    firmware VersionMismatchError) the same as a failed scan instead of
    crashing the whole program."""
    try:
        device.connect(card_color=card_color, card_serial=card_serial)
    except Exception as exc:
        print(f"Could not connect to the {label}: {exc}")
        return False
    if not device.connected:
        print(f"Could not connect to the {label} - not found.")
        return False
    return True


def compute_speeds(pose_landmarks):
    """Map one detected pose's shoulder/wrist landmarks to (left, right) motor speeds (-100..100).

    - Hands together (close to each other) -> drive backward at BACKWARD_SPEED.
    - Otherwise, both hands raised above the shoulders -> drive forward, faster
      the higher they go (up to MAX_RAISE).
    - Raising one hand higher than the other steers toward that side.
    """
    lm = pose_landmarks[0]
    left_wrist, right_wrist = lm[POSE_LEFT_WRIST], lm[POSE_RIGHT_WRIST]

    shoulder_y = (lm[POSE_LEFT_SHOULDER].y + lm[POSE_RIGHT_SHOULDER].y) / 2
    left_raise = shoulder_y - left_wrist.y
    right_raise = shoulder_y - right_wrist.y

    hand_distance = math.hypot(left_wrist.x - right_wrist.x, left_wrist.y - right_wrist.y)

    if hand_distance < TOGETHER_DISTANCE:
        throttle = -BACKWARD_SPEED
    else:
        throttle = (left_raise + right_raise) / 2 / MAX_RAISE
        throttle = max(0.0, min(1.0, throttle)) * 100

    steer = (right_raise - left_raise) * STEER_GAIN

    left_speed = max(-100.0, min(100.0, throttle + steer))
    right_speed = max(-100.0, min(100.0, throttle - steer))
    return left_speed, right_speed


def main():
    ensure_model()
    landmarker = make_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    car = le.DoubleMotor()
    connected = try_connect(car, CARD_COLOR, CARD_SERIAL, "car")
    if not connected:
        print("Running in camera preview-only mode.")

    spinner = le.SingleMotor()
    spinner_connected = try_connect(
        spinner, SINGLE_MOTOR_CARD_COLOR, SINGLE_MOTOR_CARD_SERIAL, "Single Motor"
    )
    if spinner_connected:
        spinner.motor_run(speed=SINGLE_MOTOR_SPEED, blocking=False)

    last_left, last_right = 0.0, 0.0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            left_speed = right_speed = 0.0
            if result.pose_landmarks:
                left_speed, right_speed = compute_speeds(result.pose_landmarks)
                h, w = frame.shape[:2]
                for idx in (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER, POSE_LEFT_WRIST, POSE_RIGHT_WRIST):
                    lm = result.pose_landmarks[0][idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 8, (0, 255, 0), -1)

            if connected and (
                abs(left_speed - last_left) > SEND_THRESHOLD
                or abs(right_speed - last_right) > SEND_THRESHOLD
            ):
                # blocking=False: fire-and-forget, so the camera loop never waits on BLE
                car.motor_run(motor=le.MOTOR_LEFT, speed=left_speed, blocking=False)
                car.motor_run(motor=le.MOTOR_RIGHT, speed=right_speed, blocking=False)
                last_left, last_right = left_speed, right_speed

            cv2.putText(
                frame, f"L: {left_speed:.0f}%  R: {right_speed:.0f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow("Arm-Controlled Race Car (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if connected:
            try:
                car.motor_stop(motor=le.MOTOR_BOTH)
                car.disconnect()
            except Exception as exc:
                print(f"Error while stopping/disconnecting the car: {exc}")
        if spinner_connected:
            try:
                spinner.motor_stop()
                spinner.disconnect()
            except Exception as exc:
                print(f"Error while stopping/disconnecting the Single Motor: {exc}")
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
