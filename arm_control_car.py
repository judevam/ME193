"""Steer a LEGO Education Double Motor car with your arms, tracked via mediapipe Pose.

Raise both arms above your shoulders to drive forward - the higher they go
(up to MAX_RAISE), the faster the car goes. Raise one arm higher than the
other to steer: e.g. right arm higher than left turns the car right.
Drop both arms to stop.

A separate Single Motor connects independently and spins at a constant speed
(SINGLE_MOTOR_SPEED) for the entire time this script is running, regardless
of arm position.

See README.md for setup and a discussion of the sync/async model and the
mediapipe model's training and limitations.
"""

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

# Update these to match the Connection Card plugged into the extra Single Motor
# (runs at a constant speed the whole time this script is running)
SINGLE_MOTOR_CARD_COLOR = le.LEGO_COLOR_AZURE
SINGLE_MOTOR_CARD_SERIAL = "3683"
SINGLE_MOTOR_SPEED = 50  # -100..100, constant while the script runs

CAMERA_INDEX = 0

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

MAX_RAISE = 0.35   # fraction of image height above the shoulder line for 100% throttle
STEER_GAIN = 140   # scales left/right wrist-height difference into a steering split
SEND_THRESHOLD = 3  # only send a new BLE motor command if speed changed by more than this (%)

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


def compute_speeds(pose_landmarks):
    """Map one detected pose's shoulder/wrist landmarks to (left, right) motor speeds (-100..100)."""
    lm = pose_landmarks[0]
    shoulder_y = (lm[POSE_LEFT_SHOULDER].y + lm[POSE_RIGHT_SHOULDER].y) / 2
    left_raise = shoulder_y - lm[POSE_LEFT_WRIST].y
    right_raise = shoulder_y - lm[POSE_RIGHT_WRIST].y

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
    car.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    connected = car.connected
    if not connected:
        print("Could not connect to the car - running in camera preview-only mode.")

    spinner = le.SingleMotor()
    spinner.connect(card_color=SINGLE_MOTOR_CARD_COLOR, card_serial=SINGLE_MOTOR_CARD_SERIAL)
    spinner_connected = spinner.connected
    if spinner_connected:
        spinner.motor_run(speed=SINGLE_MOTOR_SPEED, blocking=False)
    else:
        print("Could not connect to the Single Motor - it will not run.")

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
            car.motor_stop(motor=le.MOTOR_BOTH)
            car.disconnect()
        if spinner_connected:
            spinner.motor_stop()
            spinner.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
