import legoeducation as le
import time

# Update these to match the Connection Card plugged into the Single Motor
CARD_COLOR = le.LEGO_COLOR_AZURE
CARD_SERIAL = "3683"

# Connect to the Single Motor
motor = le.SingleMotor()
motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

if not motor.connected:
    print("Error connecting to Single Motor.")
    exit(1)

motor.motor_reset_relative_position()
motor.motor_run(speed=20)  # start slow

try:
    for _ in range(50):  # ~5 seconds
        print(f"Current position: {motor.motor.position}")
        if motor.motor.position > 360:
            motor.motor_run(speed=80)  # speed up after one rotation
        time.sleep(0.1)
finally:
    motor.motor_stop()
    motor.disconnect()
