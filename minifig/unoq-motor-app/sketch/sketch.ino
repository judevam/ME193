#include <Arduino_RouterBridge.h>

// Cytron Maker Drive (MX1508) -- each motor has two inputs, both PWM pins.
// Forward: A = PWM, B = LOW. Backward: A = LOW, B = PWM. Stop: both LOW (coast).
const int MOTOR1_A = 3;  // -> M1A
const int MOTOR1_B = 5;  // -> M1B
const int MOTOR2_A = 6;  // -> M2A
const int MOTOR2_B = 9;  // -> M2B

void drive(int direction, int speed);
void stop_motors();

void setup() {
  pinMode(MOTOR1_A, OUTPUT);
  pinMode(MOTOR1_B, OUTPUT);
  pinMode(MOTOR2_A, OUTPUT);
  pinMode(MOTOR2_B, OUTPUT);

  stop_motors();

  Bridge.begin();
  Bridge.provide("drive", drive);
  Monitor.begin();
  Monitor.println("motor sketch ready");
}

void loop() {}

void stop_motors() {
  analogWrite(MOTOR1_A, 0);
  analogWrite(MOTOR1_B, 0);
  analogWrite(MOTOR2_A, 0);
  analogWrite(MOTOR2_B, 0);
}

// direction: 1 = forward, -1 = backward, 0 = stop. speed: 0-255 PWM duty cycle.
void drive(int direction, int speed) {
  speed = constrain(speed, 0, 255);
  Monitor.print("drive dir=");
  Monitor.print(direction);
  Monitor.print(" speed=");
  Monitor.println(speed);

  if (direction > 0) {
    analogWrite(MOTOR1_B, 0);
    analogWrite(MOTOR2_B, 0);
    analogWrite(MOTOR1_A, speed);
    analogWrite(MOTOR2_A, speed);
  } else if (direction < 0) {
    analogWrite(MOTOR1_A, 0);
    analogWrite(MOTOR2_A, 0);
    analogWrite(MOTOR1_B, speed);
    analogWrite(MOTOR2_B, speed);
  } else {
    stop_motors();
  }
}
