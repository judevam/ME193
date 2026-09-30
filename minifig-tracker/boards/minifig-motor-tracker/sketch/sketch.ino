#include <Arduino_RouterBridge.h>

// SparkFun Maker Drive dual H-bridge -- confirm these match your wiring.
const int MOTOR_A_DIR = 4;
const int MOTOR_A_PWM = 5;
const int MOTOR_B_DIR = 7;
const int MOTOR_B_PWM = 6;

void drive(int direction, int speed);
void stop_motors();

void setup() {
  pinMode(MOTOR_A_DIR, OUTPUT);
  pinMode(MOTOR_A_PWM, OUTPUT);
  pinMode(MOTOR_B_DIR, OUTPUT);
  pinMode(MOTOR_B_PWM, OUTPUT);

  stop_motors();

  Bridge.begin();
  Bridge.provide("drive", drive);
}

void loop() {}

void stop_motors() {
  analogWrite(MOTOR_A_PWM, 0);
  analogWrite(MOTOR_B_PWM, 0);
}

// direction: 1 = forward, -1 = backward, 0 = stop. speed: 0-255 PWM duty cycle.
void drive(int direction, int speed) {
  speed = constrain(speed, 0, 255);

  if (direction > 0) {
    digitalWrite(MOTOR_A_DIR, HIGH);
    digitalWrite(MOTOR_B_DIR, HIGH);
    analogWrite(MOTOR_A_PWM, speed);
    analogWrite(MOTOR_B_PWM, speed);
  } else if (direction < 0) {
    digitalWrite(MOTOR_A_DIR, LOW);
    digitalWrite(MOTOR_B_DIR, LOW);
    analogWrite(MOTOR_A_PWM, speed);
    analogWrite(MOTOR_B_PWM, speed);
  } else {
    stop_motors();
  }
}
