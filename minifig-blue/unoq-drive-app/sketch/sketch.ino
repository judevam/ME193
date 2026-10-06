// Runs on the UNO Q's microcontroller. Turns drive(left, right) commands from
// python/main.py into PWM on a Cytron Maker Drive (two H-bridge channels).
//
// Maker Drive truth table, per motor (inputs A and B):
//   PWM on A, B low  -> forward at that duty
//   A low, PWM on B  -> backward
//   A low, B low     -> brake
// So each motor needs TWO PWM pins. UNO Q PWM pins: 3, 5, 6, 9, 10, 11.
//
// Wiring: UNO Q pin -> Maker Drive input, and UNO Q GND -> Maker Drive GND
// (common ground is required). The Maker Drive supply (VB+ / VB-) must stay
// between 2.5 and 9.5 V, 1 A per motor.
#include <Arduino_RouterBridge.h>

const int M1A = 3;   // left motor
const int M1B = 5;
const int M2A = 6;   // right motor
const int M2B = 9;

// If a wheel spins the wrong way when told to go forward, flip its flag
// instead of rewiring.
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = false;

// Stop if no drive command arrives for this long (laptop crashed, Wi-Fi
// dropped, app stopped). python/main.py re-sends every command while moving.
const unsigned long FAILSAFE_MS = 500;

volatile unsigned long last_cmd_ms = 0;
volatile bool moving = false;

void drive(int left, int right);
void stop_motors();

// NOTE: no pinMode() on these pins. On the UNO Q, calling pinMode() before
// analogWrite() breaks PWM (analogWrite configures the pin itself).
void set_motor(int pinA, int pinB, int speed, bool invert) {
  if (invert) speed = -speed;
  speed = constrain(speed, -100, 100);
  int duty = abs(speed) * 255 / 100;
  if (speed > 0) {
    analogWrite(pinB, 0);
    analogWrite(pinA, duty);
  } else if (speed < 0) {
    analogWrite(pinA, 0);
    analogWrite(pinB, duty);
  } else {
    analogWrite(pinA, 0);  // low/low = brake
    analogWrite(pinB, 0);
  }
}

void drive(int left, int right) {
  set_motor(M1A, M1B, left, INVERT_LEFT);
  set_motor(M2A, M2B, right, INVERT_RIGHT);
  moving = (left != 0 || right != 0);
  last_cmd_ms = millis();
}

void stop_motors() {
  drive(0, 0);
}

void setup() {
  stop_motors();  // also configures the four pins as PWM outputs, motors off

  Bridge.begin();
  Bridge.provide("drive", drive);
  Bridge.provide("stop_motors", stop_motors);
}

void loop() {
  if (moving && millis() - last_cmd_ms > FAILSAFE_MS) {
    stop_motors();
  }
}
