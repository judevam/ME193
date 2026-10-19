#include <Arduino_RouterBridge.h>

// HC-SR04 ultrasonic sensor + passive buzzer for the sumo robot.
//
// WIRING / VOLTAGE (TODO(hardware)):
//   HC-SR04 VCC  -> 5V          HC-SR04 GND -> GND
//   HC-SR04 TRIG -> TRIG_PIN    (3.3 V from the UNO Q is enough to trigger it)
//   HC-SR04 ECHO -> voltage divider -> ECHO_PIN
//     The UNO Q's I/O is 3.3 V. The HC-SR04's ECHO output is 5 V, so put a
//     divider on it: ECHO --1k-- ECHO_PIN --2k-- GND  (5 V * 2/3 = 3.3 V).
//     Or use a 3.3 V part (HC-SR04P / RCWL-1601) and skip the divider.
//   Buzzer (passive) + -> BUZZER_PIN, - -> GND
const int TRIG_PIN = 2;    // TODO(hardware)
const int ECHO_PIN = 4;    // TODO(hardware)
const int BUZZER_PIN = 5;  // TODO(hardware)

// Give up waiting for an echo after this long (~2 m there and back).
const unsigned long ECHO_TIMEOUT_US = 12000;

float read_cm();
void play_tone(int freq, int ms);

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);

  Bridge.begin();
  Bridge.provide("read_cm", read_cm);
  Bridge.provide("play_tone", play_tone);
  Monitor.begin();
  Monitor.println("sumo ultrasonic sketch ready");
}

void loop() {}

// Distance in cm, or -1 if no echo came back.
float read_cm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long start = micros();
  while (digitalRead(ECHO_PIN) == LOW) {
    if (micros() - start > ECHO_TIMEOUT_US) return -1;
  }
  unsigned long rise = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {
    if (micros() - rise > ECHO_TIMEOUT_US) return -1;
  }
  return (micros() - rise) / 58.0;  // 58 us per cm, there and back
}

// Start a note and return immediately. freq 0 = silence.
void play_tone(int freq, int ms) {
  if (freq > 0) {
    tone(BUZZER_PIN, freq, ms);
  } else {
    noTone(BUZZER_PIN);
  }
}
