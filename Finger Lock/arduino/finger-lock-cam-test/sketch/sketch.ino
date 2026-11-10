#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

// ---- Micro servo (latch) on D2. Red -> 5V, brown/black -> GND, signal -> D2.
// D2 has no hardware PWM, so the 50 Hz servo pulses are made in software.
// Pulses are only sent for HOLD_MS after each move, so the servo does not
// buzz or draw current while idle (the latch holds mechanically).
const int SERVO_PIN = 2;
const unsigned long HOLD_MS = 1200;
const int MIN_US = 500;    // pulse width at 0 degrees
const int MAX_US = 2400;   // pulse width at 180 degrees

volatile int targetAngle = 90;
volatile unsigned long pulseUntil = 0;  // 0 = not driving the servo yet

// ---- ARM button on D7: one leg to D7, the other leg to GND.
// Uses the internal pull-up, so the pin reads HIGH at rest and LOW while pressed.
// The sketch debounces it and remembers one press; Python asks for it with Bridge.call("button").
const int BUTTON_PIN = 7;
const unsigned long DEBOUNCE_MS = 40;
volatile bool buttonEvent = false;
int buttonRaw = HIGH;
int buttonStable = HIGH;
unsigned long buttonChangedAt = 0;

// ---- Onboard 8x13 LED matrix: lock status display
const int ROWS = 8;
const int COLS = 13;
Arduino_LED_Matrix matrix;
volatile bool alarmOn = false;
bool alarmPhase = false;
unsigned long lastAlarmToggle = 0;

// 5x7 digit font, 0-5
const uint8_t FONT[6][7] = {
  {0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110},
  {0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
  {0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111},
  {0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110},
  {0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010},
  {0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110},
};

const char* PADLOCK[ROWS] = {
  "....#####....",
  "...##...##...",
  "...##...##...",
  "..#########..",
  "..#########..",
  "..####.####..",
  "..####.####..",
  "..#########..",
};
const char* CHECK[ROWS] = {
  "............#",
  "...........##",
  ".#........##.",
  ".##......##..",
  "..##....##...",
  "...##..##....",
  "....####.....",
  ".....##......",
};

void set_servo(int angle);
void set_matrix(int mode, int val);
int get_button();

void pollButton() {
  int raw = digitalRead(BUTTON_PIN);
  if (raw != buttonRaw) {
    buttonRaw = raw;
    buttonChangedAt = millis();
  }
  if (millis() - buttonChangedAt > DEBOUNCE_MS && raw != buttonStable) {
    buttonStable = raw;
    if (buttonStable == LOW) {  // a clean press
      buttonEvent = true;
      Monitor.println("button pressed");
    }
  }
}

void showPattern(const char** rows) {
  uint8_t f[ROWS * COLS];
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) f[r * COLS + c] = (rows[r][c] == '#') ? 7 : 0;
  matrix.draw(f);
}

void showX() {
  uint8_t f[ROWS * COLS] = {0};
  for (int r = 0; r < ROWS; r++) {
    f[r * COLS + 2 + r] = 7;
    f[r * COLS + 9 - r] = 7;
  }
  matrix.draw(f);
}

void showDigit(int d) {
  uint8_t f[ROWS * COLS] = {0};
  d = constrain(d, 0, 5);
  for (int r = 0; r < 7; r++)
    for (int c = 0; c < 5; c++)
      if (FONT[d][r] & (1 << (4 - c))) f[r * COLS + 4 + c] = 7;
  matrix.draw(f);
}

// three boxes, filled for each digit entered so far
void showProgress(int entered) {
  uint8_t f[ROWS * COLS] = {0};
  for (int i = 0; i < 3; i++) {
    for (int r = 2; r <= 5; r++)
      for (int c = 0; c < 3; c++) {
        bool edge = (r == 2 || r == 5 || c == 0 || c == 2);
        if (i < entered || edge) f[r * COLS + 1 + i * 4 + c] = (i < entered) ? 7 : 2;
      }
  }
  matrix.draw(f);
}

void setup() {
  pinMode(SERVO_PIN, OUTPUT);
  digitalWrite(SERVO_PIN, LOW);  // no pulses until the first command: no jump at boot
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  matrix.begin();
  matrix.clear();

  Bridge.begin();
  Bridge.provide("servo", set_servo);
  Bridge.provide("matrix", set_matrix);
  Bridge.provide("button", get_button);
  Monitor.begin();
  Monitor.println("finger lock sketch ready");
}

void loop() {
  pollButton();
  if (alarmOn && millis() - lastAlarmToggle > 250) {
    lastAlarmToggle = millis();
    alarmPhase = !alarmPhase;
    if (alarmPhase) showX(); else matrix.clear();
  }
  if (millis() < pulseUntil) {
    int us = map(targetAngle, 0, 180, MIN_US, MAX_US);
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(us);
    digitalWrite(SERVO_PIN, LOW);
    delay(19);  // ~20 ms frame
  } else {
    delay(5);
  }
}

// 1 if the button was pressed since the last call, else 0
int get_button() {
  if (buttonEvent) {
    buttonEvent = false;
    return 1;
  }
  return 0;
}

// angle: 0-180 degrees
void set_servo(int angle) {
  angle = constrain(angle, 0, 180);
  targetAngle = angle;
  pulseUntil = millis() + HOLD_MS;
  Monitor.print("servo angle=");
  Monitor.println(angle);
}

// mode: 0 clear, 1 padlock (locked), 2 progress (val = digits entered),
//       3 check mark (open), 4 X (denied), 5 flashing X (alarm), 6 digit (val = 0-5)
void set_matrix(int mode, int val) {
  alarmOn = (mode == 5);
  switch (mode) {
    case 1: showPattern(PADLOCK); break;
    case 2: showProgress(val); break;
    case 3: showPattern(CHECK); break;
    case 4: showX(); break;
    case 5: alarmPhase = true; lastAlarmToggle = millis(); showX(); break;
    case 6: showDigit(val); break;
    default: matrix.clear();
  }
}
