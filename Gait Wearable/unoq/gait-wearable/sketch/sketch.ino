#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include "MPU6050.h"

// MPU-6050 IMU for the gait wearable. Python pulls samples with read_samples().
//
// WIRING:
//   MPU-6050 VCC -> 3.3V     GND -> GND
//   MPU-6050 SDA -> SDA      SCL -> SCL   (the pins next to AREF on the UNO header)
//   AD0 unconnected (address 0x68). Most breakout boards already have pull-ups.
//   Using the Qwiic connector instead? Change Wire to Wire1 below.
const uint16_t RATE_HZ = 100;
const int MAX_PER_CALL = 40;  // samples per read_samples() call (0.4 s at 100 Hz)

MPU6050 imu(Wire);
bool imu_ok = false;
String imu_info;

// 8 x 13 LED matrix. The animations live in python/matrix_art.py; Python sends frames here.
const int MATRIX_PIXELS = 8 * 13;
Arduino_LED_Matrix matrix;

String imu_status();
String read_samples();
String imu_debug();
void draw(String frame);

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0..7
  matrix.clear();

  // Set the IMU up before the Bridge starts, so only Bridge calls touch I2C after this.
  imu_ok = imu.begin(RATE_HZ, MPU6050::ACCEL_8G, MPU6050::GYRO_1000DPS);
  if (imu_ok) {
    imu.calibrateGyro(200);  // 2 s: keep the sensor still at power-on
    imu.fifoBegin();
    imu_info = "ok whoami=0x" + String(imu.whoAmI(), HEX) + " rate=" + String(RATE_HZ);
  } else {
    imu_info = "no MPU-6050 at 0x68: check wiring";
  }

  Bridge.begin();
  Bridge.provide("imu_status", imu_status);
  Bridge.provide("read_samples", read_samples);
  Bridge.provide("imu_debug", imu_debug);
  Bridge.provide("draw", draw);
  Monitor.begin();
  Monitor.println("gait wearable sketch: " + imu_info);
}

// Latest sample and sample count, saved by read_samples() for loop() to print.
// loop() never touches I2C itself, so it can't get in the way of the Bridge calls.
volatile int16_t latest[6];
volatile unsigned long samples_since_print = 0;

// Whole number scaled by 10^decimals -> text, e.g. (-1234, 3) -> "-1.234".
// (Avoids printing floats, which this core does unreliably.)
String fixed(long v, int decimals) {
  String sign = v < 0 ? "-" : " ";
  v = labs(v);
  long div = 1;
  for (int i = 0; i < decimals; i++) div *= 10;
  String frac = String(v % div);
  while ((int)frac.length() < decimals) frac = "0" + frac;
  return sign + String(v / div) + "." + frac;
}

void loop() {
  static unsigned long last = millis();
  if (millis() - last < 1000) return;
  last += 1000;
  if (!imu_ok) {
    Monitor.println(imu_info);
    return;
  }
  long mg[3], ddps[3];  // milli-g and tenths of deg/s
  for (int i = 0; i < 3; i++) {
    mg[i] = (long)latest[i] * 1000 / 4096;  // 4096 counts per g at 8 g
    ddps[i] = (long)latest[i + 3] * 100 / 328;  // 32.8 counts per deg/s at 1000 dps
  }
  Monitor.println(String((unsigned long)samples_since_print) + " Hz  accel g [" + fixed(mg[0], 3) + " " +
                  fixed(mg[1], 3) + " " + fixed(mg[2], 3) + "]  gyro dps [" + fixed(ddps[0], 1) +
                  " " + fixed(ddps[1], 1) + " " + fixed(ddps[2], 1) + "]");
  samples_since_print = 0;
}

String imu_status() { return imu_info; }

String imu_debug() { return imu.dump(); }

// One frame: 104 digits '0'..'7' (brightness), row by row, 13 per row.
void draw(String frame) {
  uint8_t pixels[MATRIX_PIXELS] = {0};
  for (int i = 0; i < MATRIX_PIXELS && i < (int)frame.length(); i++) {
    char c = frame[i];
    pixels[i] = (c >= '0' && c <= '7') ? c - '0' : 0;
  }
  matrix.draw(pixels);
}

// Everything in the FIFO since the last call, oldest first:
//   "<overflow 0|1>|ax,ay,az,gx,gy,gz;ax,ay,..."
// Raw counts (whole numbers; this core's float-to-text is unreliable), gyro bias removed.
// Python divides by ACCEL_COUNTS_PER_G (4096 at 8 g) and GYRO_COUNTS_PER_DPS (32.8 at 1000 dps).
// overflow=1 means samples were lost because the calls came too slowly.
String read_samples() {
  if (!imu_ok) return "E|";
  bool overflow = imu.fifoOverflowed();
  if (overflow) imu.fifoReset();

  static MPU6050::Raw buf[MAX_PER_CALL];
  int n = imu.readFifoRaw(buf, MAX_PER_CALL);
  String out = overflow ? "1|" : "0|";
  out.reserve(2 + n * 38);
  for (int i = 0; i < n; i++) {
    MPU6050::Raw r = imu.unbiased(buf[i]);
    if (i) out += ';';
    out += String(r.ax) + ',' + String(r.ay) + ',' + String(r.az) + ',' +
           String(r.gx) + ',' + String(r.gy) + ',' + String(r.gz);
    if (i == n - 1) {
      latest[0] = r.ax; latest[1] = r.ay; latest[2] = r.az;
      latest[3] = r.gx; latest[4] = r.gy; latest[5] = r.gz;
    }
  }
  samples_since_print += n;
  return out;
}
