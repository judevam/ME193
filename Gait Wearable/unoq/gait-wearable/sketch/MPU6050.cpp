#include "MPU6050.h"

namespace {
// Register addresses
const uint8_t SMPLRT_DIV = 0x19;
const uint8_t CONFIG = 0x1A;
const uint8_t GYRO_CONFIG = 0x1B;
const uint8_t ACCEL_CONFIG = 0x1C;
const uint8_t FIFO_EN = 0x23;
const uint8_t INT_STATUS = 0x3A;
const uint8_t ACCEL_XOUT_H = 0x3B;  // 14 bytes: accel XYZ, temperature, gyro XYZ
const uint8_t TEMP_OUT_H = 0x41;
const uint8_t USER_CTRL = 0x6A;
const uint8_t PWR_MGMT_1 = 0x6B;
const uint8_t FIFO_COUNTH = 0x72;
const uint8_t FIFO_R_W = 0x74;
const uint8_t WHO_AM_I = 0x75;

// Bits
const uint8_t DEVICE_RESET = 0x80;   // PWR_MGMT_1
const uint8_t CLOCK_PLL_XGYRO = 0x01;  // PWR_MGMT_1: more stable than the internal oscillator
const uint8_t DLPF_44HZ = 0x03;      // CONFIG: accel 44 Hz / gyro 42 Hz bandwidth, 1 kHz internal rate
const uint8_t FIFO_ACCEL_GYRO = 0x78;  // FIFO_EN: XG, YG, ZG and ACCEL
const uint8_t USER_FIFO_EN = 0x40;   // USER_CTRL
const uint8_t USER_FIFO_RESET = 0x04;  // USER_CTRL
const uint8_t FIFO_OFLOW_INT = 0x10;   // INT_STATUS

const int FIFO_SAMPLE_BYTES = 12;  // accel XYZ + gyro XYZ, 2 bytes each

const float ACCEL_COUNTS_PER_G[] = {16384, 8192, 4096, 2048};
const float GYRO_COUNTS_PER_DPS[] = {131, 65.5f, 32.8f, 16.4f};
}  // namespace

MPU6050::MPU6050(TwoWire& wire, uint8_t address) : wire_(wire), addr_(address) {}

bool MPU6050::begin(uint16_t sampleRateHz, AccelRange accel, GyroRange gyro) {
  wire_.begin();
  wire_.setClock(400000);

  if (!writeReg(PWR_MGMT_1, DEVICE_RESET)) return false;  // nothing answered
  delay(100);
  writeReg(PWR_MGMT_1, CLOCK_PLL_XGYRO);  // wake up (the chip starts asleep)
  delay(10);

  // With the low-pass filter on, the internal rate is 1 kHz: rate = 1000 / (1 + divider).
  rateHz_ = constrain(sampleRateHz, 4, 1000);
  writeReg(CONFIG, DLPF_44HZ);
  writeReg(SMPLRT_DIV, (uint8_t)(1000 / rateHz_ - 1));
  writeReg(ACCEL_CONFIG, (uint8_t)(accel << 3));
  writeReg(GYRO_CONFIG, (uint8_t)(gyro << 3));

  accelScale_ = 1.0f / ACCEL_COUNTS_PER_G[accel];
  gyroScale_ = 1.0f / GYRO_COUNTS_PER_DPS[gyro];
  delay(50);  // let the filter settle
  return true;
}

uint8_t MPU6050::whoAmI() { return readReg(WHO_AM_I); }

bool MPU6050::readRaw(Raw& out) {
  uint8_t b[14];
  if (!readRegs(ACCEL_XOUT_H, b, sizeof b)) return false;
  out = unpack(b, true);
  return true;
}

bool MPU6050::read(Sample& out) {
  Raw r;
  if (!readRaw(r)) return false;
  out = toUnits(r);
  return true;
}

float MPU6050::readTemperatureC() {
  uint8_t b[2];
  if (!readRegs(TEMP_OUT_H, b, 2)) return NAN;
  return (int16_t)((b[0] << 8) | b[1]) / 340.0f + 36.53f;
}

bool MPU6050::fifoBegin() {
  writeReg(FIFO_EN, 0);
  writeReg(USER_CTRL, USER_FIFO_RESET);
  delay(1);
  bool ok = writeReg(USER_CTRL, USER_FIFO_EN) && writeReg(FIFO_EN, FIFO_ACCEL_GYRO);
  readReg(INT_STATUS);  // reading clears an old overflow flag
  return ok;
}

void MPU6050::fifoReset() {
  writeReg(USER_CTRL, USER_FIFO_EN | USER_FIFO_RESET);
}

bool MPU6050::fifoOverflowed() { return readReg(INT_STATUS) & FIFO_OFLOW_INT; }

int MPU6050::readFifoRaw(Raw* out, int maxSamples) {
  uint8_t c[2];
  if (!readRegs(FIFO_COUNTH, c, 2)) return 0;
  int available = ((c[0] << 8) | c[1]) / FIFO_SAMPLE_BYTES;
  int n = min(available, maxSamples);
  uint8_t b[FIFO_SAMPLE_BYTES];
  for (int i = 0; i < n; i++) {
    // One sample per I2C read keeps us under the Wire library's buffer size.
    if (!readRegs(FIFO_R_W, b, FIFO_SAMPLE_BYTES)) return i;
    out[i] = unpack(b, false);
  }
  return n;
}

int MPU6050::readFifo(Sample* out, int maxSamples) {
  Raw raw[16];
  int total = 0;
  while (total < maxSamples) {
    int n = readFifoRaw(raw, min(16, maxSamples - total));
    for (int i = 0; i < n; i++) out[total + i] = toUnits(raw[i]);
    total += n;
    if (n < 16) break;  // FIFO is empty
  }
  return total;
}

void MPU6050::calibrateGyro(int samples) {
  long sx = 0, sy = 0, sz = 0;
  Raw r;
  int got = 0;
  unsigned long period = 1000 / rateHz_;
  for (int i = 0; i < samples; i++) {
    if (readRaw(r)) {
      sx += r.gx;
      sy += r.gy;
      sz += r.gz;
      got++;
    }
    delay(period);
  }
  if (got > 0) {
    biasX_ = (float)sx / got;
    biasY_ = (float)sy / got;
    biasZ_ = (float)sz / got;
  }
}

MPU6050::Sample MPU6050::toUnits(const Raw& r) const {
  Sample s;
  s.ax = r.ax * accelScale_;
  s.ay = r.ay * accelScale_;
  s.az = r.az * accelScale_;
  s.gx = (r.gx - biasX_) * gyroScale_;
  s.gy = (r.gy - biasY_) * gyroScale_;
  s.gz = (r.gz - biasZ_) * gyroScale_;
  return s;
}

String MPU6050::dump() {
  const uint8_t regs[] = {SMPLRT_DIV, CONFIG, GYRO_CONFIG, ACCEL_CONFIG, FIFO_EN,
                          USER_CTRL, PWR_MGMT_1, 0x6C /* PWR_MGMT_2 */, WHO_AM_I};
  String out;
  for (uint8_t r : regs) out += String(r, HEX) + "=" + String(readReg(r), HEX) + " ";
  uint8_t c[2];
  readRegs(FIFO_COUNTH, c, 2);
  out += "fifo=" + String((c[0] << 8) | c[1]) + " data=";
  uint8_t b[14];
  bool ok = readRegs(ACCEL_XOUT_H, b, sizeof b);
  for (uint8_t v : b) out += String(v, HEX) + ",";
  out += ok ? " ok" : " READFAIL";
  out += " bias=" + String((int)biasX_) + "," + String((int)biasY_) + "," + String((int)biasZ_);
  return out;
}

MPU6050::Raw MPU6050::unbiased(const Raw& r) const {
  Raw u = r;
  u.gx = (int16_t)lroundf(r.gx - biasX_);
  u.gy = (int16_t)lroundf(r.gy - biasY_);
  u.gz = (int16_t)lroundf(r.gz - biasZ_);
  return u;
}

// The chip sends each value high byte first. The data registers have the temperature
// between accel and gyro; the FIFO (configured for accel + gyro only) doesn't.
MPU6050::Raw MPU6050::unpack(const uint8_t* b, bool hasTemp) {
  // Not called "word": Arduino.h has a word() macro that would silently replace it.
  auto be16 = [&](int i) { return (int16_t)((b[i] << 8) | b[i + 1]); };
  int g = hasTemp ? 8 : 6;
  return Raw{be16(0), be16(2), be16(4), be16(g), be16(g + 2), be16(g + 4)};
}

bool MPU6050::writeReg(uint8_t reg, uint8_t value) {
  wire_.beginTransmission(addr_);
  wire_.write(reg);
  wire_.write(value);
  return wire_.endTransmission() == 0;
}

bool MPU6050::readRegs(uint8_t reg, uint8_t* buf, uint8_t len) {
  wire_.beginTransmission(addr_);
  wire_.write(reg);
  if (wire_.endTransmission(false) != 0) return false;  // repeated start
  if (wire_.requestFrom(addr_, len) != len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = wire_.read();
  return true;
}

uint8_t MPU6050::readReg(uint8_t reg) {
  uint8_t v = 0;
  readRegs(reg, &v, 1);
  return v;
}
