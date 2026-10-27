// Minimal MPU-6050 driver (accelerometer + gyroscope over I2C).
//
// Two ways to read:
//   read(sample)          the newest sample, straight from the data registers
//   readFifo(samples, n)  every sample since the last call, from the chip's 1 KB FIFO.
//                         No samples are lost as long as you call it at least every
//                         ~0.8 s at 100 Hz (85 samples fit). Use this for recording.
//
// Register numbers are from the InvenSense "MPU-6000/6050 Register Map" (RM-MPU-6000A).
#pragma once

#include <Arduino.h>
#include <Wire.h>

class MPU6050 {
 public:
  enum AccelRange { ACCEL_2G = 0, ACCEL_4G = 1, ACCEL_8G = 2, ACCEL_16G = 3 };
  enum GyroRange { GYRO_250DPS = 0, GYRO_500DPS = 1, GYRO_1000DPS = 2, GYRO_2000DPS = 3 };

  // Raw readings straight from the chip (signed 16-bit counts).
  struct Raw {
    int16_t ax, ay, az, gx, gy, gz;
  };

  // Readings in physical units.
  struct Sample {
    float ax, ay, az;  // g
    float gx, gy, gz;  // degrees per second, gyro bias removed
  };

  // address: 0x68 if the AD0 pin is low or unconnected, 0x69 if it's high.
  explicit MPU6050(TwoWire& wire = Wire, uint8_t address = 0x68);

  // Wake the chip and configure it. Returns false if nothing answers on I2C.
  // sampleRateHz: 4..1000. The digital low-pass filter is set to ~44 Hz, which suits 100 Hz.
  bool begin(uint16_t sampleRateHz = 100, AccelRange accel = ACCEL_8G, GyroRange gyro = GYRO_1000DPS);

  // WHO_AM_I register. A genuine MPU-6050 says 0x68; clones often say 0x70, 0x72 or 0x98.
  uint8_t whoAmI();

  bool read(Sample& out);
  bool readRaw(Raw& out);
  float readTemperatureC();

  // FIFO: start it with fifoBegin(), then readFifo() returns how many samples it copied
  // (up to maxSamples). Check fifoOverflowed() to know if samples were lost.
  bool fifoBegin();
  int readFifoRaw(Raw* out, int maxSamples);
  int readFifo(Sample* out, int maxSamples);
  bool fifoOverflowed();  // true once since the last call if the FIFO filled up
  void fifoReset();

  // Average the gyro for `samples` readings while the sensor is perfectly still,
  // and subtract that bias from every later reading. Takes samples / rate seconds.
  void calibrateGyro(int samples = 200);

  Sample toUnits(const Raw& r) const;
  Raw unbiased(const Raw& r) const;  // raw counts with the gyro bias subtracted

  // Debugging: the config registers and the 14 data bytes, as hex text.
  String dump();
  float accelScale() const { return accelScale_; }  // g per count
  float gyroScale() const { return gyroScale_; }    // deg/s per count

 private:
  bool writeReg(uint8_t reg, uint8_t value);
  bool readRegs(uint8_t reg, uint8_t* buf, uint8_t len);
  uint8_t readReg(uint8_t reg);
  static Raw unpack(const uint8_t* b, bool hasTemp);

  TwoWire& wire_;
  uint8_t addr_;
  uint16_t rateHz_ = 100;
  float accelScale_ = 1.0f / 4096;
  float gyroScale_ = 1.0f / 32.8f;
  float biasX_ = 0, biasY_ = 0, biasZ_ = 0;  // gyro bias in counts
};
