#pragma once
#include <Arduino.h>
#include <Wire.h>

// A. Serial read USB input from PC: system sizes (1024 electromagnet array)
constexpr int NUM_CHANNELS  = 1024;                         // controlling 1024 electromagnets
constexpr int PACKED_BYTES = NUM_CHANNELS / 2;              // using 512 bytes for 1024 electromagnets

// B. Pack X:  sizes for pico2 and pico1 (512 electromagnets)
constexpr int NUM_CHANNELS_PICO = 512;                      // 512 electromagnet channels on pico2
constexpr int PACKED_BYTES_HALF = PACKED_BYTES / 2;         // 256 bytes for 512 electromagnet

// C. X to Action: PCA9685 (Adafruit 16-Channel 12-bit PWM/Servo Driver - I2C interface)
constexpr int NUM_PCA = 32;                                 // total # of PCA9685 per I2C bus 
constexpr int X_PER_PCA = 8;                                // 8 electromagnets per PCA9685

constexpr uint8_t PCA_BASE_ADDR = 0x40;                     // the PCA9685's address A5..A0 = 000000
constexpr int ACTION_X_LEN = NUM_PCA * X_PER_PCA;           // 256 per I2C bus

//-----------////-----------////-----------//

// A. Serial read USB input from PC: blocking read of one frame (512 bytes) in real time 
bool readExactBytes(Stream& s, uint8_t* dst, int n);

// B. Pack X: 512 bytes data -> X for pico1, X for pico2
void buildX(const uint8_t* packed_in, uint8_t* X_out, int packed_bytes);

// C. X to Action: PCA9685 over I2C0/I2C1
void actionX(TwoWire& bus0, TwoWire& bus1, const uint8_t* X512);

