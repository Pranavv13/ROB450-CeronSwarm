#include "command.h"

// A. Serial Read USB input from PC: read of one frame (512 bytes) - read from USB "whatever stream, n bytes are read + save at the destination buffer (dst)"
bool readExactBytes(Stream& s, uint8_t* dst, int n) {
  int received = 0;
  while (received < n) {
    if (s.available()) {
      dst[received++] = s.read();         // read one by one until the end (n)
    }
  }
  return true;
}

// B. Pack X from packed 512 bytes (PACKED_BYTES)
void buildX(const uint8_t* packed_in, uint8_t* X_out, int pakced_bytes) {
  for (int i = 0; i < pakced_bytes; ++i) {
    uint8_t b = packed_in[i];
    X_out[2 * i]     = b & 0x0F;          // low nibble: first 4 bit
    X_out[2 * i + 1] = (b >> 4) & 0x0F;   // push 4 bit right (b >> 4) => high nibble: last 4 bit 
  }
}

// C. X to Action: PCA9685 over I2C0/I2C1
void actionX(TwoWire& i2c0, TwoWire& i2c1, const uint8_t* X512) {
  const uint8_t BASE_ADDR = 0x40;     // A5..A0 = 000000
  const uint8_t REG_LED0  = 0x06;     // LED0_ON_L
  const int X_PER_BUS = 256;          // 32 PCA * 8 Xi
  const int X_PER_PCA = 8;            // 16ch / (2ch per Xi)

  auto magToPwm = [](uint8_t m) -> uint16_t {
    if (m > 7) m = 7;
    return (uint16_t)((uint32_t)m * 4095u / 7u); // 0..4095: pwm action output
  };

  auto setPWM = [&](TwoWire& bus, uint8_t addr, uint8_t ch, uint16_t offCount) {

    uint8_t reg = (uint8_t)(REG_LED0 + 4 * ch);
    bus.beginTransmission(addr);
    bus.write(reg);
    bus.write(0x00); bus.write(0x00);                 // ON_L, ON_H
    bus.write((uint8_t)(offCount & 0xFF));            // OFF_L
    bus.write((uint8_t)((offCount >> 8) & 0x0F));     // OFF_H
    bus.endTransmission();
  };

  // 1) X --> devide it by two: X_i2c0, X_i2c1
  for (int whichBus = 0; whichBus < 2; ++whichBus) {
    TwoWire& bus = (whichBus == 0) ? i2c0 : i2c1;
    const uint8_t* X_i2c = X512 + (whichBus * X_PER_BUS);  // X_i2c0 or X_i2c1

    // 2) 8 values per a pca9685 
    for (int i = 0; i < X_PER_BUS; ++i) {
      // 3) increase i, i2c address change 
      uint8_t addr = (uint8_t)(BASE_ADDR + (i / X_PER_PCA)); // 0x40..0x5F
      uint8_t pair = (uint8_t)(i % X_PER_PCA);               // 0..7
      uint8_t chA  = (uint8_t)(2 * pair);                    // 0,2,...,14
      uint8_t chB  = (uint8_t)(chA + 1);                     // 1,3,...,15

      // 4) polarity logic
      uint8_t xi = X_i2c[i];          // 0..15
      bool neg = (xi <= 7);
      uint8_t m = neg ? xi : (uint8_t)(xi - 8);   // 0..7
      uint16_t pwm = magToPwm(m);

      if (neg) {
        setPWM(bus, addr, chA, pwm);
        setPWM(bus, addr, chB, 0);
      } else {
        setPWM(bus, addr, chA, 0);
        setPWM(bus, addr, chB, pwm);
      }
    }
  }
}
