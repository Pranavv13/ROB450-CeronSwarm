// pico1.ino
#include "command.h"

uint8_t packed_half[PACKED_BYTES_HALF];   // 256 bytes from pico2
uint8_t X[NUM_CHANNELS / 2];              // 512 channels on pico1

void setup() {
  Serial.begin(115200);    // optional debug
  Serial1.begin(115200);   // UART from pico2

  while (!Serial) {}

  Wire.begin();
  Wire1.begin();

  Serial.println("pico1 setup complete");
}

void loop() {
  // A) read 256 bytes from pico2
  readExactBytes(Serial1, packed_half, PACKED_BYTES_HALF);

  // B) unpack -> 512 X values
  buildX(packed_half, X, PACKED_BYTES_HALF);

  // C) apply to PCA9685 over I2C0/I2C1
  actionX(Wire, Wire1, X);
}
