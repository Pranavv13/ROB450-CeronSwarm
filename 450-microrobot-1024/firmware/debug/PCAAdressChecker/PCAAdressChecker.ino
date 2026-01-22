#include <Wire.h>

/*
  PCA9685 I2C Address Scanner (0x40–0x5F)

  Recap:
  This sketch scans all possible PCA9685 I2C addresses determined by the A5..A0 pins.
  PCA9685 base address is 0x40, and up to 32 devices can be addressed (0x40–0x5F).
  For each address, the code checks whether a device responds on the I2C bus.

  Output:
  - If a device does not respond, it prints the corresponding A5..A0 pin pattern
    (as T/F for HIGH/LOW) and the missing address.
  - At the end, it reports how many devices are missing.
  - If none are missing, it confirms that all 32 devices are present.

  Usage:
  Upload this sketch and open the Serial Monitor at 115200 baud.
  Make sure all PCA9685 boards are powered and connected to the same I2C bus.
*/

// Convert a 6-bit value into a string representing A5..A0 (T = HIGH, F = LOW)
static String tf6(uint8_t bits) {
  String s = "";
  for (int i = 5; i >= 0; --i) {
    s += ((bits >> i) & 1) ? 'T' : 'F';
  }
  return s; // Example: "TFFTTF" corresponds to A5..A0
}

void setup() {
  Wire.begin();          // Initialize I2C bus
  delay(100);            // Allow bus to stabilize
  Serial.begin(115200);  // Initialize serial communication

  // Wait until the serial monitor is connected
  while (!Serial);

  Serial.println("Checking PCA9685 A5..A0 = 000000..011111 (0x40..0x5F)");

  int missing = 0;

  // Scan all 32 possible address offsets (A5..A0 = 000000..011111)
  for (uint8_t off = 0; off < 32; ++off) {
    uint8_t addr = 0x40 + off;  // Valid PCA9685 address range: 0x40–0x5F

    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();

    // If no ACK is received, the device is missing at this address
    if (err != 0) {
      Serial.print(tf6(off));   // Print A5..A0 bit pattern
      Serial.print(" not exists (0x");
      if (addr < 16) Serial.print("0");
      Serial.print(addr, HEX);
      Serial.println(")");
      ++missing;
    }
  }

  // Final summary
  if (missing == 0) {
    Serial.println("All 32 devices present (0x40..0x5F).");
  } else {
    Serial.print("Missing devices: ");
    Serial.println(missing);
  }
}

void loop() {
  // No repeated scanning; runs only once in setup()
}
