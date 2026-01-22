#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

/*
  SINGLE-BOARD MODE — ADDRESS-SELECT + KEY-ADVANCE
  ------------------------------------------------
  Usage:
    1) Open Serial Monitor @ 115200, line ending = "Newline".
    2) Type an address like: 0x40 ... 0x5F  (hex)  or 64 ... 95 (decimal)
       and press Enter.
    3) If found, the sketch controls ONLY that PCA9685.
       - Shows pairs (0,1), (2,3), ..., (14,15)
       - Left ON at DUTY_PERCENT, Right OFF
       - Press any key + Enter to advance pairs
       - Press 'x' + Enter to exit back to address prompt

  Safety:
    - Before switching pairs: all channels OFF
    - Right pin ALWAYS OFF
*/

//////////////////// USER SETTINGS ////////////////////
#define NUM_BOARDS       32
#define BASE_ADDR        0x40          // valid range: 0x40 .. 0x5F
#define PWM_FREQ_HZ      1000
#define I2C_SPEED_HZ     400000

float DUTY_PERCENT = 100.0f;            // Left duty [%]
#define ENABLE_SERIAL_LOG 1
///////////////////////////////////////////////////////

Adafruit_PWMServoDriver* boards[NUM_BOARDS] = { nullptr };

// ----------------- Utilities -----------------
static inline uint16_t dutyToTicks(float percent) {
  if (percent < 0.0f)   percent = 0.0f;
  if (percent > 100.0f) percent = 100.0f;
  return (uint16_t) lroundf(4095.0f * (percent / 100.0f));
}

static inline void allOff(Adafruit_PWMServoDriver* board) {
  for (int ch = 0; ch < 16; ++ch) board->setPWM(ch, 0, 0);
}

static inline void activateLeftOnly(Adafruit_PWMServoDriver* board, int pairIndex, uint16_t leftTicks) {
  const int leftPin  = 2 * pairIndex;
  const int rightPin = leftPin + 1;
  board->setPWM(rightPin, 0, 0);         // RIGHT OFF
  board->setPWM(leftPin,  0, leftTicks); // LEFT duty
}

// Return 0 if no char (non-blocking), otherwise the first char read (and flushes rest)
static char readKeyNonBlocking() {
  if (Serial.available() == 0) return 0;
  char c = (char)Serial.read();
  while (Serial.available() > 0) (void)Serial.read();
  return c;
}

// Read a full line (ending with \n). Returns true if a line was captured in 'out'.
static bool readLine(String &out) {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      return true;            // line complete
    }
    out += c;
  }
  return false;
}

// Parse user input as hex ("0x47" or "47") or decimal ("71")
static bool parseAddr(const String &s, uint8_t &addr) {
  // Trim spaces
  String t = s;
  t.trim();
  if (t.length() == 0) return false;

  long val = -1;
  if (t.startsWith("0x") || t.startsWith("0X")) {
    // hex
    val = strtol(t.c_str(), nullptr, 16);
  } else {
    // try hex if contains A-F, else decimal
    bool hasHexDigit = false;
    for (size_t i = 0; i < t.length(); ++i) {
      char c = t[i];
      if ((c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) { hasHexDigit = true; break; }
    }
    val = strtol(t.c_str(), nullptr, hasHexDigit ? 16 : 10);
  }
  if (val < 0 || val > 127) return false;
  addr = (uint8_t)val;
  return true;
}

// I2C presence probe (ACK check). Returns true if a device ACKs at addr.
static bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

// --------------- Setup & Init ----------------
void setup() {
#if ENABLE_SERIAL_LOG
  Serial.begin(115200);
  delay(600);
  Serial.println("PCA9685 single-board debug (address-select + key-advance)");
#endif

  Wire.begin();
  Wire.setClock(I2C_SPEED_HZ);

  // Pre-instantiate all drivers (addresses 0x40..0x5F)
  for (int i = 0; i < NUM_BOARDS; ++i) {
    uint8_t addr = (uint8_t)(BASE_ADDR + i);
    boards[i] = new Adafruit_PWMServoDriver(addr);
    boards[i]->begin();
    boards[i]->setPWMFreq(PWM_FREQ_HZ);
    allOff(boards[i]);
#if ENABLE_SERIAL_LOG
    Serial.print("Ready: PCA9685 @ 0x");
    if (addr < 0x10) Serial.print('0');
    Serial.println(addr, HEX);
#endif
    delay(2);
  }

#if ENABLE_SERIAL_LOG
  Serial.println();
  Serial.println("Type I2C address (e.g., 0x47 or 71) and press Enter:");
#endif
}

// --------------- Main Loop -------------------
void loop() {
  static String line;
  // 1) Wait for a line (address)
  if (!readLine(line)) {
    delay(2);
    return;
  }

  // Got a line: parse address
  uint8_t addr = 0;
  if (!parseAddr(line, addr)) {
#if ENABLE_SERIAL_LOG
    Serial.println("Invalid input. Enter address like 0x40..0x5F or 64..95.");
#endif
    line = "";
    return;
  }
  line = "";

  // 2) Validate range
  if (addr < BASE_ADDR || addr >= (BASE_ADDR + NUM_BOARDS)) {
#if ENABLE_SERIAL_LOG
    Serial.print("Address out of range: 0x");
    if (addr < 0x10) Serial.print('0');
    Serial.print(addr, HEX);
    Serial.println("  (valid: 0x40..0x5F)");
#endif
    return;
  }

  // 3) Probe device presence
  if (!i2cPresent(addr)) {
#if ENABLE_SERIAL_LOG
    Serial.print("No ACK from device at 0x");
    if (addr < 0x10) Serial.print('0');
    Serial.print(addr, HEX);
    Serial.println(". Check wiring/power.");
#endif
    return;
  }

  // 4) Control ONLY this board
  Adafruit_PWMServoDriver* board = boards[addr - BASE_ADDR];
  const uint16_t leftTicks = dutyToTicks(DUTY_PERCENT);

#if ENABLE_SERIAL_LOG
  Serial.print("\n=== Controlling PCA9685 @ 0x");
  if (addr < 0x10) Serial.print('0');
  Serial.print(addr, HEX);
  Serial.println(" ===");
  Serial.println("Press any key + Enter to advance pairs; 'x' + Enter to exit.");
#endif

  // Ensure clean start
  allOff(board);

  for (int pair = 0; pair < 8; ++pair) {
    allOff(board);
    activateLeftOnly(board, pair, leftTicks);

#if ENABLE_SERIAL_LOG
    Serial.print("Pair ");
    Serial.print(pair);
    Serial.print(": LEFT ON (ch ");
    Serial.print(2 * pair);
    Serial.print("), RIGHT OFF (ch ");
    Serial.print(2 * pair + 1);
    Serial.print("), duty = ");
    Serial.print(DUTY_PERCENT);
    Serial.println("%");
    Serial.println("  -> Press any key + Enter to continue, or 'x' + Enter to exit.");
#endif

    // Wait here for user input
    char c = 0;
    while ((c = readKeyNonBlocking()) == 0) {
      delay(1);
    }
    if (c == 'x' || c == 'X') {
#if ENABLE_SERIAL_LOG
      Serial.println("Exiting to address prompt...");
#endif
      break;
    }
  }

  // Clean up
  allOff(board);

#if ENABLE_SERIAL_LOG
  Serial.println("Done with this board. Enter another address (e.g., 0x40..0x5F):");
#endif
}
