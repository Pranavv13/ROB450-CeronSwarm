#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

/*
  SIMPLE DEBUG SWEEP — KEY-ADVANCE MODE
  ----------------------------------------
  Goal:
    - Control 32 PCA9685 boards (I2C addresses 0x40–0x5F).
    - Each board has 16 PWM channels -> 8 magnet pairs: (0,1), (2,3), ..., (14,15).
    - Activate ONLY the LEFT pin of one pair (right pin OFF).
    - Move to the NEXT pair ONLY WHEN a key is pressed on the Serial Monitor.
    - After finishing 8 pairs on this board, move to the next board, loop forever.

  Safety:
    - Before switching to a new pair, turn ALL channels OFF.
    - Left pin ON, Right pin always OFF (never ON at the same time).

  How to use:
    - Open Serial Monitor at 115200 baud.
    - Press any key (and hit Enter) to advance to the next pair/board.
*/

//////////////////// USER SETTINGS ////////////////////
#define NUM_BOARDS       32          // Number of PCA9685 devices: addresses 0x40..0x5F
#define BASE_ADDR        0x40        // PCA9685 base I2C address
#define PWM_FREQ_HZ      1000        // PCA9685 PWM frequency (Hz)
#define I2C_SPEED_HZ     400000      // I2C bus speed (Hz); 400kHz fast-mode

float DUTY_PERCENT = 85.0f;         // Left pin duty [%] (right pin is always 0%)
#define ENABLE_SERIAL_LOG 1           // Set to 0 to disable Serial logs
///////////////////////////////////////////////////////

// Array of driver pointers (one per board)
Adafruit_PWMServoDriver* boards[NUM_BOARDS] = { nullptr };

// Convert duty percentage (0..100) to 12-bit PWM ticks (0..4095)
static inline uint16_t dutyToTicks(float percent) {
  if (percent < 0.0f)   percent = 0.0f;
  if (percent > 100.0f) percent = 100.0f;
  return (uint16_t) lroundf(4095.0f * (percent / 100.0f));
}

// Turn OFF all 16 channels on a given board
static inline void allOff(Adafruit_PWMServoDriver* board) {
  for (int ch = 0; ch < 16; ++ch) {
    board->setPWM(ch, 0, 0);              // OFF
  }
}

// Activate one pair with LEFT ON (at ticks) and RIGHT OFF
static inline void activateLeftOnly(Adafruit_PWMServoDriver* board, int pairIndex, uint16_t leftTicks) {
  const int leftPin  = 2 * pairIndex;     // even channel (left)
  const int rightPin = leftPin + 1;       // odd channel  (right)

  board->setPWM(rightPin, 0, 0);          // RIGHT must be OFF
  board->setPWM(leftPin,  0, leftTicks);  // LEFT at requested duty
}

// Wait here until the user presses any key (and Enter) in the Serial Monitor
static inline void waitForKey(const char* prompt) {
#if ENABLE_SERIAL_LOG
  Serial.println(prompt);
  Serial.println("  -> Press any key (then Enter) to advance...");
#endif
  // Block until at least one byte arrives
  while (Serial.available() == 0) {
    // idle loop; you could add a tiny delay to reduce CPU usage
    delay(1);
  }
  // Drain the input buffer (consume what user typed)
  while (Serial.available() > 0) {
    (void)Serial.read();
  }
}

void setup() {
#if ENABLE_SERIAL_LOG
  Serial.begin(115200);                    // Serial for logs & key-advance
  delay(1000);                             // Give time for Serial to attach
  Serial.println("Starting PCA9685 debug sweep (key-advance mode)...");
#endif

  Wire.begin();                            // Start I2C as master
  Wire.setClock(I2C_SPEED_HZ);             // Set I2C bus speed

  // Initialize all PCA9685 boards (0x40..0x5F)
  for (int i = 0; i < NUM_BOARDS; ++i) {
    const uint8_t addr = (uint8_t)(BASE_ADDR + i);
    boards[i] = new Adafruit_PWMServoDriver(addr);

    boards[i]->begin();                    // Initialize PCA9685 at this address
    boards[i]->setPWMFreq(PWM_FREQ_HZ);    // Set PWM frequency
    allOff(boards[i]);                     // Ensure all outputs are OFF

#if ENABLE_SERIAL_LOG
    Serial.print("Initialized PCA9685 at 0x");
    if (addr < 0x10) Serial.print('0');
    Serial.println(addr, HEX);
#endif

    delay(2);                              // Small settle time per device
  }

#if ENABLE_SERIAL_LOG
  Serial.println("All boards ready.");
#endif
}

void loop() {
  const uint16_t leftTicks = dutyToTicks(DUTY_PERCENT);

  // Sweep from board 0x40 (000000) to 0x5F (011111)
  for (int b = 0; b < NUM_BOARDS; ++b) {
    Adafruit_PWMServoDriver* board = boards[b];
    allOff(board);                         // Safety: everything OFF before starting

#if ENABLE_SERIAL_LOG
    uint8_t addr = (uint8_t)(BASE_ADDR + b);
    Serial.print("\n=== Board index ");
    Serial.print(b);
    Serial.print(" (I2C 0x");
    if (addr < 0x10) Serial.print('0');
    Serial.print(addr, HEX);
    Serial.println(") ===");
#endif

    // Each board controls 8 magnet pairs: (0,1), (2,3), ..., (14,15)
    for (int pair = 0; pair < 8; ++pair) {
      // Turn everything OFF, then activate the target pair's LEFT pin
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
#endif

      // Wait for user key to proceed to next pair
      waitForKey("Holding current pair.");
    }

    // After finishing this board, ensure all channels are OFF
    allOff(board);

#if ENABLE_SERIAL_LOG
    Serial.println("Board done. All channels OFF.");
    waitForKey("Ready to move to the next board.");
#endif
  }

  // After going through all 32 boards, start again
#if ENABLE_SERIAL_LOG
  Serial.println("\nSweep complete. Restarting from the first board.");
  waitForKey("Press a key to restart the entire sweep.");
#endif
}
