![Group 110](https://github.com/user-attachments/assets/3c313865-117c-45a5-b102-63f436cded69)

# Firmware
- firmware/pico1
- firmware/pico2
- firmware/debug

## pico1
1. From UART (pico2) recieve 512 electromagnet signal
2. Action1: send action to i2c0, i2c1

## pico2
1. From USB (PC) recieve 1024 electromagnet signal
2. Send 512 electromagnet signal to pico1
3. Action: send action to i2c0, i2c1

## debug 
Each `.ino` file is designed for a specific debugging purpose:

- **PCAAddressChecker**  
  Verifies I²C communication and detects connected PCA9685 devices on the bus.  
  Used to confirm correct wiring and address configuration.

- **debug_i2c_bus**  
  Performs one-by-one testing of individual electromagnets on a single I²C bus.  
  Used to validate channel mapping and basic actuation functionality.

- **goToAddress**  
  Connects to a specific PCA9685 device and activates only the selected electromagnet.  
  Used to debug targeting, addressing, and selective channel control.


![Frame 1](https://github.com/user-attachments/assets/d9fcbcf4-063a-46f3-a9ff-f5d9b0826f95)

# Dual-Pico Electromagnet Control Firmware

This project controls a 1,024-channel electromagnet array using two Raspberry Pi Pico boards (pico2 and pico1) and multiple PCA9685 PWM driver boards. A PC streams packed command frames over USB to pico2. The frame is split between the two Picos, unpacked into per-channel commands, and applied to the PCA9685 devices over two I2C buses per Pico.

---

## System Overview

Each control frame represents commands for 1,024 electromagnets.

- The PC sends **512 bytes per frame** over USB serial.
- Each byte encodes **two 4-bit channel commands** (two nibbles).
- pico2 receives the full frame, forwards the first half to pico1 over UART, and processes the second half locally.
- Each Pico controls **512 electromagnets** using two I2C buses.
- Each I2C bus drives **32 PCA9685 boards**, with **8 electromagnets per PCA**.

Total mapping:

- 512 bytes → 1,024 channels  
- 256 bytes per Pico → 512 channels per Pico  
- 256 channels per I2C bus  

---

## Data Format

Each byte from the PC contains two channel commands:

- Low nibble (bits 3–0): first channel  
- High nibble (bits 7–4): second channel  

Each channel command `Xi ∈ {0…15}` encodes polarity and magnitude:

- `0…7`  → negative polarity, magnitude `0…7`  
- `8…15` → positive polarity, magnitude `Xi − 8`  

Magnitude `m ∈ {0…7}` is linearly mapped to a 12-bit PCA9685 PWM value in the range `0…4095`.

---

## pico2 Role (USB Receiver + Dispatcher + Local Driver)

`pico2` is the main entry point for each frame.

For every loop iteration:

1. **Read from PC (USB):**  
   Blocks until exactly 512 bytes are received from `Serial`.

2. **Forward to pico1 (UART):**  
   Sends the first 256 bytes of the frame to pico1 using `Serial1`.

3. **Local processing:**  
   Uses the remaining 256 bytes, unpacks them into 512 channel commands, and applies them to PCA9685 devices over `Wire` and `Wire1`.

`pico2` initializes:
- `Serial` for USB communication  
- `Serial1` for the pico2 → pico1 UART link  
- `Wire` and `Wire1` for I2C output  

---

## pico1 Role (UART Receiver + Local Driver)

`pico1` only receives data from pico2 and drives its own 512 channels.

For every loop iteration:

1. **Read from pico2 (UART):**  
   Blocks until 256 bytes are received from `Serial1`.

2. **Unpack:**  
   Converts the 256 packed bytes into 512 channel commands.

3. **Apply outputs:**  
   Writes PWM values to PCA9685 devices over `Wire` and `Wire1`.

This mirrors pico2’s local processing path, but uses UART instead of USB as the data source.

---

## Shared Command Layer (`command.h` / `command.cpp`)

Both Picos use the same command layer.

### Blocking Serial Read

`readExactBytes(Stream& s, uint8_t* dst, int n)`

Reads exactly `n` bytes from any Arduino `Stream` (USB or UART). The function blocks until the full frame is received.

---

### Unpacking Packed Frames

`buildX(const uint8_t* packed_in, uint8_t* X_out, int packed_bytes)`

Converts packed bytes into 4-bit channel commands:

- Each input byte produces two output values.
- Output length is `2 × packed_bytes`.

---

### PCA9685 Output Mapping

`actionX(TwoWire& bus0, TwoWire& bus1, const uint8_t* X512)`

- Splits 512 channel commands into two groups of 256.
- Sends the first 256 commands over `bus0` and the second 256 over `bus1`.
- Maps channels to PCA9685 devices:
  - Base address: `0x40`
  - 32 devices per bus (`0x40 … 0x5F`)
  - 8 electromagnets per PCA (2 PWM channels per magnet)

For each channel:
- Selects polarity by choosing which of the two PCA channels is active.
- Writes a 12-bit PWM “OFF” count proportional to the commanded magnitude.


