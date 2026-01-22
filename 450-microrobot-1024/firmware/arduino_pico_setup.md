## Use Raspberry Pi Pico with Arduino (.ino)

1. Install **Arduino IDE**.
2. Add the RP2040 board package:
   - Arduino IDE → **Preferences** → **Additional Boards Manager URLs**  
     Add:
     - `https://github.com/earlephilhower/arduino-pico/releases/download/global/package_rp2040_index.json`
3. Install the core:
   - **Tools → Board → Boards Manager** → search **“Raspberry Pi Pico/RP2040”** → Install.
4. Select your board:
   - **Tools → Board** → **Raspberry Pi Pico** (or **Pico W**).
5. Enter BOOTSEL mode (first-time or if upload fails):
   - Hold **BOOTSEL** on the Pico, plug in USB → a drive **RPI-RP2** appears.
6. Upload your sketch:
   - **Tools → Port** (select Pico) → click **Upload**.
7. Verify serial output (optional):
   - `Serial.begin(115200);` → **Tools → Serial Monitor** → set **115200**.


https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html

<img width="1398" height="892" alt="Screenshot 2026-01-21 at 9 32 43 AM" src="https://github.com/user-attachments/assets/b88a6fef-82c1-48fa-ba01-e40b726cb964" />
