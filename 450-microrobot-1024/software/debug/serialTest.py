import serial

NUM_CHANNELS = 1024
PACKED_BYTES = NUM_CHANNELS // 2  # 512

# 64 pattern repeated 16 times to make 1024 length
_BASE64 = [
    11, 2, 14, 7, 0, 9, 4, 15,
    6, 1, 13, 8, 3, 12, 5, 10,
    4, 15, 9, 0, 7, 14, 2, 11,
    13, 8, 1, 6, 10, 5, 12, 3,
    1, 6, 10, 5, 12, 3, 8, 13,
    2, 11, 15, 4, 9, 0, 14, 7,
    12, 3, 5, 10, 6, 1, 13, 8,
    15, 4, 0, 9, 11, 2, 7, 14
]
X = _BASE64 * 16  # length 1024


def pack_nibbles(x):
    out = bytearray(PACKED_BYTES)
    for k in range(PACKED_BYTES):
        lo = x[2 * k] & 0x0F        # lower nibble
        hi = x[2 * k + 1] & 0x0F    # higher nibble
        out[k] = (hi << 4) | lo
    return bytes(out)


def send_over_serial(port: str, baudrate: int = 115200):
    payload = pack_nibbles(X)          # 512 bytes
    with serial.Serial(port, baudrate, timeout=1) as ser:
        ser.write(payload)
        ser.flush()


if __name__ == "__main__":
    # Windows: "COM5" 
    # macOS: "/dev/tty.usbmodemXXXX"
    send_over_serial(port="COM5", baudrate=115200)
