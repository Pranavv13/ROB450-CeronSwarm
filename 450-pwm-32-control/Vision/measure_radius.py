import time
import av
import cv2
import numpy as np

# ===============================
# Camera config (must match supported modes)
# ===============================
DEVICE_INDEX = "0"  # [0] HD USB Camera
CAM_OPTIONS = {"video_size": "1920x1080", "framerate": "60", "input_format": "mjpeg"}

# ===============================
# Mouse callback function
# ===============================
click_points = []  # store clicked points for distance measurement

def show_xy(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = param.copy()
        cv2.putText(img_copy, f"({x}, {y})", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # show last click points if any
        for p in click_points:
            cv2.circle(img_copy, p, 4, (0, 255, 0), -1)

        # if two points clicked, show distance
        if len(click_points) == 2:
            (x1, y1), (x2, y2) = click_points
            dist = ((x1-x2)**2 + (y1-y2)**2) ** 0.5
            cv2.line(img_copy, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_copy, f"dist(px)={dist:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Snapshot - Move mouse to see coordinates", img_copy)

    elif event == cv2.EVENT_LBUTTONDOWN:
        print(f"[CLICK] x={x}, y={y}")
        click_points.append((x, y))

        # keep only last 2 points
        if len(click_points) > 2:
            click_points = click_points[-2:]

        if len(click_points) == 2:
            (x1, y1), (x2, y2) = click_points
            dist = ((x1-x2)**2 + (y1-y2)**2) ** 0.5
            print(f"[DIAMETER] pixel distance = {dist:.2f} px (between last two clicks)")

# ===============================
# Capture one frame from camera
# ===============================
def capture_one_frame():
    container = av.open(f"{DEVICE_INDEX}:none", format="avfoundation", options=CAM_OPTIONS)
    stream = container.streams.video[0]

    frame_bgr = None
    try:
        # warm up a few frames to stabilize exposure
        warm = 10
        got = 0
        while got < warm:
            try:
                fr = next(container.decode(stream))
            except av.error.BlockingIOError:
                time.sleep(0.002)
                continue
            got += 1

        # take snapshot
        while True:
            try:
                fr = next(container.decode(stream))
                frame_bgr = fr.to_ndarray(format="bgr24")
                break
            except av.error.BlockingIOError:
                time.sleep(0.002)
                continue
    finally:
        container.close()

    return frame_bgr

# ===============================
# Main
# ===============================
frame = capture_one_frame()
if frame is None:
    print("Failed to capture frame.")
    raise SystemExit

cv2.namedWindow("Snapshot - Move mouse to see coordinates", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Snapshot - Move mouse to see coordinates", show_xy, frame)

cv2.imshow("Snapshot - Move mouse to see coordinates", frame)

print("\nMove the mouse on the window to see coordinates.")
print("Click to print coordinate values in console.")
print("Click two points to print pixel distance (diameter).")
print("Press any key to exit.\n")

cv2.waitKey(0)
cv2.destroyAllWindows()
