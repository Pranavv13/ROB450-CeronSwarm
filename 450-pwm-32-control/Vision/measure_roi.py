import os
import json
import time
import av
import cv2

# ===============================
# Camera config (must be supported)
# ===============================
DEVICE_INDEX = "0"   # [0] HD USB Camera
CAM_OPTIONS = {
    "video_size": "1920x1080",
    "framerate": "60",
    "input_format": "mjpeg",
}

# ===============================
# Output file
# ===============================
ROI_FILE = "roi.json"


# ===============================
# Capture one frame from camera
# ===============================
def capture_one_frame():
    container = av.open(f"{DEVICE_INDEX}:none",
                        format="avfoundation",
                        options=CAM_OPTIONS)
    stream = container.streams.video[0]

    frame_bgr = None
    try:
        # warm-up frames
        for _ in range(10):
            try:
                next(container.decode(stream))
            except av.error.BlockingIOError:
                time.sleep(0.002)

        # grab one frame
        while True:
            try:
                fr = next(container.decode(stream))
                frame_bgr = fr.to_ndarray(format="bgr24")
                break
            except av.error.BlockingIOError:
                time.sleep(0.002)
    finally:
        container.close()

    return frame_bgr


# ===============================
# Select ROI interactively
# ===============================
def select_roi(frame):
    print("\n[ROI] Drag to select ROI, press ENTER to confirm, 'c' to cancel.")
    roi = cv2.selectROI(
        "Select ROI",
        frame,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    frame = capture_one_frame()
    if frame is None:
        print("Failed to capture frame.")
        raise SystemExit

    roi = select_roi(frame)
    if roi is None:
        print("[ROI] No ROI selected.")
        raise SystemExit

    with open(ROI_FILE, "w") as f:
        json.dump(roi, f, indent=2)

    print("\n[ROI] Saved to", ROI_FILE)
    print("[ROI] Content:", roi)
