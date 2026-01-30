import time
import csv
import av
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import json
import os

# ============================================================
# Camera config
# ============================================================
DEVICE_INDEX = "0"  # HD USB Camera

CAM_OPTIONS = {
    "video_size": "1920x1080",
    "framerate": "60",
    "input_format": "mjpeg",
}

OUTPUT_DIR = "exp1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECORD_PATH = os.path.join(OUTPUT_DIR, "record_raw.mp4")
PREVIEW_ANNOT_PATH = os.path.join(OUTPUT_DIR, "record_preview.mp4")

# ============================================================
# Binary Processing
# ============================================================
threshold_value = 80

# ============================================================
# Hough Parameters (legacy / TODO remove later)
# ============================================================
hough_R = 3
hough_range = 1
hough_mindist_factor = 1.2
hough_dp = 1.4
hough_param1 = 40
hough_param2 = 12

hough_params = (
    hough_R,
    hough_range,
    hough_mindist_factor,
    hough_dp,
    hough_param1,
    hough_param2,
)

# ============================================================
# ROI
# ============================================================
ROI_FILE = "roi.json"
ROI = None

if os.path.exists(ROI_FILE):
    with open(ROI_FILE, "r") as f:
        r = json.load(f)
        ROI = (r["x"], r["y"], r["w"], r["h"])
        print("[INFO] Loaded ROI:", ROI)
else:
    print("[INFO] No ROI file found, using full frame.")

# ============================================================
# Detection (still using Hough, TODO remove later)
# ============================================================
def detect_circles(frame_bgr, hough_params, roi=None):
    R, Rrange, mindist_factor, dp, param1, param2 = hough_params

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    binary = np.zeros_like(gray, dtype=np.uint8)

    if roi is None:
        _, binary = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY_INV
        )
    else:
        x0, y0, rw, rh = roi
        roi_gray = gray[y0:y0+rh, x0:x0+rw]
        _, roi_bin = cv2.threshold(
            roi_gray, threshold_value, 255, cv2.THRESH_BINARY_INV
        )
        binary[y0:y0+rh, x0:x0+rw] = roi_bin

    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=int(R * mindist_factor),
        param1=param1,
        param2=param2,
        minRadius=R - Rrange,
        maxRadius=R + Rrange
    )

    dets = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            area = float(np.pi * r * r)
            dets.append((float(x), float(y), area, int(r)))

    return dets, binary

# ============================================================
# Stage 1: Record + Live Preview
# ============================================================
def record_camera(record_path, preview_annot_path, cam_options, show_hough=True):
    fps = float(cam_options["framerate"])
    w, h = map(int, cam_options["video_size"].split("x"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_raw = cv2.VideoWriter(record_path, fourcc, fps, (w, h))
    writer_annot = None
    if preview_annot_path:
        writer_annot = cv2.VideoWriter(preview_annot_path, fourcc, fps, (w, h))

    container = av.open(
        f"{DEVICE_INDEX}:none",
        format="avfoundation",
        options=cam_options
    )
    stream = container.streams.video[0]

    print(f"[Record] {w}x{h}@{fps}")
    print("[Record] Press 'q' or ESC to stop")

    frame_count = 0
    t0 = time.time()

    # === MAGNIFIED CENTER DEBUG ===
    CROP_W = 100
    CROP_H = 100
    MAG = 6  # magnification factor

    try:
        while True:
            try:
                frame = next(container.decode(stream))
            except av.error.BlockingIOError:
                time.sleep(0.002)
                continue

            img = frame.to_ndarray(format="bgr24")
            writer_raw.write(img)

            vis = img.copy()

            if show_hough:
                dets, binary = detect_circles(img, hough_params, roi=ROI)

            # ====================================================
            # Magnified center 100x100 debug view
            # ====================================================
            H, W = img.shape[:2]
            cx, cy = W // 2, H // 2

            x0 = max(0, cx - CROP_W // 2)
            y0 = max(0, cy - CROP_H // 2)
            x1 = min(W, x0 + CROP_W)
            y1 = min(H, y0 + CROP_H)

            crop = img[y0:y1, x0:x1]
            crop_big = cv2.resize(
                crop,
                None,
                fx=MAG,
                fy=MAG,
                interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow("Center 100x100 (magnified)", crop_big)

            if show_hough:
                bin_crop = binary[y0:y1, x0:x1]
                bin_big = cv2.resize(
                    bin_crop,
                    None,
                    fx=MAG,
                    fy=MAG,
                    interpolation=cv2.INTER_NEAREST
                )
                cv2.imshow("Center 100x100 (binary)", bin_big)

            # ====================================================

            if writer_annot is not None:
                writer_annot.write(vis)

            cv2.imshow("Live", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            frame_count += 1
            if frame_count % 120 == 0:
                dt = time.time() - t0
                print(f"[Record] fps ~ {frame_count/dt:.1f}")

    finally:
        container.close()
        writer_raw.release()
        if writer_annot is not None:
            writer_annot.release()
        cv2.destroyAllWindows()

    print("[Record] Done")
    return fps

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    fps_used = record_camera(
        RECORD_PATH,
        PREVIEW_ANNOT_PATH,
        CAM_OPTIONS,
        show_hough=True
    )

    print("\n========================")
    print("ALL PROCESSING COMPLETE")
    print("========================\n")
