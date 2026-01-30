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
# Camera config (choose ONE supported mode)
# ============================================================
DEVICE_INDEX = "1"  # [0] HD USB Camera - TODO: we will use 4 cameras 

CAM_OPTIONS = {
    "video_size": "1920x1080",
    "framerate": "10",
    "input_format": "mjpeg",
}
# CAM_OPTIONS = {"video_size":"1280x720", "framerate":"120", "input_format":"mjpeg"}
# CAM_OPTIONS = {"video_size":"640x360",  "framerate":"260", "input_format":"mjpeg"}

OUTPUT_DIR = "exp1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECORD_PATH = os.path.join(OUTPUT_DIR, "record_raw.mp4")
PREVIEW_ANNOT_PATH = os.path.join(OUTPUT_DIR, "record_preview.mp4")  # Leave blank to disable this function


# ============================================================
# Binary Processing
# ============================================================

threshold_value = 80      
# Threshold for converting grayscale to binary (THRESH_BINARY_INV).
# Lower value → more pixels become white (detected as circles).
# Higher value → stricter segmentation, fewer detected regions.


# ============================================================
# Hough Transform Parameters (Circle Detection)
# ============================================================

hough_R = 3         
# Expected radius of circles in pixels.
# Increase if the circles appear larger; decrease if smaller.

hough_range = 1
# Allowed variation around the expected radius.
# Larger value → more tolerant radius detection but slower and noisier.

hough_mindist_factor = 1.2
# Minimum distance between detected circle centers = hough_R * factor.
# Increase to avoid multiple detections on a single circle.
# Decrease to allow detection of circles closer together.

hough_dp = 1.4
# Inverse accumulator resolution.
# Higher dp → faster but less accurate detection.
# Lower dp → more accurate but slower.

hough_param1 = 40
# First threshold for Canny edge detector.
# Higher value → requires stronger edges to detect circles.
# Lower value → more sensitive but produces noise.

hough_param2 = 12
# Accumulator threshold for Hough detection.
# Higher value → fewer false positives, but may miss weak circles.
# Lower value → detects more circles including noise.

# Grouped parameters for easy passing into functions
hough_params = (hough_R, hough_range, hough_mindist_factor, hough_dp, hough_param1, hough_param2)

# ============================================================
# Kalman Filter Parameters
# ============================================================

kalman_max_lost = 60
# Number of consecutive frames a track can be lost before deletion.
# Increase to maintain tracks longer during occlusion.
# Decrease to remove lost tracks quickly.

kalman_uncertainty = 20
# Initial covariance (uncertainty) of the Kalman filter.
# Higher → model assumes its initial estimate is poor.
# Lower → model trusts its initial estimate more.

kalman_measurement_noise = 0.8
# Noise level of the measurement (Hough detection).
# Increase if detections jitter a lot.
# Decrease if detections are stable.

kalman_process_noise = 0.5
# Noise added during prediction step.
# Increase → smoother trajectories but slower to respond to changes.
# Decrease → follows fast changes better but may jitter.

kalman_damping = 0.95
# Velocity damping factor to reduce excessive speed.
# Lower value (e.g., 0.90) → slow movement more aggressively.
# Higher value (e.g., 0.99) → preserve velocity more.

kalman_max_speed = 80
# Cap for maximum allowed object speed.
# Increase if objects move faster in video.
# Decrease to prevent incorrect fast jumps.

kalman_margin = 10
# Boundary margin to prevent tracks from going outside the frame.
# Increase if many detections occur near image border.

# ============================================================
# Tracking / Matching Thresholds
# ============================================================

track_dist_thre1 = 40
# Strict threshold for Hungarian matching (predicted vs. detected)
# Smaller → more selective, avoids wrong matches but may miss correct ones.
# Larger → more tolerant but increases false matches.

track_dist_thre2 = 70
# Loose matching threshold (Stage 2 manual matching).
# Acts as fallback matching when Hungarian fails.

track_dist_reb = 200
# Threshold for reassigning detections to previously lost tracks.
# Increase if reappearance happens far from predicted position.
# Decrease to reduce mistaken reattachments.

# ============================================================
# Saving Parameter Configuration
# ============================================================

config_path = os.path.join(OUTPUT_DIR, "config.json")

config = {
    "camera_options": CAM_OPTIONS,
    "hough_params": {
        "threshold_value": threshold_value,
        "R": hough_R,
        "Rrange": hough_range,
        "mindist_factor": hough_mindist_factor,
        "dp": hough_dp,
        "param1": hough_param1,
        "param2": hough_param2
    },
    "tracking_params": {
        "kalman_max_lost": kalman_max_lost,
        "track_dist_thre1": track_dist_thre1,
        "track_dist_thre2": track_dist_thre2,
        "track_dist_reb": track_dist_reb
    }
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"[INFO] Config saved to: {config_path}")

# ============================================================
# ROI Implementation
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
# Kalman filter & MultiTracker (your original, unchanged)
# TODO: I don't think this is neccessary 
# ============================================================
#  I don't think this is neccesarry get rid of it (TODO)
class KalmanTrack:
    def __init__(self, track_id, init_xy):
        self.id = track_id
        self.lost = 0

        self.state = np.array([init_xy[0], init_xy[1], 0, 0], float)
        self.P = np.eye(4) * kalman_uncertainty

        self.F = np.array([[1,0,1,0],
                           [0,1,0,1],
                           [0,0,1,0],
                           [0,0,0,1]], float)

        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        self.R = np.eye(2) * kalman_measurement_noise
        self.Q = np.eye(4) * kalman_process_noise

    def predict(self, w, h):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        x,y,vx,vy = self.state
        vx *= kalman_damping
        vy *= kalman_damping

        speed = np.sqrt(vx*vx + vy*vy)
        if speed > kalman_max_speed:
            vx = vx / speed * kalman_max_speed
            vy = vy / speed * kalman_max_speed

        margin = kalman_margin
        if x < margin:
            x = margin; vx = 0
        elif x > w-margin:
            x = w-margin; vx = 0
        if y < margin:
            y = margin; vy = 0
        elif y > h-margin:
            y = h-margin; vy = 0

        self.state = np.array([x,y,vx,vy])
        return self.state[:2]

    def update(self, meas_xy):
        z = np.array(meas_xy)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.lost = 0

    def mark_lost(self):
        self.lost += 1


class MultiTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0
        self.frame_w = 0
        self.frame_h = 0

    def dist(self,p,q): return np.linalg.norm(np.array(p)-np.array(q))

    def update(self, detections):
        preds = [trk.predict(self.frame_w, self.frame_h) for trk in self.tracks]

        matched_tracks = set()
        used_dets = set()

        if len(preds)>0 and len(detections)>0:
            cost = np.zeros((len(preds), len(detections)))
            for i,p in enumerate(preds):
                for j,d in enumerate(detections):
                    cost[i,j] = self.dist(p, d[:2])

            row, col = linear_sum_assignment(cost)
            for r,c in zip(row,col):
                if cost[r,c] < track_dist_thre1:
                    self.tracks[r].update(detections[c][:2])
                    matched_tracks.add(r)
                    used_dets.add(c)

        for i,trk in enumerate(self.tracks):
            if i in matched_tracks: 
                continue
            best_j, best_d = -1, 99999
            for j,d in enumerate(detections):
                if j in used_dets: 
                    continue
                dxy = self.dist(trk.state[:2], d[:2])
                if dxy < best_d and dxy < track_dist_thre2:
                    best_d = dxy; best_j = j
            if best_j!=-1:
                self.tracks[i].update(detections[best_j][:2])
                matched_tracks.add(i)
                used_dets.add(best_j)

        for i,trk in enumerate(self.tracks):
            if i not in matched_tracks:
                trk.mark_lost()

        lost_candidates = [i for i,t in enumerate(self.tracks) if t.lost>=1]
        unused_dets = [j for j in range(len(detections)) if j not in used_dets]
        for j in unused_dets:
            det_xy = detections[j][:2]
            best_i, best_d = None, 99999
            for i in lost_candidates:
                pred = self.tracks[i].state[:2]
                d = self.dist(pred, det_xy)
                if d < best_d:
                    best_d = d; best_i = i
            if best_i is not None and best_d < track_dist_reb:
                self.tracks[best_i].update(det_xy)
                self.tracks[best_i].lost = 0
                lost_candidates.remove(best_i)

        self.tracks = [t for t in self.tracks if t.lost < kalman_max_lost]

        out=[]
        for t in self.tracks:
            x,y = t.state[:2]
            out.append((t.id, int(x), int(y)))
        return out


# TODO: don't use Hough detection 
# ============================================================
# Hough detection on a frame (reused by both live preview & offline tracking)
# ============================================================

def detect_circles(frame_bgr, hough_params, roi=None):
    R, Rrange, mindist_factor, dp, param1, param2 = hough_params


    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # TODO 
    #do gussian filter first
    #have lookup table
    #gamma transformation through cv.LUT() 


    binary = np.zeros_like(gray, dtype=np.uint8)

    if roi is None:
        _, binary = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY_INV
        )
    else:
        x0, y0, rw, rh = roi

        h, w = gray.shape
        x0 = max(0, min(x0, w-1))
        y0 = max(0, min(y0, h-1))
        rw = max(1, min(rw, w - x0))
        rh = max(1, min(rh, h - y0))

        roi_gray = gray[y0:y0+rh, x0:x0+rw]
        _, roi_bin = cv2.threshold(
            roi_gray, threshold_value, 255, cv2.THRESH_BINARY_INV
        )

        binary[y0:y0+rh, x0:x0+rw] = roi_bin

    # TODO don't use circle
    # TODO: Connected components: In a binary image, a connected component is a group of adjacent foreground pixels
    # (4- or 8-connectivity) forming one object.
    # Each component can be treated as one robot if its pixel count exceeds a threshold.
    # Object position is estimated using the component centroid (mean of pixel coordinates).

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
# Stage 1: Record from camera + live preview with Hough overlay
#   - Press 'q' or ESC to stop recording
# ============================================================
def record_camera(record_path, preview_annot_path, cam_options, show_hough=True):
    fps = float(cam_options["framerate"])
    w, h = map(int, cam_options["video_size"].split("x"))

    # Writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_raw = cv2.VideoWriter(record_path, fourcc, fps, (w, h))
    writer_annot = None
    if preview_annot_path:
        writer_annot = cv2.VideoWriter(preview_annot_path, fourcc, fps, (w, h))

    container = av.open('video=HD USB Camera', format='dshow', options=cam_options)
    stream = container.streams.video[0]

    print(f"[Record] Start recording -> {record_path}  ({w}x{h}@{fps})")
    print("[Record] Press 'q' or ESC to stop.")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            try:
                frame = next(container.decode(stream))
            except av.error.BlockingIOError:
                time.sleep(0.002)
                continue

            img = frame.to_ndarray(format="bgr24")
            writer_raw.write(img)

            vis = img
            if show_hough:
                dets, binary = detect_circles(img, hough_params, roi = ROI)
                vis = img.copy()
                #    cv2.circle(vis, (int(x), int(y)), r, (0,255,0), 2)
                #    cv2.circle(vis, (int(x), int(y)), 2, (0,0,255), -1)
                #cv2.putText(vis, f"dets={len(dets)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            if writer_annot is not None:
                writer_annot.write(vis)

            cv2.imshow("Live (Hough overlay)" if show_hough else "Live", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            frame_count += 1
            if frame_count % 120 == 0:
                dt = time.time() - t0
                print(f"[Record] frames={frame_count}, approx_fps={frame_count/dt:.1f}")

    finally:
        try:
            container.close()
        except:
            pass
        writer_raw.release()
        if writer_annot is not None:
            writer_annot.release()
        cv2.destroyAllWindows()

    print(f"[Record] Done. Saved: {record_path}")
    if preview_annot_path:
        print(f"[Record] Preview annotated saved: {preview_annot_path}")
    return fps


# ============================================================
# Stage 2: Offline tracking on saved video (your original run_tracking, adapted)
#   - outputs: output.mp4 + tracks.csv
# ============================================================
def run_tracking(video_path, hough_params, fps_override=None):
    R, Rrange, mindist_factor, dp, param1, param2 = hough_params

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_override is not None:
        fps = float(fps_override)
    if fps <= 0:
        fps = 30.0

    ret, first = cap.read()
    if not ret:
        print("Cannot open video:", video_path)
        return

    h, w = first.shape[:2]
    output_video_path = os.path.join(OUTPUT_DIR, "output.mp4")
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    tracker = MultiTracker()
    tracker.frame_w = w
    tracker.frame_h = h

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_index = 0
    out_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets, _ = detect_circles(frame, hough_params, roi=ROI)
        # dets: (x,y,area,r)
        det_xy_area = [(d[0], d[1], d[2]) for d in dets]

        if frame_index == 0:
            for (x, y, _) in det_xy_area:
                tracker.tracks.append(KalmanTrack(tracker.next_id, (x, y)))
                tracker.next_id += 1
                out_tracks.append((tracker.next_id-1, 0, int(x), int(y)))
            writer.write(frame)
            frame_index += 1
            continue

        tracked = tracker.update(det_xy_area)

        for tid, x, y in tracked:
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame, str(tid), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            out_tracks.append((tid, frame_index, x, y))
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    print("[Offline] output.mp4 generated.")

    write_csv(out_tracks, fps)


def write_csv(out_tracks, fps):
    track_dict = defaultdict(dict)
    for tid, f, x, y in out_tracks:
        track_dict[tid][f] = (x, y)

    new_out = []
    for tid, frames in track_dict.items():
        f_sorted = sorted(frames.keys())
        for i in range(len(f_sorted) - 1):
            f1, f2 = f_sorted[i], f_sorted[i+1]
            x1, y1 = frames[f1]
            x2, y2 = frames[f2]
            new_out.append((tid, f1, x1, y1))

            gap = f2 - f1
            if gap > 1:
                for k in range(1, gap):
                    t = k / gap
                    xi = int(x1 + (x2 - x1) * t)
                    yi = int(y1 + (y2 - y1) * t)
                    new_out.append((tid, f1 + k, xi, yi))

        last = f_sorted[-1]
        x_last, y_last = frames[last]
        new_out.append((tid, last, x_last, y_last))

    new_out.sort(key=lambda x: x[1])

    ids = sorted(list({r[0] for r in new_out}))
    time_dict = defaultdict(dict)
    for tid, f, x, y in new_out:
        time_dict[f / fps][tid] = (x, y)

    csv_path = os.path.join(OUTPUT_DIR, "tracks.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        header = ["time"]
        for tid in ids:
            header += [f"id{tid}_x", f"id{tid}_y"]
        wr.writerow(header)

        for t in sorted(time_dict.keys()):
            row = [t]
            for tid in ids:
                if tid in time_dict[t]:
                    row += list(time_dict[t][tid])
                else:
                    row += ["", ""]
            wr.writerow(row)

    print(f"[Offline] CSV saved to: {csv_path}")



# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":

    # 1) Record from camera with live preview + (optional) hough overlay
    fps_used = record_camera(RECORD_PATH, PREVIEW_ANNOT_PATH, CAM_OPTIONS, show_hough=True)

    # 2) Offline tracking -> output.mp4 + tracks.csv
    run_tracking(RECORD_PATH, hough_params, fps_override=fps_used)

    print("\n========================")
    print("ALL PROCESSING COMPLETE")
    print("========================\n")
