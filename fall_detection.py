import cv2
import math
import time
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# ================= SETTINGS =================
FALL_ANGLE = 55          # torso tilt from vertical (degrees) that counts as "lying"
FALL_HOLD_SECONDS = 1.0  # must stay lying this long before an alert
KP_CONF = 0.5            # minimum confidence to trust a body point
ALERT_COOLDOWN = 10      # seconds before the same person's alert repeats
# ============================================

model = YOLO("yolov8n-pose.pt")   # downloads automatically the first time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the webcam.")
    raise SystemExit

lying_since = {}      # person ID -> time they started lying
last_alert = {}       # person ID -> last alert time
events = deque(maxlen=4)

# COCO keypoint numbers
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def midpoint(kps, confs, a, b):
    """Average of two body points, or None if either is unreliable."""
    if confs is not None and (confs[a] < KP_CONF or confs[b] < KP_CONF):
        return None
    return ((kps[a][0] + kps[b][0]) / 2, (kps[a][1] + kps[b][1]) / 2)


def torso_angle(kps, confs):
    """Angle of shoulder-to-hip line from vertical. 0 = standing, 90 = lying flat."""
    sh = midpoint(kps, confs, L_SH, R_SH)
    hip = midpoint(kps, confs, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None
    dx = abs(sh[0] - hip[0])
    dy = abs(sh[1] - hip[1])
    return math.degrees(math.atan2(dx, dy))


print("Running fall detection. Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(frame, conf=0.5, persist=True,
                          tracker="bytetrack.yaml", verbose=False)
    r = results[0]
    annotated = r.plot(boxes=False)    # draws the skeleton lines
    now = time.time()
    seen_now = set()
    fall_now = False

    if r.boxes.id is not None and r.keypoints is not None:
        kp_xy = r.keypoints.xy.cpu().numpy()
        kp_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

        for i, (box, track_id) in enumerate(zip(r.boxes.xyxy, r.boxes.id)):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            seen_now.add(tid)

            conf_i = kp_conf[i] if kp_conf is not None else None
            angle = torso_angle(kp_xy[i], conf_i)

            if angle is not None:
                lying = angle > FALL_ANGLE
                info = f"tilt {int(angle)} deg"
            else:
                lying = (x2 - x1) > 1.2 * (y2 - y1)   # wider than tall
                info = "box shape"

            if lying:
                lying_since.setdefault(tid, now)
            else:
                lying_since.pop(tid, None)

            held = now - lying_since[tid] if tid in lying_since else 0
            is_fall = held >= FALL_HOLD_SECONDS

            if is_fall:
                fall_now = True
                if now - last_alert.get(tid, 0) >= ALERT_COOLDOWN:
                    last_alert[tid] = now
                    line = f"[{datetime.now():%H:%M:%S}] FALL: Person {tid} is down"
                    print(line)
                    events.append(line)

            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid} {'FALL!' if is_fall else 'OK'} ({info})"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Forget people who left the frame
    for tid in list(lying_since):
        if tid not in seen_now:
            lying_since.pop(tid, None)

    h, w = annotated.shape[:2]
    if fall_now:
        cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(annotated, "FALL DETECTED", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    for i, line in enumerate(events):
        cv2.putText(annotated, line, (10, h - 10 - 22 * (len(events) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SentinelEye - Step 3D: Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()