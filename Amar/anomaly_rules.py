import cv2
import time
import math
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# ================= SETTINGS (change these to experiment) =================
# Restricted zone as fractions of the screen (x, y), 0.0 to 1.0.
# This default is a box covering the RIGHT side of the screen.
ZONE_FRACTIONS = [(0.55, 0.15), (0.95, 0.15), (0.95, 0.95), (0.55, 0.95)]

LOITER_SECONDS = 8        # use 30+ in real life; 8 is for quick testing
LOITER_RADIUS = 60        # pixels a person may move and still count as "staying"
ALERT_COOLDOWN = 10       # seconds before the same alert repeats

AFTER_HOURS_START = 22    # 10 PM
AFTER_HOURS_END = 6       # 6 AM
# ==========================================================================

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the webcam.")
    raise SystemExit

anchors = {}                  # person ID -> (x, y, time they started staying there)
last_alert = {}               # (person ID, alert type) -> last time it fired
events = deque(maxlen=5)      # recent alerts shown on screen
simulate_after_hours = False  # press 'a' to toggle for testing


def is_after_hours():
    if simulate_after_hours:
        return True
    hour = datetime.now().hour
    return hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END


def raise_alert(tid, kind, message):
    """Print and log an alert, but not more than once per cooldown."""
    now = time.time()
    key = (tid, kind)
    if now - last_alert.get(key, 0) >= ALERT_COOLDOWN:
        last_alert[key] = now
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line)
        events.append(line)


print("Running. Press 'a' to simulate after-hours, 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    zone = np.array([(int(x * w), int(y * h)) for x, y in ZONE_FRACTIONS], np.int32)

    # Draw the restricted zone as a see-through red area
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], (0, 0, 255))
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (zone[0][0] + 5, zone[0][1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    results = model.track(frame, classes=[0], conf=0.5, persist=True,
                          tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes
    now = time.time()
    any_alert = False

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            feet = (float(cx), float(y2))   # bottom-centre of the box

            # --- 3A: restricted zone ---
            in_zone = cv2.pointPolygonTest(zone, feet, False) >= 0

            # --- 3B: loitering ---
            ax, ay, start = anchors.get(tid, (cx, cy, now))
            if math.hypot(cx - ax, cy - ay) > LOITER_RADIUS:
                ax, ay, start = cx, cy, now      # moved away, restart timer
            anchors[tid] = (ax, ay, start)
            stayed = now - start
            loitering = stayed >= LOITER_SECONDS

            # --- 3C: after-hours ---
            after_hours = is_after_hours()

            flags = []
            if in_zone:
                flags.append("IN ZONE")
                raise_alert(tid, "zone", f"INTRUSION: Person {tid} entered the restricted zone")
            if loitering:
                flags.append(f"LOITERING {int(stayed)}s")
                raise_alert(tid, "loiter", f"LOITERING: Person {tid} stayed in one spot for {int(stayed)}s")
            if after_hours:
                flags.append("AFTER-HOURS")
                raise_alert(tid, "after", f"AFTER-HOURS: Person {tid} detected during quiet hours")

            danger = len(flags) > 0
            any_alert = any_alert or danger
            color = (0, 0, 255) if danger else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid}" + ("  " + ", ".join(flags) if flags else "")
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Alert banner at the top
    if any_alert:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Status and recent events
    mode = "ON (simulated)" if simulate_after_hours else ("ON" if is_after_hours() else "OFF")
    cv2.putText(frame, f"After-hours mode: {mode}", (10, h - 15 - 22 * len(events)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for i, line in enumerate(events):
        cv2.putText(frame, line, (10, h - 10 - 22 * (len(events) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SentinelEye - Step 3: Anomaly Rules", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("a"):
        simulate_after_hours = not simulate_after_hours

cap.release()
cv2.destroyAllWindows()
