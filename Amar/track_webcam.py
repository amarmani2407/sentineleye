import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam.")
    raise SystemExit

# For each person ID, remember the last 40 centre points (the trail)
trails = defaultdict(lambda: deque(maxlen=40))
seen_ids = set()

print("Tracking started. Press 'q' in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # track() = detect + give each person an ID that stays the same across frames
    results = model.track(frame, classes=[0], conf=0.5,
                          persist=True, tracker="bytetrack.yaml", verbose=False)

    boxes = results[0].boxes
    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            seen_ids.add(tid)

            # Save the centre of the box for the trail
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trails[tid].append((cx, cy))

            # Box and ID label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw the movement trail
            points = list(trails[tid])
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 200, 255), 2)

    cv2.putText(frame, f"Unique people seen: {len(seen_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("SentinelEye - Step 2: Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
