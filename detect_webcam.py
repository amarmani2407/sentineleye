import cv2
from ultralytics import YOLO

# Load a small, fast pretrained model (downloads automatically the first time)
model = YOLO("yolov8n.pt")

# Open the default webcam (0 = first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam. Close other apps using it and try again.")
    raise SystemExit

print("Webcam started. Press 'q' in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame from the webcam.")
        break

    # Detect only people (class 0 in the COCO dataset), ignore weak guesses
    results = model(frame, classes=[0], conf=0.5, verbose=False)

    person_count = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        person_count += 1

        # Draw the box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {confidence:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the live person count
    cv2.putText(frame, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("SentinelEye - Step 1", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()