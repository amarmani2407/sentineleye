"""Step 6c: watch the anomaly score live and check it behaves sensibly."""
import cv2
from anomaly_model import AnomalyDetector, available

if not available():
    print("No trained model found. Run collect_normal.py and then train_anomaly.py first.")
    raise SystemExit

detector = AnomalyDetector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Could not open the webcam.")
    raise SystemExit

print(f"Alert threshold: {detector.threshold:.5f}. Press 'q' in the window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    score = detector.score(frame)
    unusual = score > detector.threshold
    color = (0, 0, 255) if unusual else (0, 200, 0)

    cv2.putText(frame, f"Score {score:.5f}   Limit {detector.threshold:.5f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "UNUSUAL" if unusual else "normal", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Anomaly test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
