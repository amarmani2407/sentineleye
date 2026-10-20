"""Step 6a: record what your scene looks like when everything is NORMAL."""
import time
import cv2
import numpy as np
from anomaly_model import preprocess

MINUTES = 10        # how long to record
INTERVAL = 0.5      # save one picture every 0.5 seconds

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # same size the server uses
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Could not open the webcam.")
    raise SystemExit

print(f"Recording normal footage for {MINUTES} minutes. Press 'q' in the window to stop early.")
print("Keep the camera still. Do normal things only. No unusual events!")

frames = []
start = time.time()
last_saved = 0.0

while time.time() - start < MINUTES * 60:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    if now - last_saved >= INTERVAL:
        frames.append(np.round(preprocess(frame) * 255).astype(np.uint8))
        last_saved = now

    left = int(MINUTES * 60 - (now - start))
    shown = frame.copy()
    cv2.putText(shown, f"Collected {len(frames)} pictures | {left}s left | q = stop",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Collecting NORMAL footage", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(frames) < 200:
    print(f"Only {len(frames)} pictures. Try to record at least 5 minutes (600 pictures).")
if frames:
    np.save("normal_frames.npy", np.array(frames))
    print(f"Saved {len(frames)} pictures to normal_frames.npy")
