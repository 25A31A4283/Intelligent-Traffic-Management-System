from ultralytics import YOLO
import cv2
import os
from traffic_logic import SignalController

folder = os.path.dirname(__file__)
cap = cv2.VideoCapture(os.path.join(folder, "traffic.mp4"))
if not cap.isOpened():
    print("ERROR: could not open video.")
    raise SystemExit

model = YOLO("yolov8n.pt")
amb_model = YOLO(os.path.join(folder, "best.pt"))

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
NUM_LANES = 4
controller = SignalController(NUM_LANES)
frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    frame = cv2.resize(frame, (960, 540))

    h, w = frame.shape[:2]
    lane_w = w // NUM_LANES
    counts = [0] * NUM_LANES

    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        if int(box.cls[0]) not in VEHICLES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        lane = min(((x1 + x2) // 2) // lane_w, NUM_LANES - 1)
        counts[lane] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ambulance check every 3rd frame to keep it fast
    if frame_no % 3 == 0:
        amb = amb_model(frame, verbose=False)[0]
        for b in amb.boxes:
            if float(b.conf[0]) >= 0.7:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                lane = min(((x1 + x2) // 2) // lane_w, NUM_LANES - 1)
                controller.emergency(lane)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "AMBULANCE", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    green_lane, time_left = controller.update(counts)

    for i in range(NUM_LANES):
        x = i * lane_w
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0), 2)
        is_green = (i == green_lane)
        color = (0, 200, 0) if is_green else (0, 0, 255)
        label = f"Lane {i+1}: {counts[i]}  " + (f"GREEN {time_left}s" if is_green else "RED")
        cv2.putText(frame, label, (x + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Smart Traffic", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()