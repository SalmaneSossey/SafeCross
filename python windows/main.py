# main.py
import cv2
import numpy as np
import time
from detection import load_detector, run_detection, VEHICLE_CLASSES
from optical_flow import compute_motion
from decision import decide_safety, road_instruction_overlay, SafetyHistory

TARGET_WIDTH = 600
cap = cv2.VideoCapture(0)  # set to 1 if u have camera usb
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25  

ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()
first_frame = cv2.resize(first_frame, (TARGET_WIDTH, int(first_frame.shape[0] * TARGET_WIDTH / first_frame.shape[1])))
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
TARGET_HEIGHT = first_frame.shape[0]

output_file = "output.mp4" #optional
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

net = load_detector()

WINDOW_SIZE = int(fps * 2)
safety_history = SafetyHistory(window_size=WINDOW_SIZE, threshold=0.7)

instruction_mode = False
instruction_trigger_time = None
INSTRUCTION_DURATION = 3  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (TARGET_WIDTH, int(frame.shape[0] * TARGET_WIDTH / frame.shape[1])))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    detections = run_detection(net, frame)
    moving_objects = []
    road_detected = False

    for (label, confidence, bbox) in detections:
        (startX, startY, endX, endY) = bbox
        if label.lower() == "road":
            road_detected = True
        median_disp, moving = compute_motion(prev_gray, gray, bbox, (w, h))
        if moving and label in VEHICLE_CLASSES:
            moving_objects.append((label, bbox, median_disp))
        
        box_color = (0, 255, 0) if moving else (255, 0, 0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.putText(frame, f"Disp: {median_disp:.2f}", (startX, endY + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if road_detected and not instruction_mode:
        instruction_mode = True
        instruction_trigger_time = time.time()

    if instruction_mode:
        frame = road_instruction_overlay(frame)
        if time.time() - instruction_trigger_time < INSTRUCTION_DURATION:
            cv2.imshow("SafeCross AI", frame)
            out.write(frame)   
            prev_gray = gray.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            instruction_mode = False

    raw_decision = decide_safety(moving_objects, TARGET_HEIGHT)
    safety_history.update(raw_decision)
    final_decision = safety_history.get_smoothed_decision()

    decision_text = "SAFE TO CROSS" if final_decision else "DO NOT CROSS"
    decision_color = (0, 255, 0) if final_decision else (0, 0, 255)
    cv2.putText(frame, decision_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, decision_color, 3)

    cv2.imshow("SafeCross AI", frame)
    out.write(frame)  # optionnal
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
