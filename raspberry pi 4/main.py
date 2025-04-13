# main.py
import cv2
import numpy as np
import time
from detection import load_detector, run_detection, VEHICLE_CLASSES
from optical_flow import compute_motion
from decision import decide_safety, road_instruction_overlay, SafetyHistory
from tts import speak

TARGET_WIDTH = 600
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

ret, first_frame = cap.read()
if not ret:
    print("Err: No video")
    exit()
first_frame = cv2.resize(first_frame, (TARGET_WIDTH, int(first_frame.shape[0] * TARGET_WIDTH / first_frame.shape[1])))
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
TARGET_HEIGHT = first_frame.shape[0]

net = load_detector()

WINDOW_SIZE = int(fps * 2)
safety_history = SafetyHistory(win_size=WINDOW_SIZE, thresh=0.7)

instruction_mode = False
instr_time = None
INSTR_DURATION = 3
tts_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (TARGET_WIDTH, int(frame.shape[0] * TARGET_WIDTH / frame.shape[1])))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    detections = run_detection(net, frame)
    moving_objs = []
    road_det = False

    for (label, conf, bbox) in detections:
        (sx, sy, ex, ey) = bbox
        if label.lower() == "road":
            road_det = True
        median_disp, moving = compute_motion(prev_gray, gray, bbox, (w, h))
        if moving and label in VEHICLE_CLASSES:
            moving_objs.append((label, bbox, median_disp))
        # (No graphic display; logging omitted)

    if road_det and not instruction_mode:
        instruction_mode = True
        instr_time = time.time()
        tts_done = False

    if instruction_mode:
        # Overlay instruction for TTS trigger (overlay not displayed)
        frame = road_instruction_overlay(frame)
        if not tts_done:
            speak("Stop! Look left and right.")
            tts_done = True
        if time.time() - instr_time < INSTR_DURATION:
            prev_gray = gray.copy()
            time.sleep(0.01)
            continue
        else:
            instruction_mode = False

    raw_dec = decide_safety(moving_objs, TARGET_HEIGHT)
    safety_history.update(raw_dec)
    final_dec = safety_history.get_smoothed_decision()
    # Optionally log the decision:
    print("SAFE TO CROSS" if final_dec else "DO NOT CROSS")
    
    prev_gray = gray.copy()

cap.release()
