# main.py
import cv2
import numpy as np
import time
from detection import load_detector, run_detection, VEHICLE_CLASSES
from optical_flow import compute_motion
from decision import decide_safety, road_instruction_overlay, SafetyHistory

try:
    import RPi.GPIO as GPIO
    USE_GPIO = True
except ImportError:
    USE_GPIO = False

if USE_GPIO:
    GPIO.setmode(GPIO.BCM)
    VIBRATION_BACK_PIN = 17  # Motor for safe to cross
    VIBRATION_END_PIN = 27   # Motor for unsafe to cross
    GPIO.setup(VIBRATION_BACK_PIN, GPIO.OUT)
    GPIO.setup(VIBRATION_END_PIN, GPIO.OUT)
    GPIO.output(VIBRATION_BACK_PIN, GPIO.LOW)
    GPIO.output(VIBRATION_END_PIN, GPIO.LOW)

def vibrate(pin, duration=0.2):
    if USE_GPIO:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)

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

last_vibration_state = None

try:
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
        if road_det and not instruction_mode:
            instruction_mode = True
            instr_time = time.time()
        if instruction_mode:
            # No display; time delay for instruction processing
            _ = road_instruction_overlay(frame)
            if time.time() - instr_time < INSTR_DURATION:
                prev_gray = gray.copy()
                time.sleep(0.01)
                continue
            else:
                instruction_mode = False

        raw_dec = decide_safety(moving_objs, TARGET_HEIGHT)
        safety_history.update(raw_dec)
        final_dec = safety_history.get_smoothed_decision()

        if last_vibration_state is None or final_dec != last_vibration_state:
            if final_dec:
                vibrate(VIBRATION_BACK_PIN)
                print("SAFE TO CROSS")
            else:
                vibrate(VIBRATION_END_PIN)
                print("DO NOT CROSS")
            last_vibration_state = final_dec

        prev_gray = gray.copy()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass

cap.release()
if USE_GPIO:
    GPIO.cleanup()
