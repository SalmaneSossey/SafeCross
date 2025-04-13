# decision.py
import cv2

def decide_safety(moving_objs, target_h):
    for obj in moving_objs:
        label, bbox, _ = obj
        center = (bbox[1] + bbox[3]) / 2.0
        if label in {"bus", "car", "truck", "motorbike"} and center > 0.6 * target_h:
            return False
    return True

def road_instruction_overlay(frame):
    txt = "STOP! Look LEFT and RIGHT."
    cv2.putText(frame, txt, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame

class SafetyHistory:
    def __init__(self, win_size, thresh=0.7):
        self.win_size = win_size
        self.thresh = thresh
        self.history = []
        self.last_safe = False

    def update(self, raw_decision):
        self.history.append(raw_decision)
        if len(self.history) > self.win_size:
            self.history.pop(0)

    def get_smoothed_decision(self):
        if not self.history:
            return False
        ratio = self.history.count(True) / float(len(self.history))
        safe = ratio >= self.thresh
        if safe and not self.last_safe:
            from tts import speak
            speak("You can cross now")
            self.last_safe = True
        elif not safe:
            self.last_safe = False
        return safe
