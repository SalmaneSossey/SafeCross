import cv2

def decide_safety(moving_objs, target_height):

    for obj in moving_objs:
        label, bbox, _ = obj
        # Compute the vertical center of the detected object.
        center_y = (bbox[1] + bbox[3]) / 2.0
        # Mark as unsafe if a vehicle is detected in the critical lower region.
        if label in {"bus", "car", "truck", "motorbike"} and center_y > 0.6 * target_height:
            return False
    return True

def road_instruction_overlay(frame):
    overlay_text = "STOP! Look LEFT and RIGHT."
    cv2.putText(frame, overlay_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame

class SafetyHistory:

    def __init__(self, window_size, threshold=0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []

    def update(self, raw_decision):

        self.history.append(raw_decision)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_smoothed_decision(self):

        if not self.history:
            return False  # Default (not safe) if history is empty.
        safe_ratio = self.history.count(True) / float(len(self.history))
        return safe_ratio >= self.threshold
