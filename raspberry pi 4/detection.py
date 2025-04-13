# detection.py
import cv2
import numpy as np

PROTO_TXT_PATH = r"deploy.prototxt"  # prototxt file path
MODEL_PATH = r"mobilenet_iter_73000.caffemodel"  # model weights

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor", "road"
]
VEHICLE_CLASSES = {"bus", "car", "truck", "motorbike"}
THRESHOLD = 0.3

def load_detector():
    net = cv2.dnn.readNetFromCaffe(PROTO_TXT_PATH, MODEL_PATH)
    return net

def run_detection(net, frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    result = []
    for i in range(int(detections.shape[2])):
        conf = detections[0, 0, i, 2]
        if conf > THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES):
                continue
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sx, sy, ex, ey) = box.astype("int")
            sx, sy = max(0, sx), max(0, sy)
            ex, ey = min(w, ex), min(h, ey)
            result.append((label, conf, (sx, sy, ex, ey)))
    return result
