# detection.py
import cv2
import numpy as np

# Path to model files (update these paths as needed)
PROTO_TXT_PATH = r"deploy.prototxt"
MODEL_PATH = r"mobilenet_iter_73000.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "road"]

VEHICLE_CLASSES = {"bus", "car", "truck", "motorbike"}

CONFIDENCE_THRESHOLD = 0.3

def load_detector():

    net = cv2.dnn.readNetFromCaffe(PROTO_TXT_PATH, MODEL_PATH)
    return net

def run_detection(net, frame):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    detections_list = []
    
    for i in range(int(detections.shape[2])):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES):
                continue
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            detections_list.append((label, confidence, (startX, startY, endX, endY)))
    
    return detections_list
