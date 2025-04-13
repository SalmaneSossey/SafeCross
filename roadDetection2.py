import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import cv2
import numpy as np
from PIL import Image

# Load a lightweight segmentation model (you can try deeplabv3_mobilenet_v3_small if needed)
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()

# Get preprocessing transforms
preprocess = weights.transforms()

# Open webcam or video file
cap = cv2.VideoCapture("cross_stree_video.mp4")  # Use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize input (optional, for performance)
    orig_h, orig_w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (512, 512))

    # Convert to PIL and preprocess
    pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(pil_image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # Get predictions
    predictions = output.argmax(0).byte().cpu().numpy()

    # Adjust if road class is different (usually 0 for road, check dataset used)
    road_mask = (predictions == 0).astype(np.uint8) * 255

    # Resize mask to original frame size
    road_mask = cv2.resize(road_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Apply yellow overlay where road is detected
    yellow_overlay = frame.copy()
    yellow_overlay[road_mask == 255] = (0, 255, 255)  # Yellow in BGR

    # Blend with original image if needed
    result = cv2.addWeighted(frame, 0.4, yellow_overlay, 0.6, 0)

    # Show result
    cv2.imshow("Yellow Road Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
