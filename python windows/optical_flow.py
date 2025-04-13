# optical_flow.py
import cv2
import numpy as np

# Lucas–Kanade optical flow parameters
LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

BASE_MOTION_THRESHOLD = 1.5

def compute_motion(prev_gray, curr_gray, bbox, frame_dims):

    (startX, startY, endX, endY) = bbox
    roi_prev = prev_gray[startY:endY, startX:endX]
    roi_curr = curr_gray[startY:endY, startX:endX]
    median_disp = 0.0

    if roi_prev.size > 0 and roi_curr.size > 0:
        pts = cv2.goodFeaturesToTrack(roi_prev, maxCorners=40, qualityLevel=0.3, minDistance=7)
        if pts is not None:
            pts += np.array([[startX, startY]], dtype=np.float32)
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **LK_PARAMS)
            if pts_next is not None and status is not None:
                status = status.flatten()
                valid_pts = pts_next[status == 1]
                orig_pts = pts[status == 1]
                if valid_pts.shape[0] > 0:
                    displacements = np.linalg.norm(valid_pts - orig_pts, axis=1)
                    median_disp = np.median(displacements)
    
    (w, h) = frame_dims
    roi_area = (endX - startX) * (endY - startY)
    adaptive_threshold = BASE_MOTION_THRESHOLD * ((roi_area / float(w * h)) ** 0.5)
    moving = median_disp > adaptive_threshold
    
    return median_disp, moving
