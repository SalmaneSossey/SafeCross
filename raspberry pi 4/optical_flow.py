# optical_flow.py
import cv2
import numpy as np

LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
BASE_THRESH = 1.5

def compute_motion(prev_gray, curr_gray, bbox, dims):
    (sx, sy, ex, ey) = bbox
    roi_prev = prev_gray[sy:ey, sx:ex]
    roi_curr = curr_gray[sy:ey, sx:ex]
    median_disp = 0.0
    if roi_prev.size > 0 and roi_curr.size > 0:
        pts = cv2.goodFeaturesToTrack(roi_prev, maxCorners=40, qualityLevel=0.3, minDistance=7)
        if pts is not None:
            pts += np.array([[sx, sy]], dtype=np.float32)
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **LK_PARAMS)
            if pts_next is not None and status is not None:
                status = status.flatten()
                valid = pts_next[status == 1]
                orig = pts[status == 1]
                if valid.shape[0] > 0:
                    disp = np.linalg.norm(valid - orig, axis=1)
                    median_disp = np.median(disp)
    (w, h) = dims
    roi_area = (ex - sx) * (ey - sy)
    adpt_thresh = BASE_THRESH * ((roi_area / float(w * h)) ** 0.5)
    moving = median_disp > adpt_thresh
    return median_disp, moving
