import cv2
import numpy as np
import time
import torch
from collections import deque

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog


# Try import Hungarian (scipy)
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# PARAMETERS 
video_path = r"etc/vid/WIN_20251002_10_34_38_Pro.mp4"  
weights_path = "output/model_final.pth"                   
DATASET_METADATA_NAME = "my_dataset_train"               
CLASS_NAMES = ["objects", "Nextar", "Steam Cake"]         
SCORE_THRESH = 0.9
LINE_X1, LINE_X2 = 620, 622    # kordinat
IOU_THRESHOLD = 0.3     # untuk data association fallback (not used in Hungarian)
MAX_AGE = 30            # frames until track is deleted
MIN_HITS = 2            # hits to consider a track confirmed
MAX_DISTANCE = 200      # used for center-distance gating in greedy fallback
KALMAN_STD_POS = 1.0
KALMAN_STD_VEL = 0.5



# Utility functions
def xywh_to_x1y1x2y2(xywh):
    x, y, w, h = xywh
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2]

def iou(bb_test, bb_gt):
    # bb = [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    o = inter / (area1 + area2 - inter + 1e-6)
    return o


# Simple Kalman implementation for bounding boxes (adapted for SORT-like)
# State: [cx, cy, s, r, vx, vy, vs] where s=scale(area), r=aspect ratio
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        """
        bbox: [x1,y1,x2,y2]
        """
        # define constant velocity model in state-space
        # state vector: [cx, cy, s, r, vx, vy, vs]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w * h
        r = w / float(h + 1e-6)

        # initialize state
        self.x = np.zeros((7,1))
        self.x[0,0] = cx
        self.x[1,0] = cy
        self.x[2,0] = s
        self.x[3,0] = r

        # covariance
        self.P = np.eye(7) * 10.0

        # motion matrix
        self.F = np.eye(7)
        dt = 1.0
        self.F[0,4] = dt
        self.F[1,5] = dt
        self.F[2,6] = dt

        # observation matrix - we observe cx,cy,s,r
        self.H = np.zeros((4,7))
        self.H[0,0] = 1.0
        self.H[1,1] = 1.0
        self.H[2,2] = 1.0
        self.H[3,3] = 1.0

        # process noise
        q = 1.0
        self.Q = np.eye(7) * q
        # measurement noise
        r_var = 1.0
        self.R = np.eye(4) * r_var

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

        self.history = []

    def predict(self):
        # x = F x
        self.x = np.dot(self.F, self.x)
        # P = F P F^T + Q
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        # return predicted bbox
        return self.get_state()

    def update(self, bbox):
        # bbox: [x1,y1,x2,y2]
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w * h
        r = w / float(h + 1e-6)
        z = np.array([[cx],[cy],[s],[r]])
        # y = z - Hx
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S + 1e-6*np.eye(4)))
        self.x = self.x + K.dot(y)
        I = np.eye(7)
        self.P = (I - K.dot(self.H)).dot(self.P)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        # convert state to bbox [x1,y1,x2,y2]
        cx = self.x[0,0]
        cy = self.x[1,0]
        s = max(1e-6, self.x[2,0])
        r = self.x[3,0]
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        return [x1, y1, x2, y2]




# SORT Manager
class Sort:
    def __init__(self, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # list of KalmanBoxTracker
        self.frame_count = 0

        # crossing bookkeeping
        self.count_in = 0
        self.count_out = 0
        self.counted_ids = set()  # to avoid double counting per crossing (we'll manage with state)

        # per-track crossing state
        self.track_history = {}  # id -> deque of centers (last N)

    def update(self, dets):
        """
        dets: numpy array Nx5 of detections [x1,y1,x2,y2,score]
        returns: list of tracks: [[x1,y1,x2,y2,track_id], ...]
        """
        self.frame_count += 1

        # Predict existing trackers
        trks = []
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks.append(pos)
        trks = np.array(trks) if len(trks) > 0 else np.empty((0,4))

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

        # Update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t in matched:
                d = matched[t]
                trk.update(dets[d, :4])
                # reset tracked history store
                tid = trk.id
                center = self._center_of_bbox(dets[d,:4])
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=10)
                self.track_history[tid].append(center)
            else:
                # let tracker age
                trk.time_since_update += 1

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4].tolist())
            self.trackers.append(trk)
            tid = trk.id
            center = self._center_of_bbox(dets[i,:4])
            self.track_history[tid] = deque(maxlen=10)
            self.track_history[tid].append(center)

        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                # delete track and its history
                try:
                    del self.track_history[trk.id]
                except Exception:
                    pass
                self.trackers.remove(trk)

        # Prepare return list
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                tid = trk.id
                ret.append([bbox[0], bbox[1], bbox[2], bbox[3], tid])

        return np.array(ret)

    def _center_of_bbox(self, bb):
        x1,y1,x2,y2 = bb[:4]
        return (int((x1+x2)/2.0), int((y1+y2)/2.0))

    def _associate_detections_to_trackers(self, dets, trks):
        """
        dets: Nx5, trks: Mx4
        returns:
            matched: dict {trk_index: det_index}
            unmatched_dets: list of det indices
            unmatched_trks: list of trk indices
        """
        if len(trks) == 0:
            return {}, list(range(dets.shape[0])), []

        if dets.shape[0] == 0:
            return {}, [], list(range(trks.shape[0]))

        # compute cost matrix as 1 - iou
        cost_matrix = np.zeros((trks.shape[0], dets.shape[0]), dtype=np.float32)
        for t in range(trks.shape[0]):
            for d in range(dets.shape[0]):
                cost_matrix[t, d] = 1.0 - iou(trks[t], dets[d, :4])

        # if scipy available -> hungarian
        matched_indices = []
        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices = list(zip(row_ind.tolist(), col_ind.tolist()))
            # filter by iou threshold
            matched = {}
            unmatched_trks = list(range(trks.shape[0]))
            unmatched_dets = list(range(dets.shape[0]))
            for r,c in matched_indices:
                if cost_matrix[r,c] < (1.0 - self.iou_threshold):
                    matched[r] = c
                    if r in unmatched_trks: unmatched_trks.remove(r)
                    if c in unmatched_dets: unmatched_dets.remove(c)
            return matched, unmatched_dets, unmatched_trks
        else:
            # greedy match based on IoU & distance gating
            matched = {}
            unmatched_trks = list(range(trks.shape[0]))
            unmatched_dets = list(range(dets.shape[0]))
            # compute center distances for gating if desired
            centers_trk = np.array([[ (t[0]+t[2])/2.0, (t[1]+t[3])/2.0 ] for t in trks])
            centers_det = np.array([[ (d[0]+d[2])/2.0, (d[1]+d[3])/2.0 ] for d in dets])
            D = np.linalg.norm(centers_trk[:,None,:] - centers_det[None,:,:], axis=2)
            # greedy: highest IoU first
            pairs = []
            for t in range(trks.shape[0]):
                for d in range(dets.shape[0]):
                    pairs.append((cost_matrix[t,d], t, d))
            pairs = sorted(pairs, key=lambda x: x[0])  # ascending cost -> descending IoU
            assigned_tr = set()
            assigned_dt = set()
            for cost, t, d in pairs:
                if t in assigned_tr or d in assigned_dt:
                    continue
                if cost > (1.0 - self.iou_threshold):
                    continue
                # gating by center distance
                if D[t,d] > MAX_DISTANCE:
                    continue
                matched[t] = d
                assigned_tr.add(t)
                assigned_dt.add(d)
                if t in unmatched_trks: unmatched_trks.remove(t)
                if d in unmatched_dets: unmatched_dets.remove(d)
            return matched, unmatched_dets, unmatched_trks



# Setup Detectron2 predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

try:
    meta = MetadataCatalog.get(DATASET_METADATA_NAME)
    class_names = meta.get("thing_classes", CLASS_NAMES)
except Exception:
    class_names = CLASS_NAMES




# Main loop
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: cannot open video:", video_path)
    raise SystemExit

sort_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
center_line_x = int((LINE_X1 + LINE_X2) / 2)

print("Starting processing... (press 'q' to quit)  Has SciPy:", _HAS_SCIPY)
fps_time = time.time()
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    frame_idx += 1

    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")

    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
    else:
        boxes = np.array([])
        classes = np.array([])
        scores = np.array([])

    dets = []
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box.astype(int)
        score = float(scores[i]) if len(scores)>0 else 1.0
        if score < SCORE_THRESH:
            continue
        dets.append([x1, y1, x2, y2, score, int(classes[i])])
    dets_np = np.array([d[:5] for d in dets]) if len(dets)>0 else np.empty((0,5))

    # update tracker (returns tracked bboxes and track ids)
    tracked = sort_tracker.update(dets_np)

    # tracked: Nx5 [x1,y1,x2,y2,tid]
    # draw tracked objects & count crossings using track history from sort_tracker
    for t in tracked:
        x1,y1,x2,y2,tid = t
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        # draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)
        cv2.putText(frame, f"ID:{int(tid)}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # manage counting using track_history - the tracker already stores histories internally
        # we decide crossing if the history shows center was left of LINE_X1 and now right of LINE_X2, or vice versa
        # Identify last two centers if available
        # get history if exists
        hist = sort_tracker.track_history.get(int(tid), None)
        if hist is not None and len(hist) >= 2:
            prev_cx, prev_cy = hist[-2]
            curr_cx, curr_cy = hist[-1]
            # left -> right
            if prev_cx < LINE_X1 and curr_cx > LINE_X2:
                # ensure we only count once per track per crossing direction (we use counted_ids with (tid,dir))
                key = (int(tid), "in")
                if key not in sort_tracker.counted_ids:
                    sort_tracker.count_in += 1
                    sort_tracker.counted_ids.add(key)
                    print(f"ID {tid} crossed LEFT->RIGHT (IN). total IN: {sort_tracker.count_in}")
            elif prev_cx > LINE_X2 and curr_cx < LINE_X1:
                key = (int(tid), "out")
                if key not in sort_tracker.counted_ids:
                    sort_tracker.count_out += 1
                    sort_tracker.counted_ids.add(key)
                    print(f"ID {tid} crossed RIGHT->LEFT (OUT). total OUT: {sort_tracker.count_out}")

    # draw counting area and center line
    cv2.line(frame, (LINE_X1,0), (LINE_X1, frame.shape[0]), (255,0,0), 2)
    cv2.line(frame, (LINE_X2,0), (LINE_X2, frame.shape[0]), (255,0,0), 2)
    cv2.line(frame, (center_line_x,0),(center_line_x,frame.shape[0]), (0,255,255), 1)

    # overlay counts
    cv2.putText(frame, f"IN: {sort_tracker.count_in}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"OUT: {sort_tracker.count_out}", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # fps display
    if frame_idx % 10 == 0:
        now = time.time()
        fps = 10.0 / (now - fps_time + 1e-6)
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("SORT Counting (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
