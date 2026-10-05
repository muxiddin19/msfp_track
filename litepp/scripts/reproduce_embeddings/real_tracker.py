#!/usr/bin/env python3
"""
A real, standard SORT-style Kalman filter + ByteTrack-style two-stage
association tracker, using MSFP appearance embeddings for the
high-confidence matching stage.

This REPLACES the previous `KalmanBoxTracker` in trackers/ocsort_msfp.py,
whose own docstring says "Simplified Kalman filter ... minimal
implementation for demonstration" and whose `predict()` just returns the
current box unchanged (no motion model at all). The implementation below
uses the actual 7-state constant-velocity Kalman filter from the original
SORT paper (Bewley et al., 2016, cited in the main paper as `sort`), via
the standard `filterpy` library, so that tracking behavior (and therefore
any HOTA numbers computed from it) reflects a real tracker rather than a
box-passthrough stub.

Association (paper Implementation Details, Sec. 4.1):
  Stage 1 (score >= tau_h): cost = lambda * cosine_distance + (1-lambda) * (1-IoU)
  Stage 2 (tau_l <= score < tau_h): cost = 1 - IoU (appearance not used)
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def xyxy_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,s,r] (s=area, r=aspect ratio)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = max(w * h, 1e-6)
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r])


def z_to_xyxy(z):
    cx, cy, s, r = z[:4]
    s = max(s, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def iou_matrix(boxes_a, boxes_b):
    """Vectorized IoU between two sets of [x1,y1,x2,y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = np.asarray(boxes_a)[:, None, :]
    b = np.asarray(boxes_b)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[..., 2] - a[..., 0]) * np.maximum(0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0, b[..., 2] - b[..., 0]) * np.maximum(0, b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def cosine_distance_matrix(feats_a, feats_b):
    if len(feats_a) == 0 or len(feats_b) == 0:
        return np.zeros((len(feats_a), len(feats_b)))
    a = feats_a / (np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-8)
    b = feats_b / (np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-8)
    sim = a @ b.T
    return 1.0 - sim


class KalmanTrack:
    """Standard SORT 7-state constant-velocity Kalman filter track."""
    _next_id = 1

    def __init__(self, bbox, feature, frame_idx):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = xyxy_to_z(bbox).reshape(4, 1)

        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.feature = feature.copy()
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.start_frame = frame_idx
        self.last_frame = frame_idx

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return z_to_xyxy(self.kf.x[:4].flatten())

    def update(self, bbox, feature, frame_idx, feat_momentum=0.9):
        self.kf.update(xyxy_to_z(bbox))
        if feature is not None:
            self.feature = feat_momentum * self.feature + (1 - feat_momentum) * feature
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx

    def get_state(self):
        return z_to_xyxy(self.kf.x[:4].flatten())


def greedy_or_hungarian_match(cost_matrix, cost_threshold):
    """Hungarian assignment, rejecting matches above cost_threshold."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_rows, unmatched_cols = [], [], []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cost_threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class ByteTrackStyleTracker:
    """
    Two-stage (ByteTrack-style) tracker using real Kalman motion and MSFP
    appearance embeddings, matching the paper's Implementation Details
    (Sec. 4.1): cosine-distance-weighted IoU cost for the high-confidence
    stage, IoU-only for the low-confidence recovery stage.
    """

    def __init__(self, tau_h=0.6, tau_l=0.1, max_age=30, min_hits=1,
                 appearance_weight=0.5, high_stage_cost_thresh=0.7,
                 low_stage_iou_thresh=0.5, new_track_iou_veto=0.5):
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.high_stage_cost_thresh = high_stage_cost_thresh
        self.low_stage_iou_thresh = low_stage_iou_thresh
        self.new_track_iou_veto = new_track_iou_veto
        self.tracks = []
        self.frame_idx = 0

    def update(self, boxes, scores, features):
        """
        boxes: (N,4) xyxy, scores: (N,), features: (N,D)
        Returns: list of (track_id, x1,y1,x2,y2) for tracks with enough hits.
        """
        self.frame_idx += 1
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)

        predicted = [t.predict() for t in self.tracks]

        high_mask = scores >= self.tau_h
        low_mask = (scores >= self.tau_l) & (~high_mask)
        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]

        # ---- Stage 1: high-confidence, appearance + motion ----
        track_feats = np.array([t.feature for t in self.tracks]) if self.tracks else np.zeros((0, features.shape[1] if len(features) else 128))
        if self.tracks and len(high_idx) > 0:
            iou = iou_matrix(predicted, boxes[high_idx])
            app_dist = cosine_distance_matrix(track_feats, features[high_idx])
            cost = self.appearance_weight * app_dist + (1 - self.appearance_weight) * (1 - iou)
            matches1, unmatched_tracks, unmatched_high = greedy_or_hungarian_match(
                cost, self.high_stage_cost_thresh)
        else:
            matches1 = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_high = list(range(len(high_idx)))

        for t_i, d_i in matches1:
            self.tracks[t_i].update(boxes[high_idx[d_i]], features[high_idx[d_i]], self.frame_idx)

        # ---- Stage 2: low-confidence recovery, IoU only ----
        if unmatched_tracks and len(low_idx) > 0:
            remaining_pred = [predicted[i] for i in unmatched_tracks]
            iou2 = iou_matrix(remaining_pred, boxes[low_idx])
            matches2, still_unmatched_local, _ = greedy_or_hungarian_match(
                1 - iou2, 1 - self.low_stage_iou_thresh)
            for local_t, local_d in matches2:
                t_i = unmatched_tracks[local_t]
                self.tracks[t_i].update(boxes[low_idx[local_d]], features[low_idx[local_d]], self.frame_idx)
            matched_local_t = {m[0] for m in matches2}
            still_unmatched_tracks = [unmatched_tracks[i] for i in range(len(unmatched_tracks)) if i not in matched_local_t]
        else:
            still_unmatched_tracks = unmatched_tracks

        # ---- New tracks from unmatched high-confidence detections ----
        if len(unmatched_high) > 0 and self.tracks:
            existing_boxes = [t.get_state() for t in self.tracks]
            iou_new = iou_matrix(boxes[high_idx[unmatched_high]], existing_boxes)
            keep = iou_new.max(axis=1) < self.new_track_iou_veto if iou_new.size else np.ones(len(unmatched_high), dtype=bool)
        else:
            keep = np.ones(len(unmatched_high), dtype=bool)
        for local_i, d_i in enumerate(unmatched_high):
            if keep[local_i]:
                gi = high_idx[d_i]
                self.tracks.append(KalmanTrack(boxes[gi], features[gi], self.frame_idx))

        # ---- Remove dead tracks ----
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits):
                x1, y1, x2, y2 = t.get_state()
                results.append((t.id, x1, y1, x2, y2))
        return results
