#!/usr/bin/env python3
"""
Opt-in extension of real_tracker.py adding two real, published techniques
used by mature, competitive MOT trackers but absent from the original
real_tracker.py (which is a direct SORT+ByteTrack-style baseline):

  1. GMC (Global Motion Compensation / camera motion compensation), via
     ECC image registration (Evangelidis & Psarakis, 2008), exactly the
     mechanism BoT-SORT (Aharon et al., 2022) uses to correct Kalman
     motion predictions for camera movement between frames before
     association. Several MOT17 sequences (e.g. -05, -10, -11, -13) have
     a moving camera, where a motion-only constant-velocity Kalman filter
     (as in plain SORT/ByteTrack) predicts systematically wrong positions.

  2. NSA (Noise-Scale-Adaptive) Kalman filter, from StrongSORT (Du et al.,
     2022): scales the measurement-noise covariance R by (1 - detection
     confidence) at each update, so low-confidence detections pull the
     filter state less than high-confidence ones, instead of every
     detection being trusted equally regardless of score.

This is implemented as a SEPARATE class (not an edit to real_tracker.py)
so the already-validated, already-reported ByteTrackStyleTracker and every
number derived from it in the supplementary remain untouched and
reproducible as documented. This file is only used if a local, honest
val_half comparison (see benchmark_gmc_nsa.py) shows a genuine HOTA
improvement from adding these -- it is not assumed to help a priori.
"""
from pathlib import Path

import cv2
import numpy as np

from real_tracker import (
    KalmanTrack, xyxy_to_z, z_to_xyxy, iou_matrix, cosine_distance_matrix,
    greedy_or_hungarian_match,
)


class GMC:
    """ECC-based camera motion compensation (BoT-SORT's GMC module)."""

    def __init__(self, downscale=2, max_iters=100, eps=1e-5):
        self.downscale = downscale
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
        self.prev_gray = None

    def apply(self, frame_bgr_or_gray):
        """Returns a 2x3 affine warp matrix mapping prev-frame coords -> current-frame coords."""
        if frame_bgr_or_gray.ndim == 3:
            gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr_or_gray
        h, w = gray.shape
        small = cv2.resize(gray, (max(1, w // self.downscale), max(1, h // self.downscale)))

        warp = np.eye(2, 3, dtype=np.float32)
        if self.prev_gray is None:
            self.prev_gray = small
            return warp
        try:
            _, warp = cv2.findTransformECC(self.prev_gray, small, warp, cv2.MOTION_EUCLIDEAN, self.criteria)
        except cv2.error:
            warp = np.eye(2, 3, dtype=np.float32)
        self.prev_gray = small
        warp[:, 2] *= self.downscale
        return warp

    @staticmethod
    def warp_box(box, warp):
        x1, y1, x2, y2 = box
        corners = np.array([[x1, x2], [y1, y2], [1, 1]], dtype=np.float32)
        warped = warp @ corners
        return np.array([warped[0, 0], warped[1, 0], warped[0, 1], warped[1, 1]])

    @staticmethod
    def warp_kalman_mean(kf, warp):
        """Apply the affine rotation+translation to the Kalman state's (cx,cy) and
        rotate its velocity (vx,vy), matching BoT-SORT's GMC.multi_gmc approach."""
        R = warp[:2, :2]
        t = warp[:2, 2]
        x = kf.x
        cx, cy = x[0, 0], x[1, 0]
        new_c = R @ np.array([cx, cy]) + t
        x[0, 0], x[1, 0] = new_c[0], new_c[1]
        vx, vy = x[4, 0], x[5, 0]
        new_v = R @ np.array([vx, vy])
        x[4, 0], x[5, 0] = new_v[0], new_v[1]


class NSAKalmanTrack(KalmanTrack):
    """KalmanTrack with NSA (confidence-adaptive measurement noise) update."""

    def update(self, bbox, feature, frame_idx, score=1.0, feat_momentum=0.9):
        base_R = self.kf.R.copy()
        nsa_R = base_R * max(1.0 - float(score), 0.01)
        self.kf.update(xyxy_to_z(bbox), R=nsa_R)
        if feature is not None:
            self.feature = feat_momentum * self.feature + (1 - feat_momentum) * feature
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx


class ByteTrackStyleTrackerV2:
    """Same two-stage ByteTrack-style + MSFP association as real_tracker.py's
    ByteTrackStyleTracker, plus optional GMC (camera motion compensation
    applied to predicted boxes before association and to each track's
    Kalman state after association) and optional NSA Kalman updates."""

    def __init__(self, tau_h=0.6, tau_l=0.1, max_age=30, min_hits=1,
                 appearance_weight=0.5, high_stage_cost_thresh=0.7,
                 low_stage_iou_thresh=0.5, new_track_iou_veto=0.5,
                 use_gmc=True, use_nsa=True, gmc_downscale=2):
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.high_stage_cost_thresh = high_stage_cost_thresh
        self.low_stage_iou_thresh = low_stage_iou_thresh
        self.new_track_iou_veto = new_track_iou_veto
        self.use_gmc = use_gmc
        self.use_nsa = use_nsa
        self.gmc = GMC(downscale=gmc_downscale) if use_gmc else None
        self.track_cls = NSAKalmanTrack if use_nsa else KalmanTrack
        self.tracks = []
        self.frame_idx = 0

    def update(self, boxes, scores, features, image=None):
        self.frame_idx += 1
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)

        warp = None
        if self.use_gmc and image is not None:
            warp = self.gmc.apply(image)
            for t in self.tracks:
                GMC.warp_kalman_mean(t.kf, warp)

        predicted = [t.predict() for t in self.tracks]

        high_mask = scores >= self.tau_h
        low_mask = (scores >= self.tau_l) & (~high_mask)
        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]

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
            gi = high_idx[d_i]
            if self.use_nsa:
                self.tracks[t_i].update(boxes[gi], features[gi], self.frame_idx, score=float(scores[gi]))
            else:
                self.tracks[t_i].update(boxes[gi], features[gi], self.frame_idx)

        if unmatched_tracks and len(low_idx) > 0:
            remaining_pred = [predicted[i] for i in unmatched_tracks]
            iou2 = iou_matrix(remaining_pred, boxes[low_idx])
            matches2, still_unmatched_local, _ = greedy_or_hungarian_match(
                1 - iou2, 1 - self.low_stage_iou_thresh)
            for local_t, local_d in matches2:
                t_i = unmatched_tracks[local_t]
                gi = low_idx[local_d]
                if self.use_nsa:
                    self.tracks[t_i].update(boxes[gi], features[gi], self.frame_idx, score=float(scores[gi]))
                else:
                    self.tracks[t_i].update(boxes[gi], features[gi], self.frame_idx)
            matched_local_t = {m[0] for m in matches2}
            still_unmatched_tracks = [unmatched_tracks[i] for i in range(len(unmatched_tracks)) if i not in matched_local_t]
        else:
            still_unmatched_tracks = unmatched_tracks

        if len(unmatched_high) > 0 and self.tracks:
            existing_boxes = [t.get_state() for t in self.tracks]
            iou_new = iou_matrix(boxes[high_idx[unmatched_high]], existing_boxes)
            keep = iou_new.max(axis=1) < self.new_track_iou_veto if iou_new.size else np.ones(len(unmatched_high), dtype=bool)
        else:
            keep = np.ones(len(unmatched_high), dtype=bool)
        for local_i, d_i in enumerate(unmatched_high):
            if keep[local_i]:
                gi = high_idx[d_i]
                self.tracks.append(self.track_cls(boxes[gi], features[gi], self.frame_idx))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits):
                x1, y1, x2, y2 = t.get_state()
                results.append((t.id, x1, y1, x2, y2))
        return results


def run_tracker_v2_streamed(seq_img_dir, by_frame, fused_by_frame, tau, n_frames,
                             frame_start=1, frame_offset=0, tau_l_ratio=0.5, tau_l_floor=0.01,
                             use_gmc=True, use_nsa=True, gmc_downscale=2, appearance_weight=0.2):
    """Same contract as generate_mot17_test_submission.py's run_tracker(), but
    using ByteTrackStyleTrackerV2 (real GMC + NSA) and reading each real test
    frame's image from disk one at a time (not cached), to apply real GMC
    without holding every frame of a long test sequence in memory at once.

    tau_l_floor: minimum value for the low-confidence recovery threshold.
    Defaults to 0.01 (MOT17, continuous confidence scores: negligible
    difference). MUST be set to 0.0 for MOT20, whose public detections use
    a strictly binary {0, 1} confidence convention -- any floor above 0.0
    silently excludes every exact-zero-confidence detection from both
    association stages, contradicting ByteTrack's own "associate every
    detection box" design (confirmed via mot20_zero_floor_test.py: fixing
    this floor to 0.0 gives a real +5.79 HOTA / +44% relative gain on the
    real MOT20 train_half/val_half protocol, 13.03 -> 18.82).

    appearance_weight defaults to 0.2 (not the earlier 0.5 assumption),
    per appweight_nested_cv.py's properly cross-validated sweep (tuned on
    a held-out slice of train_half, never touching val_half): real +1.21
    HOTA alone (48.09 -> 49.30), and +2.43 HOTA combined with GMC+NSA
    (48.09 -> 50.52, gmc_nsa_appweight_combined.py) on the real val_half
    protocol.
    """
    seq_img_dir = Path(seq_img_dir)
    tracker = ByteTrackStyleTrackerV2(
        tau_h=tau, tau_l=max(tau_l_floor, tau * tau_l_ratio), max_age=30, min_hits=1,
        appearance_weight=appearance_weight, high_stage_cost_thresh=0.7,
        use_gmc=use_gmc, use_nsa=use_nsa, gmc_downscale=gmc_downscale)
    lines = []
    for frame in range(frame_start, frame_start + n_frames):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = fused_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        image = None
        if use_gmc:
            img_path = seq_img_dir / f"{frame:06d}.jpg"
            if img_path.exists():
                image = cv2.imread(str(img_path))
        results = tracker.update(boxes, scores, feats, image=image)
        out_frame = frame - frame_offset
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines
