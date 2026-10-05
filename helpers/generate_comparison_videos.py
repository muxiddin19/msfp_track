#!/usr/bin/env python3
"""
generate_comparison_videos.py

Runs LITE (single-layer baseline) and MSFP-Track (multi-scale + ATL) on
MOT17 sequences, then renders side-by-side MP4 comparison videos for
ACCV 2026 supplementary material.

Usage (run from ~/lite/):
    python litepp/experiments/generate_comparison_videos.py \
        --data_root /nas/Dataset/MOT/MOT17 \
        --sequences MOT17-02 MOT17-04 \
        --output_dir accv_videos \
        --max_frames 300

Requirements (already in your env):
    pip install scipy opencv-python-headless tqdm ultralytics
"""

import argparse
import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from litepp.trackers import LITEPlusPlusTracker


# ── Colour assignment ─────────────────────────────────────────────────────────
def id_colour(track_id: int):
    """Deterministic BGR colour per track id (visually distinct)."""
    hue = ((track_id * 37) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


# ── IoU helpers ───────────────────────────────────────────────────────────────
def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-6)


def _iou_matrix(dets, tracks):
    nd, nt = len(dets), len(tracks)
    if nd == 0 or nt == 0:
        return np.zeros((nd, nt))
    mat = np.zeros((nd, nt))
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            mat[i, j] = _iou(d[:4], t.box)
    return mat


def _cosine_dist(fd, ft):
    if len(fd) == 0 or len(ft) == 0:
        return np.zeros((len(fd), len(ft)))
    fd = fd / (np.linalg.norm(fd, axis=1, keepdims=True) + 1e-6)
    ft = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-6)
    return 1.0 - fd @ ft.T


# ── Track object ──────────────────────────────────────────────────────────────
class Track:
    def __init__(self, tid, box, feat):
        self.id = tid
        self.box = box.copy()
        self.feat = feat.copy()
        self.hits = 1
        self.time_since_update = 0

    def update(self, box, feat):
        self.box = box.copy()
        self.feat = 0.9 * self.feat + 0.1 * feat   # EMA feature update
        self.hits += 1
        self.time_since_update = 0

    @property
    def confirmed(self):
        return self.hits >= 3


# ── Simple ByteTrack-style tracker ────────────────────────────────────────────
class ByteTracker:
    """
    Two-stage association tracker matching the MSFP-Track paper design:
      Stage 1 — high-confidence detections matched with IoU + cosine cost
      Stage 2 — low-confidence detections matched to lost tracks (IoU only)
    """

    def __init__(self, max_age=30, high_thresh=0.5, low_thresh=0.1,
                 match_thresh=0.7):
        self.max_age = max_age
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.tracks = []
        self._next_id = 1
        self.frame_id = 0

    def reset(self):
        self.tracks = []
        self._next_id = 1
        self.frame_id = 0

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def _hungarian(self, cost, thresh):
        """Run Hungarian on cost matrix; return (matched_r, matched_c, unmatched_r, unmatched_c)."""
        if cost.size == 0:
            return [], [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_r, matched_c = [], []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < thresh:
                matched_r.append(r)
                matched_c.append(c)
        unmatched_r = [i for i in range(cost.shape[0]) if i not in matched_r]
        unmatched_c = [j for j in range(cost.shape[1]) if j not in matched_c]
        return matched_r, matched_c, unmatched_r, unmatched_c

    def update(self, boxes_conf, features):
        """
        Args:
            boxes_conf  (N,5)  [x1,y1,x2,y2,conf]
            features    (N,D)  L2-normalised appearance embeddings

        Returns:
            list of (track_id, x1, y1, x2, y2) for visible confirmed tracks
        """
        self.frame_id += 1

        # Age all existing tracks
        for t in self.tracks:
            t.time_since_update += 1

        # Split detections
        hi_mask = boxes_conf[:, 4] >= self.high_thresh
        lo_mask = (boxes_conf[:, 4] >= self.low_thresh) & ~hi_mask
        hi_dets, hi_feat = boxes_conf[hi_mask], features[hi_mask]
        lo_dets, lo_feat = boxes_conf[lo_mask], features[lo_mask]

        active_idx = list(range(len(self.tracks)))

        # ── Stage 1: high-conf ↔ all tracks ──────────────────────────────────
        s1_matched_r, s1_matched_c = [], []
        if len(hi_dets) > 0 and len(self.tracks) > 0:
            track_feats = np.array([t.feat for t in self.tracks])
            iou_cost = 1 - _iou_matrix(hi_dets, self.tracks)
            cos_cost = _cosine_dist(hi_feat, track_feats)
            cost = 0.5 * iou_cost + 0.5 * cos_cost
            mr, mc, _, _ = self._hungarian(cost, self.match_thresh)
            for r, c in zip(mr, mc):
                self.tracks[c].update(hi_dets[r, :4], hi_feat[r])
                s1_matched_r.append(r)
                s1_matched_c.append(c)

        unmatched_hi = [i for i in range(len(hi_dets)) if i not in s1_matched_r]
        unmatched_tracks = [i for i in active_idx if i not in s1_matched_c]

        # ── Stage 2: low-conf ↔ unmatched lost tracks (IoU only) ─────────────
        if len(lo_dets) > 0 and len(unmatched_tracks) > 0:
            lost_tracks = [self.tracks[i] for i in unmatched_tracks
                           if self.tracks[i].time_since_update > 0]
            lost_idx = [i for i in unmatched_tracks
                        if self.tracks[i].time_since_update > 0]
            if lost_tracks:
                iou_mat = _iou_matrix(lo_dets, lost_tracks)
                cost2 = 1 - iou_mat
                mr, mc, _, _ = self._hungarian(cost2, 0.5)
                matched_lost = set()
                for r, c in zip(mr, mc):
                    if iou_mat[r, c] >= 0.5:
                        self.tracks[lost_idx[c]].update(lo_dets[r, :4], lo_feat[r])
                        matched_lost.add(lost_idx[c])
                unmatched_tracks = [i for i in unmatched_tracks
                                    if i not in matched_lost]

        # ── Create new tracks for unmatched high-conf detections ──────────────
        for r in unmatched_hi:
            self.tracks.append(Track(self._new_id(), hi_dets[r, :4], hi_feat[r]))

        # ── Delete stale tracks ───────────────────────────────────────────────
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        # Return confirmed, freshly-updated tracks
        return [
            (t.id, *t.box.astype(int))
            for t in self.tracks
            if t.confirmed and t.time_since_update == 0
        ]


# ── MOT17 detection loader ────────────────────────────────────────────────────
def load_detections(det_file):
    """Load MOTChallenge public detections → dict {frame_id: (N,5) array}."""
    raw = np.loadtxt(str(det_file), delimiter=',')
    if raw.ndim == 1:
        raw = raw[np.newaxis]
    dets = {}
    for row in raw:
        fid = int(row[0])
        x1, y1, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        conf = float(row[6])
        dets.setdefault(fid, []).append([x1, y1, x1 + w, y1 + h, conf])
    return {k: np.array(v, dtype=np.float32) for k, v in dets.items()}


# ── Draw bounding boxes ───────────────────────────────────────────────────────
def draw_tracks(frame, tracks):
    out = frame.copy()
    for item in tracks:
        tid, x1, y1, x2, y2 = item[0], item[1], item[2], item[3], item[4]
        col = id_colour(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f'ID:{tid}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


# ── Run one tracker on one sequence ──────────────────────────────────────────
def run_sequence(seq_path, litepp_tracker, byte_tracker,
                 detections, max_frames, fixed_threshold=None):
    """
    Returns list of (bgr_frame, active_tracks) for each processed frame.
    """
    img_dir = seq_path / 'img1'
    img_files = sorted(img_dir.glob('*.jpg')) or sorted(img_dir.glob('*.png'))
    if max_frames:
        img_files = img_files[:max_frames]

    byte_tracker.reset()
    litepp_tracker.reset()
    results = []

    for frame_idx, img_path in enumerate(tqdm(img_files, desc=seq_path.name,
                                               leave=False)):
        fid = frame_idx + 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            results.append((np.zeros((1, 1, 3), dtype=np.uint8), []))
            continue

        frame_dets = detections.get(fid, np.empty((0, 5), dtype=np.float32))
        if len(frame_dets) == 0:
            results.append((frame, []))
            continue

        # Threshold selection
        if fixed_threshold is not None:
            thresh = fixed_threshold
        else:
            thresh = litepp_tracker.get_current_threshold()

        mask = frame_dets[:, 4] >= thresh
        filtered = frame_dets[mask]
        if len(filtered) == 0:
            results.append((frame, []))
            continue

        # Feature extraction
        features = litepp_tracker.extract_appearance_features(frame, filtered)
        if features is None or len(features) == 0:
            results.append((frame, []))
            continue

        features = np.array(features)

        # Associate
        active = byte_tracker.update(filtered, features)
        results.append((frame, active))

    return results


# ── Render side-by-side MP4 ───────────────────────────────────────────────────
def render_video(results_lite, results_msfp, output_path, fps=10):
    if not results_lite:
        print(f'  No frames to render for {output_path.name}')
        return

    h, w = results_lite[0][0].shape[:2]
    bar_h = 40
    out_w = w * 2 + 4          # 4px white divider
    total_h = h + bar_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, total_h))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for (fl, tl), (fm, tm) in zip(results_lite, results_msfp):
        drawn_l = draw_tracks(fl, tl)
        drawn_m = draw_tracks(fm, tm)

        bar_l = np.full((bar_h, w, 3), 30, dtype=np.uint8)
        bar_m = np.full((bar_h, w, 3), 30, dtype=np.uint8)

        # Blue-ish label for LITE
        cv2.putText(bar_l, 'LITE  (single-layer baseline)',
                    (10, 28), font, 0.68, (100, 200, 255), 2, cv2.LINE_AA)
        # Green-ish label for MSFP-Track
        cv2.putText(bar_m, 'MSFP-Track  (ours)',
                    (10, 28), font, 0.68, (80, 220, 80), 2, cv2.LINE_AA)

        left  = np.vstack([bar_l, drawn_l])
        right = np.vstack([bar_m, drawn_m])
        divider = np.full((total_h, 4, 3), 200, dtype=np.uint8)  # grey divider
        combined = np.hstack([left, divider, right])
        writer.write(combined)

    writer.release()
    print(f'  Saved: {output_path}  ({len(results_lite)} frames @ {fps} fps)')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate LITE vs MSFP-Track comparison videos')
    parser.add_argument('--data_root',  default='/nas/Dataset/MOT/MOT17')
    parser.add_argument('--sequences',  nargs='+',
                        default=['MOT17-02', 'MOT17-04', 'MOT17-09'])
    parser.add_argument('--output_dir', default='accv_videos')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Max frames per sequence (0 = all)')
    parser.add_argument('--fps',        type=int, default=10)
    parser.add_argument('--yolo_weights', default='yolov8m.pt')
    parser.add_argument('--device',     default='cuda:0')
    parser.add_argument('--lite_threshold', type=float, default=0.25,
                        help='Fixed confidence threshold for LITE baseline')
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames > 0 else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print('Loading YOLO model...')
    yolo = YOLO(args.yolo_weights)

    print('Initialising LITE tracker  (single-layer, fixed threshold)...')
    tracker_lite = LITEPlusPlusTracker(
        model=yolo,
        fusion_type='concat',
        enable_adaptive_threshold=False,
        device=args.device,
        layers=["14"],        # single deepest layer — matches LITE baseline
        output_dim=64,
    )

    print('Initialising MSFP-Track   (multi-scale, adaptive threshold)...')
    tracker_msfp = LITEPlusPlusTracker(
        model=yolo,
        fusion_type='attention',
        enable_adaptive_threshold=True,
        device=args.device,
        # layers defaults to [4, 9, 14] inside create_litepp
        output_dim=128,
    )

    byte_lite = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    byte_msfp = ByteTracker(high_thresh=0.5, low_thresh=0.1)

    # ── Per-sequence loop ─────────────────────────────────────────────────────
    for seq_name in args.sequences:
        print(f'\n── {seq_name} ──────────────────────────')

        seq_path = Path(args.data_root) / 'train' / seq_name
        if not seq_path.exists():
            # Try DPM detector variant (MOT17-XX-DPM)
            for det_type in ['DPM', 'FRCNN', 'SDP']:
                alt = Path(args.data_root) / 'train' / f'{seq_name}-{det_type}'
                if alt.exists():
                    seq_path = alt
                    print(f'  Using detector variant: {seq_path.name}')
                    break
            else:
                print(f'  Not found — skipping')
                continue

        det_file = seq_path / 'det' / 'det.txt'
        if not det_file.exists():
            print(f'  det.txt not found at {det_file}')
            continue

        detections = load_detections(det_file)
        print(f'  Loaded {sum(len(v) for v in detections.values())} '
              f'detections across {len(detections)} frames')

        print('  Running LITE baseline...')
        res_lite = run_sequence(seq_path, tracker_lite, byte_lite,
                                detections, max_frames,
                                fixed_threshold=args.lite_threshold)

        print('  Running MSFP-Track...')
        res_msfp = run_sequence(seq_path, tracker_msfp, byte_msfp,
                                detections, max_frames,
                                fixed_threshold=None)   # ATL adaptive

        out_path = output_dir / f'{seq_name}_comparison.mp4'
        render_video(res_lite, res_msfp, out_path, fps=args.fps)

    print(f'\nDone. Videos saved to: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
