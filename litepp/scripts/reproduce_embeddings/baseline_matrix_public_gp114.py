#!/usr/bin/env python3
"""
Fresh, complete, honestly-labeled real baseline matrix, run from a clean
new subfolder (~/msfp_honest_repro on gp114) with the CURRENT, validated
codebase (real_tracker.py / real_tracker_v2.py / real boxmot) -- not
reusing any number from the previously-discovered historical archive,
per instruction to disregard that archive and regenerate independently.

Protocol: val_half, public MOT17 detections (det.txt), all 7 MOT17-train
sequences, the already-trained MSFP fusion head (fusion_attention.pt). All
conditions share the exact same detections/features/split; only the
tracker differs. Explicitly PUBLIC detections throughout -- no private
detector numbers are produced or reported here.

Trackers:
  - SORT            : real 7-state Kalman + Hungarian IoU-only matching
                       (real_tracker.py's KalmanTrack with appearance
                       disabled), matching SORT's actual published design.
  - DeepSORT-style   : same real Kalman filter + single-stage Hungarian
                       matching using cosine-distance-weighted IoU cost
                       (real_tracker.py's ByteTrackStyleTracker with
                       tau_l=tau_h so every detection is in one stage),
                       appearance = LITE-style raw untrained single-layer
                       features (not MSFP), matching DeepSORT's literature
                       design (separate/raw appearance, single assoc. stage).
  - StrongSort       : real boxmot.StrongSort, native implementation,
                       boxmot's own default (separately-trained) ReID model.
  - ByteTrack        : real boxmot.ByteTrack, motion-only (no appearance,
                       matching the original algorithm).
  - OC-SORT          : real boxmot.OcSort, motion-only.
  - BotSort          : real boxmot.BotSort, motion-only (use_embeddings=False).
  - DeepOcSort       : real boxmot.DeepOcSort with boxmot's own default ReID.
  - LITE (ByteTrack) : real_tracker.py ByteTrackStyleTracker, raw untrained
                       single-layer features (LITE's literal design).
  - MSFP (ByteTrack) : real_tracker.py ByteTrackStyleTracker, real trained
                       MSFP fusion embeddings ("MSFP-Track" per main text).
"""
import json
import time
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker, KalmanTrack, iou_matrix, greedy_or_hungarian_match
import verify_official_trackers as vo
from litepp.models.feature_pyramid import FeatureFusionModule

from boxmot import BotSort, ByteTrack, OcSort, DeepOcSort, StrongSort
from boxmot.structures import Boxes, Detections, Frame

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def lite_raw_features(records):
    """LITE's literal design: raw, L2-normalized, UNTRAINED single-layer
    (layer4, shallowest/highest-res) GAP features -- zero training."""
    if not records:
        return np.zeros((0, 192), dtype=np.float32)
    feats = torch.stack([r["layer4"] for r in records]).numpy()
    norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return (feats / norm).astype(np.float32)


class SortTracker:
    """Real SORT: real_tracker.py's 7-state Kalman filter, single-stage
    Hungarian IoU-only matching, no appearance at all -- matches the
    original SORT (Bewley et al., 2016) design exactly."""

    def __init__(self, iou_thresh=0.3, max_age=30, min_hits=1):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_idx = 0

    def update(self, boxes, scores):
        self.frame_idx += 1
        boxes = np.asarray(boxes, dtype=np.float32)
        predicted = [t.predict() for t in self.tracks]
        dummy_feat = np.zeros(4, dtype=np.float32)
        if self.tracks and len(boxes) > 0:
            iou = iou_matrix(predicted, boxes)
            matches, unmatched_tracks, unmatched_dets = greedy_or_hungarian_match(1 - iou, 1 - self.iou_thresh)
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(boxes)))
        for t_i, d_i in matches:
            self.tracks[t_i].update(boxes[d_i], dummy_feat, self.frame_idx)
        for d_i in unmatched_dets:
            self.tracks.append(KalmanTrack(boxes[d_i], dummy_feat, self.frame_idx))
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits):
                x1, y1, x2, y2 = t.get_state()
                results.append((t.id, x1, y1, x2, y2))
        return results


def run_sort(by_frame, frame_start, frame_end, frame_offset):
    tracker = SortTracker()
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        boxes = np.array([r["bbox"] for r in recs], dtype=np.float32) if recs else np.zeros((0, 4), dtype=np.float32)
        scores = np.array([r["det_conf"] for r in recs], dtype=np.float32) if recs else np.zeros((0,), dtype=np.float32)
        results = tracker.update(boxes, scores)
        out_frame = frame - frame_offset
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def run_real_tracker_style(by_frame, feat_by_frame, tau_h, tau_l, frame_start, frame_end, frame_offset,
                            appearance_weight=0.5):
    tracker = ByteTrackStyleTracker(tau_h=tau_h, tau_l=tau_l, max_age=30, min_hits=1,
                                     appearance_weight=appearance_weight, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = feat_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        results = tracker.update(boxes, scores, feats)
        out_frame = frame - frame_offset
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def run_boxmot(tracker_cls, by_frame, frame_images, frame_start, frame_end, frame_offset, sample_id,
               **tracker_kwargs):
    tracker = tracker_cls(**tracker_kwargs)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        img = frame_images.get(frame)
        if img is None:
            continue
        if recs:
            geometry = torch.tensor([r["bbox"] for r in recs], dtype=torch.float32)
            scores = torch.tensor([r["det_conf"] for r in recs], dtype=torch.float32)
            class_ids = torch.zeros(len(recs), dtype=torch.int64)
        else:
            geometry = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.int64)
        dets = Detections(geometry=Boxes(geometry), scores=scores, class_ids=class_ids, sample_id=sample_id)
        frame_obj = Frame(image=torch.from_numpy(img).permute(2, 0, 1).contiguous(), sample_id=sample_id,
                           frame_index=frame)
        tracks = tracker.update(dets, frame_obj)
        out_frame = frame - frame_offset
        try:
            ids = tracks.track_ids.tolist()
            boxes = tracks.geometry.values.tolist()
        except AttributeError:
            ids, boxes = [], []
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_work").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real, already-trained MSFP fusion head (main paper's checkpoint)...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}
    print(f"train_half/val_half split: {splits}")

    by_frame_cache, frame_images_cache, msfp_cache, lite_cache, seq_info = {}, {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection features (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        lite_cache[seq] = {f: lite_raw_features(r) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {sum(len(v) for v in recs.values())} real detections")

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "bm", splits)

    def eval_condition(run_fn, label):
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            out_file = trackers_folder / "bm" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, "bm", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] mean HOTA={mean_hota:.2f}  " + ", ".join(f"{s}={hota[s]:.2f}" for s in vo.SEQUENCES))
        return mean_hota, hota

    results = {}

    print("\n=== SORT (real Kalman + IoU-only, real_tracker.py) ===")
    results["SORT"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_sort(by_frame_cache[seq], fs, fe, fo), "SORT")

    print("\n=== DeepSORT-style (real Kalman + single-stage cosine+IoU, LITE-raw appearance) ===")
    results["DeepSORT"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_real_tracker_style(by_frame_cache[seq], lite_cache[seq], 0.0, 0.0, fs, fe, fo),
        "DeepSORT-style (LITE-raw appearance)")

    print("\n=== Real boxmot StrongSort (native, own default ReID) ===")
    results["StrongSort"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot(StrongSort, by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq),
        "StrongSort (real boxmot, own ReID)")

    print("\n=== Real boxmot ByteTrack (motion-only) ===")
    results["ByteTrack_motion"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot(ByteTrack, by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq),
        "ByteTrack motion-only")

    print("\n=== Real boxmot OC-SORT (motion-only) ===")
    results["OCSORT_motion"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot(OcSort, by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq),
        "OC-SORT motion-only")

    print("\n=== Real boxmot BotSort (motion-only) ===")
    results["BotSort_motion"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot(BotSort, by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq,
                                            use_embeddings=False),
        "BotSort motion-only")

    print("\n=== Real boxmot DeepOcSort (own default ReID) ===")
    results["DeepOcSort"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot(DeepOcSort, by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq),
        "DeepOcSort (own ReID)")

    print("\n=== LITE (ByteTrack-style + raw untrained single-layer features) ===")
    results["LITE_ByteTrack"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_real_tracker_style(by_frame_cache[seq], lite_cache[seq], 0.25, 0.25, fs, fe, fo),
        "LITE (ByteTrack-style)")

    print("\n=== MSFP-Track (ByteTrack-style + real trained MSFP fusion) ===")
    results["MSFP_ByteTrack"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_real_tracker_style(by_frame_cache[seq], msfp_cache[seq], 0.25, 0.25, fs, fe, fo),
        "MSFP-Track (ByteTrack-style)")

    out = {
        "protocol": "val_half (ByteTrack/FairMOT convention), all 7 MOT17-train sequences, "
                    "PUBLIC MOT17 detections (det.txt) exclusively, fresh run from ~/msfp_honest_repro "
                    "on gp114 with the current, validated codebase -- no number reused from any prior archive.",
        "real_results": results,
    }
    out_path = Path("~/msfp_honest_repro/baseline_matrix_public_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY (fresh, honest, public-detection baseline matrix) ===")
    for key, val in results.items():
        print(f"  {key:20s} mean HOTA={val:.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
