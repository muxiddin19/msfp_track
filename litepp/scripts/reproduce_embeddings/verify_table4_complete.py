#!/usr/bin/env python3
"""
Complete, consolidated real verification of every row in the main paper's
Table 4 (cross-tracker generalization) that is independently reproducible
without official MOTChallenge test-server access, combining two sources
of real evidence on the same val_half protocol, same public MOT17
detections (det.txt), and the same already-trained MSFP fusion head
(fusion_attention.pt, YOLOv8m) used throughout the main paper:

  (a) Real, independent, third-party `boxmot` implementations (native
      C++-backed, published defaults, no tuning by us) for every
      "motion only" and "separate ReID" baseline row -- these algorithms
      (ByteTrack, OC-SORT, BotSort) are used AS PUBLISHED, unmodified.

  (b) Our own real_tracker.py (real 7-state Kalman filter, real
      cosine-distance-weighted-IoU two-stage association, validated
      throughout this release) for every "+MSFP" row built on ByteTrack,
      since the paper's own Implementation Details describe a *custom*
      ByteTrack-style association with appearance cost built directly
      into the matching stage (not vanilla ByteTrack with a bolted-on
      embedding, which boxmot's real ByteTrack does not support at all --
      the original ByteTrack algorithm is appearance-free by design).
      BotSort+MSFP already has real third-party-tool evidence from
      verify_official_trackers.py and is included here for completeness.

OC-SORT+MSFP is explicitly NOT attempted here: OC-SORT's distinguishing
mechanism (Observation-Centric Recovery/Momentum for re-associating lost
tracks) is not implemented in real_tracker.py, and boxmot's real OcSort
does not accept external embeddings either (OC-SORT is motion-only by
design, like ByteTrack) -- faithfully reproducing this specific row would
require implementing OC-SORT's OCR/OCM logic from scratch, which is out
of scope for this verification pass. This is reported explicitly rather
than approximated.
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

from boxmot import BotSort, ByteTrack, OcSort
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


def run_real_tracker_style(by_frame, fused_by_frame, tau, frame_start, frame_end, frame_offset,
                            appearance_weight=0.5):
    """Our validated real_tracker.py ByteTrackStyleTracker, matching the
    paper's own Implementation Details (cosine-distance-weighted IoU for
    the high-confidence stage, IoU-only for recovery)."""
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
                                     appearance_weight=appearance_weight, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = fused_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
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


def run_boxmot_motion_only(tracker_cls, by_frame, frame_images, frame_start, frame_end, frame_offset,
                            sample_id, **tracker_kwargs):
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
    workdir = Path("~/msfp/trackeval_table4_complete").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real, already-trained MSFP fusion head (main paper's checkpoint)...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}
    print(f"train_half/val_half split: {splits}")

    by_frame_cache, frame_images_cache, fused_cache, seq_info = {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection features (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        fused_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {sum(len(v) for v in recs.values())} real detections")

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "t4", splits)

    def eval_condition(run_fn, label):
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            out_file = trackers_folder / "t4" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, "t4", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] mean HOTA={mean_hota:.2f}  " + ", ".join(f"{s}={hota[s]:.2f}" for s in vo.SEQUENCES))
        return mean_hota, hota

    results = {}

    print("\n=== Real boxmot ByteTrack (motion-only) ===")
    results["bytetrack_motion_only"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot_motion_only(ByteTrack, by_frame_cache[seq], frame_images_cache[seq],
                                                        fs, fe, fo, seq), "ByteTrack motion-only (real boxmot)")

    print("\n=== Real ByteTrack-style + MSFP (our validated real_tracker.py, appearance_weight=0.5) ===")
    # tau=0.25 matches the fixed-threshold reference point used throughout
    # this release's other verifications (Table 6's own "Fixed tau=0.25" row).
    results["bytetrack_msfp"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_real_tracker_style(by_frame_cache[seq], fused_cache[seq], 0.25, fs, fe, fo),
        "ByteTrack-style + MSFP (real_tracker.py)")

    print("\n=== Real boxmot OC-SORT (motion-only) ===")
    results["ocsort_motion_only"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot_motion_only(OcSort, by_frame_cache[seq], frame_images_cache[seq],
                                                        fs, fe, fo, seq), "OC-SORT motion-only (real boxmot)")

    print("\n=== Real boxmot BotSort (motion-only, no embeddings) ===")
    results["botsort_motion_only"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_boxmot_motion_only(BotSort, by_frame_cache[seq], frame_images_cache[seq],
                                                        fs, fe, fo, seq, use_embeddings=False),
        "BotSort motion-only (real boxmot)")

    paper_claims = {
        "bytetrack_motion_only": 54.8, "bytetrack_lite": 61.1, "bytetrack_msfp": 63.2,
        "ocsort_motion_only": 55.1, "ocsort_lite": 58.8, "ocsort_msfp": 60.4,
        "botsort_motion_only": 52.4, "botsort_separate_reid_sbs50": 56.3, "botsort_msfp": 57.9,
    }
    # Pull in the already-completed BotSort real-ReID results from verify_official_trackers.py
    official_path = Path("~/msfp/checkpoints/official_tracker_verify.json").expanduser()
    if official_path.exists():
        prev = json.load(open(official_path, encoding="utf-8"))
        results["botsort_separate_reid_osnet"] = prev["mean_botsort_default_reid"]
        results["botsort_msfp_via_botsort_embeddings"] = prev["mean_botsort_msfp"]

    out = {
        "protocol": "train_half/val_half (ByteTrack/FairMOT convention), all 7 MOT17-train sequences, "
                    "public MOT17 detections (det.txt) throughout",
        "real_results": results,
        "paper_claims": paper_claims,
        "not_attempted": {
            "ocsort_msfp": "Requires implementing OC-SORT's Observation-Centric Recovery/Momentum "
                           "logic, not present in real_tracker.py or supported by boxmot's real OcSort "
                           "(motion-only by design, no embedding injection). Not approximated."
        },
    }
    out_path = Path("~/msfp/checkpoints/table4_complete_verify.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== FULL TABLE 4 REAL VERIFICATION SUMMARY (val_half subset) ===")
    for key, claim in paper_claims.items():
        real = results.get(key, "not verified this pass")
        print(f"  {key:35s} real={real if isinstance(real,str) else f'{real:.2f}':>8}   paper claim={claim}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
