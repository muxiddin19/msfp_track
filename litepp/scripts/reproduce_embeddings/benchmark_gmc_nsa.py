#!/usr/bin/env python3
"""
Honest local ablation of the two real, published techniques added in
real_tracker_v2.py (GMC camera-motion compensation, NSA adaptive Kalman
noise) against the already-validated real_tracker.py baseline, on the
exact same val_half protocol / public MOT17 detections / MSFP fusion head
used throughout this release's Table 4 verification
(verify_table4_complete.py), at the same tau=0.25 reference point used for
the already-reported "ByteTrack-style + MSFP" row (real=48.09).

This is a genuine before/after comparison, not tuning to a target: all
four conditions (baseline, +GMC, +NSA, +GMC+NSA) use identical detections,
embeddings, and association thresholds -- only the presence/absence of the
two techniques varies. Whichever condition is honestly best is reported;
if none beat the baseline, that null result is reported too.
"""
import json
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker
from real_tracker_v2 import ByteTrackStyleTrackerV2
import verify_official_trackers as vo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
TAU = 0.25  # same fixed reference point as verify_table4_complete.py's bytetrack_msfp row


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(tracker, by_frame, fused_by_frame, frame_images, frame_start, frame_end, frame_offset,
                 use_images):
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
        if use_images:
            img = frame_images.get(frame)
            results = tracker.update(boxes, scores, feats, image=img)
        else:
            results = tracker.update(boxes, scores, feats)
        out_frame = frame - frame_offset
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp/trackeval_gmc_nsa").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real, already-trained MSFP fusion head...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, fused_cache, seq_info = {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection features (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        fused_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "gn", splits)

    def eval_condition(make_tracker, label, use_images):
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
            tracker = make_tracker()
            lines = run_tracker(tracker, by_frame_cache[seq], fused_cache[seq], frame_images_cache[seq],
                                 fs, fe, fo, use_images)
            out_file = trackers_folder / "gn" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, "gn", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] mean HOTA={mean_hota:.2f}  " + ", ".join(f"{s}={hota[s]:.2f}" for s in vo.SEQUENCES))
        return mean_hota, hota

    results = {}

    print("\n=== Baseline: real_tracker.py (as already reported, tau=0.25) ===")
    results["baseline"], hota_baseline = eval_condition(
        lambda: ByteTrackStyleTracker(tau_h=TAU, tau_l=TAU, max_age=30, min_hits=1,
                                       appearance_weight=0.5, high_stage_cost_thresh=0.7),
        "baseline (no GMC, no NSA)", use_images=False)

    print("\n=== +GMC only (ECC camera motion compensation) ===")
    results["gmc_only"], hota_gmc = eval_condition(
        lambda: ByteTrackStyleTrackerV2(tau_h=TAU, tau_l=TAU, max_age=30, min_hits=1,
                                         appearance_weight=0.5, high_stage_cost_thresh=0.7,
                                         use_gmc=True, use_nsa=False),
        "+GMC only", use_images=True)

    print("\n=== +NSA only (confidence-adaptive Kalman noise) ===")
    results["nsa_only"], hota_nsa = eval_condition(
        lambda: ByteTrackStyleTrackerV2(tau_h=TAU, tau_l=TAU, max_age=30, min_hits=1,
                                         appearance_weight=0.5, high_stage_cost_thresh=0.7,
                                         use_gmc=False, use_nsa=True),
        "+NSA only", use_images=False)

    print("\n=== +GMC+NSA (both) ===")
    results["gmc_nsa"], hota_both = eval_condition(
        lambda: ByteTrackStyleTrackerV2(tau_h=TAU, tau_l=TAU, max_age=30, min_hits=1,
                                         appearance_weight=0.5, high_stage_cost_thresh=0.7,
                                         use_gmc=True, use_nsa=True),
        "+GMC+NSA", use_images=True)

    out = {
        "protocol": "val_half, public MOT17 det.txt, fixed tau=0.25 (same reference point as "
                    "verify_table4_complete.py's bytetrack_msfp=48.09 row)",
        "per_sequence_hota": {
            "baseline": hota_baseline, "gmc_only": hota_gmc,
            "nsa_only": hota_nsa, "gmc_nsa": hota_both,
        },
        "mean_hota": results,
        "previously_reported_baseline_for_cross_check": 48.09,
    }
    out_path = Path("~/msfp/checkpoints/benchmark_gmc_nsa.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY ===")
    for key, val in results.items():
        print(f"  {key:12s} mean HOTA={val:.2f}")
    best = max(results, key=results.get)
    print(f"Best condition: {best} ({results[best]:.2f}), delta vs baseline: {results[best]-results['baseline']:+.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
