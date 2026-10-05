#!/usr/bin/env python3
"""
Tests whether MOT20's catastrophic HOTA on MOT20-03/05 (and, we now know,
the real OFFICIAL TEST sequences 04/06/08 too -- checked directly: conf is
strictly binary {0,1} dataset-wide, and 04/06/08 have only 0.65-1.06%
conf=1 detections, even worse than train-03/05) is driven by a specific,
fixable implementation detail: real_tracker.py's low-confidence recovery
stage requires score >= tau_l, and ATLPaperFaithful hardcodes tau_min=0.01,
so EVERY exact-conf=0.0 detection is silently excluded from both
association stages -- directly contradicting ByteTrack's own core design
principle (used throughout this pipeline's Implementation Details) of
associating every detection box, including zero-confidence ones, via the
low-confidence recovery stage rather than discarding them outright.

Tests tau_l=0.0 (truly inclusive: score >= 0.0 catches conf=0 detections
too) against the current floored tau_l=max(0.01, ...) on the same real
MOT20 train_half/val_half protocol.
"""
import json
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mot20_baseline_matrix import (MOT20_SEQUENCES, extract_mot20_val_half, fuse_msfp,
                                    setup_trackeval_dirs_mot20, evaluate_full_mot20)
from real_tracker import ByteTrackStyleTracker
from lite_faithful_public import LiteBackbone
from atl_paper_faithful import ATLPaperFaithful
from litepp.models.feature_pyramid import FeatureFusionModule
import verify_official_trackers as vo

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}


def run_tracker(by_frame, feat_by_frame, tau_h, tau_l, fs, fe, fo):
    tracker = ByteTrackStyleTracker(tau_h=tau_h, tau_l=tau_l, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(fs, fe + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = feat_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, scores, feats):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame - fo},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT20/train")
    workdir = Path("~/msfp_honest_repro/trackeval_mot20_zerofloor").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    atl_ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/atl_paper_faithful_mot20.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = vo.HookedBackbone(device)
    lite_backbone = LiteBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in MOT20_SEQUENCES}

    by_frame_cache, msfp_cache, seq_info, taus = {}, {}, {}, {}
    for seq in MOT20_SEQUENCES:
        print(f"[{seq}] extracting...")
        recs, lrecs, imgs, scene_feat = extract_mot20_val_half(mot_root / seq, backbone, lite_backbone, device,
                                                                 splits[seq])
        by_frame_cache[seq] = recs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        n_valid = sum(1 for fr in recs.values() for r in fr if r["det_conf"] > 0)
        n_total_dets = sum(len(v) for v in recs.values())
        with torch.no_grad():
            tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())
        taus[seq] = tau
        print(f"  {n_total_dets} total real detections, {n_valid} with conf>0 ({100*n_valid/max(1,n_total_dets):.1f}%), ATL tau={tau:.3f}")

    gt_folder, trackers_folder = setup_trackeval_dirs_mot20(workdir, mot_root, MOT20_SEQUENCES, "_tmp", splits)

    def run_and_eval(name, tau_l_fn):
        (trackers_folder / name / "data").mkdir(parents=True, exist_ok=True)
        for seq in MOT20_SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
            lines = run_tracker(by_frame_cache[seq], msfp_cache[seq], taus[seq], tau_l_fn(taus[seq]), fs, fe, fo)
            (trackers_folder / name / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        per_seq, agg = evaluate_full_mot20(gt_folder, trackers_folder, MOT20_SEQUENCES, name, seq_info)
        print(f"\n[{name}] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} DetA={agg['DetA']:.2f} "
              f"IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")
        for seq, m in per_seq.items():
            print(f"    {seq}: HOTA={m['HOTA']:.2f}")
        return per_seq, agg

    print("\n=== CURRENT: tau_l floored at 0.01 (excludes exact conf=0) ===")
    per_seq_old, agg_old = run_and_eval("floored", lambda t: max(0.01, t * 0.5))

    print("\n=== FIXED: tau_l=0.0 (truly inclusive, catches conf=0 detections too) ===")
    per_seq_new, agg_new = run_and_eval("zerofloor", lambda t: 0.0)

    out = {"floored_tau_l": {"per_seq": per_seq_old, "agg": agg_old},
           "zero_tau_l": {"per_seq": per_seq_new, "agg": agg_new}, "atl_taus": taus}
    out_path = Path("~/msfp_honest_repro/mot20_zerofloor_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
