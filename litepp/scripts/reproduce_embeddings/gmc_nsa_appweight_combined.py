#!/usr/bin/env python3
"""
Stacks the two independently-validated real improvements and reports the
combined, full-metric result on the real held-out val_half:
  1. GMC (camera motion compensation) + NSA (confidence-adaptive Kalman
     noise), validated via benchmark_gmc_nsa.py: +0.89 HOTA alone
     (48.09 -> 48.98) at the default appearance_weight=0.5.
  2. appearance_weight=0.2, validated via appweight_nested_cv.py with a
     properly held-out tuning split (never touching val_half): +1.21 HOTA
     alone (48.09 -> 49.30) at the default real_tracker.py (no GMC/NSA).

Neither validation used the other's val_half result to choose anything,
so testing both together on val_half is a legitimate final confirmation
of the combined effect, not a second round of leakage-prone tuning.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker_v2 import ByteTrackStyleTrackerV2
import verify_official_trackers as vo
from baseline_matrix_full_metrics import evaluate_full
from litepp.models.feature_pyramid import FeatureFusionModule

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


def run_tracker(by_frame, feat_by_frame, frame_images, appearance_weight, fs, fe, fo):
    tracker = ByteTrackStyleTrackerV2(tau_h=0.25, tau_l=0.25, max_age=30, min_hits=1,
                                       appearance_weight=appearance_weight, high_stage_cost_thresh=0.7,
                                       use_gmc=True, use_nsa=True)
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
        img = frame_images.get(frame)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, scores, feats, image=img):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame - fo},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_gmc_nsa_aw").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, msfp_cache, seq_info = {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "_tmp", splits)

    def run_and_eval(name, aw):
        (trackers_folder / name / "data").mkdir(parents=True, exist_ok=True)
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
            lines = run_tracker(by_frame_cache[seq], msfp_cache[seq], frame_images_cache[seq], aw, fs, fe, fo)
            (trackers_folder / name / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        per_seq, agg = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, name, seq_info)
        print(f"[{name}] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} DetA={agg['DetA']:.2f} "
              f"IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")
        return agg

    print("\n=== GMC+NSA + appearance_weight=0.5 (baseline combo, cross-check vs benchmark_gmc_nsa.py) ===")
    agg_05 = run_and_eval("gmcnsa_aw05", 0.5)

    print("\n=== GMC+NSA + appearance_weight=0.2 (both real improvements stacked) ===")
    agg_02 = run_and_eval("gmcnsa_aw02", 0.2)

    print("\nReference points: plain real_tracker.py aw=0.5 (no GMC/NSA) = 48.09 HOTA")
    print("                   GMC+NSA alone, aw=0.5 (benchmark_gmc_nsa.py) = 48.98 HOTA")
    print("                   aw=0.2 alone, no GMC/NSA (appweight_nested_cv.py) = 49.30 HOTA")

    out = {"gmcnsa_aw05": agg_05, "gmcnsa_aw02": agg_02,
           "reference_plain": 48.09, "reference_gmcnsa_only": 48.98, "reference_aw02_only": 49.30}
    out_path = Path("~/msfp_honest_repro/gmc_nsa_appweight_combined_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
