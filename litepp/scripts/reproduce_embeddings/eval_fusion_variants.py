#!/usr/bin/env python3
"""
Evaluates every fusion-head variant (the original, possibly leak-inflated
fusion_attention.pt, plus the 4 genuinely leak-free variants from
train_fusion_sweep.py) through the full validated real tracking stack
(GMC+NSA+appearance_weight=0.2) on the real held-out val_half, with full
metrics. This answers two real questions at once: whether the original
checkpoint's reported performance was inflated by training-data leakage,
and whether any dropout/weight_decay setting gives a genuine improvement.
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

VARIANTS = {
    "original_fusion_attention (possibly leak-inflated)": ("~/msfp_honest_repro/checkpoints/fusion_attention.pt", 0.1),
    "default_leakfree": ("~/msfp_honest_repro/checkpoints/fusion_default_leakfree.pt", 0.1),
    "dropout0.3": ("~/msfp_honest_repro/checkpoints/fusion_dropout0.3.pt", 0.3),
    "wd1e-4": ("~/msfp_honest_repro/checkpoints/fusion_wd1e-4.pt", 0.1),
    "dropout0.3_wd1e-4": ("~/msfp_honest_repro/checkpoints/fusion_dropout0.3_wd1e-4.pt", 0.3),
}


def run_tracker(by_frame, feat_by_frame, frame_images, fs, fe, fo):
    tracker = ByteTrackStyleTrackerV2(tau_h=0.25, tau_l=0.25, max_age=30, min_hits=1,
                                       appearance_weight=0.2, high_stage_cost_thresh=0.7,
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
    workdir = Path("~/msfp_honest_repro/trackeval_fusion_variants").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, seq_info = {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection boxes+features (val_half)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "_tmp", splits)

    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    results = {}
    for name, (ckpt_path, dropout_p) in VARIANTS.items():
        print(f"\n=== {name} ===")
        ckpt = torch.load(Path(ckpt_path).expanduser(), map_location=device, weights_only=False)
        fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                      fusion_type="attention", dropout_p=dropout_p).to(device)
        fusion.load_state_dict(ckpt["state_dict"])
        fusion.eval()

        @torch.no_grad()
        def fuse(records, fusion=fusion):
            if not records:
                return np.zeros((0, 128), dtype=np.float32)
            layer_feats = [torch.stack([r[n] for r in records]).to(device) for n in LAYER_ORDER]
            emb = fusion(layer_feats)
            emb = F.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy()

        tag = name.split()[0].replace("(", "").replace(")", "")
        (trackers_folder / tag / "data").mkdir(parents=True, exist_ok=True)
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
            feat_by_frame = {f: fuse(r) for f, r in by_frame_cache[seq].items()}
            lines = run_tracker(by_frame_cache[seq], feat_by_frame, frame_images_cache[seq], fs, fe, fo)
            (trackers_folder / tag / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        _, agg = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, tag, seq_info)
        results[name] = agg
        print(f"  HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} DetA={agg['DetA']:.2f} "
              f"IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")

    out_path = Path("~/msfp_honest_repro/fusion_variants_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (all real, same GMC+NSA+aw0.2 stack, val_half) ===")
    for name, agg in results.items():
        print(f"  {name:55s} HOTA={agg['HOTA']:.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
