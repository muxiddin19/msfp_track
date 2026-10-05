#!/usr/bin/env python3
"""
Bounded recalibration check for the BotSort+MSFP real-tracker verification
(verify_official_trackers.py): BotSort's default appearance_thresh=0.25 /
proximity_thresh=0.5 are tuned for its own default ReID embedding
distribution (OSNet, trained explicitly for person re-identification);
MSFP's embeddings come from a frozen detection backbone via triplet loss
with margin 0.3 and are not guaranteed to have the same genuine/impostor
cosine-similarity distribution. This script holds BotSort's real motion
model, camera-motion compensation, and Kalman filter completely fixed and
sweeps only appearance_thresh over a small grid for the MSFP-embeddings
condition, to check whether the near-zero measured gain
(verify_official_trackers.py: 50.96 vs 51.03, i.e. -0.07, vs the paper's
claimed +1.6 over BotSort's own separate ReID) is an artifact of judging a
different embedding space at a threshold calibrated for a different one.
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
import verify_official_trackers as v

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

APPEARANCE_THRESH_GRID = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp/trackeval_botsort_sweep").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head and extracting real features (same as verify_official_trackers.py)...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [v.LAYER_CHANNELS_V8M[n] for n in v.LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = v.HookedBackbone(device)
    splits = {seq: v.half_split_frame_range(mot_root / seq) for seq in v.SEQUENCES}

    by_frame_cache, frame_images_cache, fused_cache, seq_info = {}, {}, {}, {}
    for seq in v.SEQUENCES:
        recs, imgs = v.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        fused_cache[seq] = {f: v.fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {seq}: {sum(len(r) for r in recs.values())} real detections")

    gt_folder, trackers_folder = v.setup_trackeval_dirs(workdir, mot_root, v.SEQUENCES, "sweep", splits)

    results = {}
    for thresh in APPEARANCE_THRESH_GRID:
        for seq in v.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = v.run_real_botsort(
                by_frame_cache[seq], frame_images_cache[seq], splits[seq] + 1, n_total, splits[seq], seq,
                embeddings_by_frame=fused_cache[seq], use_embeddings=True, appearance_thresh=thresh)
            out_file = trackers_folder / "sweep" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = v.evaluate_hota(gt_folder, trackers_folder, v.SEQUENCES, "sweep", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        results[thresh] = {"mean_hota": mean_hota, "per_seq": hota}
        print(f"appearance_thresh={thresh}: mean HOTA={mean_hota:.2f}  "
              + ", ".join(f"{s}={hota[s]:.2f}" for s in v.SEQUENCES))

    best_thresh = max(results, key=lambda t: results[t]["mean_hota"])
    out = {
        "grid": APPEARANCE_THRESH_GRID,
        "results": results,
        "best_thresh": best_thresh,
        "best_mean_hota": results[best_thresh]["mean_hota"],
        "default_thresh_025_mean_hota": results[0.25]["mean_hota"],
        "botsort_default_reid_mean_hota": 51.03,
        "paper_claimed_botsort_msfp": 57.9,
        "paper_claimed_botsort_default_reid": 56.3,
    }
    out_path = Path("~/msfp/checkpoints/botsort_msfp_thresh_sweep.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Appearance-threshold sweep summary (BotSort + MSFP, val_half) ===")
    print(f"Default thresh (0.25, BotSort's own default): {results[0.25]['mean_hota']:.2f}")
    print(f"Best thresh ({best_thresh}):                   {results[best_thresh]['mean_hota']:.2f}")
    print(f"Reference: BotSort + default OSNet ReID:       51.03")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
