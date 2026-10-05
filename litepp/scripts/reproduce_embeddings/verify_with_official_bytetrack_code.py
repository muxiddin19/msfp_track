#!/usr/bin/env python3
"""
Uses the LITERAL original ByteTrack source code (Zhang et al., ECCV 2022,
cloned from github.com/ifzhang/ByteTrack, yolox/tracker/byte_tracker.py --
not a third-party reimplementation) with the paper's own official default
hyperparameters (track_thresh=0.6, match_thresh=0.9, track_buffer=30,
confirmed from tools/track.py's own argument defaults, which differ from
boxmot's defaults of 0.45/0.8/25 used in our earlier verification), fed
real MOTChallenge public detections (det.txt) directly -- bypassing their
own YOLOX detector entirely, exactly matching the main paper's own stated
"public detections exclusively" protocol.

This is the most faithful possible local check of the "ByteTrack (motion
only)" baseline: the authors' own unmodified algorithm, their own
unmodified hyperparameters, on the same real public detections and
train_half/val_half protocol used throughout this release.

For "+MSFP", the original BYTETracker class has no appearance input at
all (pure IoU/motion association -- confirmed by reading its source), so
we additionally re-run our own validated real_tracker.py at the SAME
official track_thresh=0.6 (rather than the earlier, less-principled 0.25)
for a properly like-for-like comparison against this now more faithful
motion-only baseline.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo
import verify_table4_complete as t4

sys.path.insert(0, "/home/muhiddin/msfp/bytetrack_official")
from yolox.tracker.byte_tracker import BYTETracker

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}

# ByteTrack's own official defaults (tools/track.py), not boxmot's.
OFFICIAL_ARGS = SimpleNamespace(track_thresh=0.6, track_buffer=30, match_thresh=0.9, mot20=False)


def run_official_bytetrack(by_frame, frame_start, frame_end, frame_offset, img_h, img_w):
    tracker = BYTETracker(OFFICIAL_ARGS, frame_rate=30)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        if recs:
            output = np.array([[r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3], r["det_conf"]]
                                for r in recs], dtype=np.float32)
        else:
            output = np.zeros((0, 5), dtype=np.float32)
        # img_size == img_info disables their internal detector-input-scale rescaling,
        # since det.txt boxes are already in original image pixel coordinates.
        online_targets = tracker.update(output, (img_h, img_w), (img_h, img_w))
        out_frame = frame - frame_offset
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            lines.append(f"{out_frame},{t.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp/trackeval_official_bytetrack_code").expanduser()
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

    by_frame_cache, fused_cache, seq_info, img_sizes = {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection features (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        fused_cache[seq] = {f: t4.fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        first_img = next(iter(imgs.values()))
        img_sizes[seq] = first_img.shape[:2]  # (h, w)
        print(f"  {sum(len(v) for v in recs.values())} real detections, image size {img_sizes[seq]}")

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "obt", splits)

    def eval_condition(run_fn, label):
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            out_file = trackers_folder / "obt" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, "obt", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] mean HOTA={mean_hota:.2f}  " + ", ".join(f"{s}={hota[s]:.2f}" for s in vo.SEQUENCES))
        return mean_hota, hota

    print("\n=== LITERAL original ByteTrack code (track_thresh=0.6, match_thresh=0.9, buffer=30) ===")
    mean_official, per_seq_official = eval_condition(
        lambda seq, fs, fe, fo: run_official_bytetrack(by_frame_cache[seq], fs, fe, fo, *img_sizes[seq]),
        "Official ByteTrack code, motion-only")

    print("\n=== Our real_tracker.py + MSFP, same official tau=0.6 for fair comparison ===")
    mean_msfp_06, per_seq_msfp_06 = eval_condition(
        lambda seq, fs, fe, fo: t4.run_real_tracker_style(by_frame_cache[seq], fused_cache[seq], 0.6, fs, fe, fo),
        "real_tracker.py + MSFP (tau=0.6)")

    result = {
        "official_bytetrack_motion_only": {"mean": mean_official, "per_seq": per_seq_official},
        "real_tracker_msfp_tau06": {"mean": mean_msfp_06, "per_seq": per_seq_msfp_06},
        "delta_msfp_vs_official_motion_only": mean_msfp_06 - mean_official,
        "reference_motchallenge_byte_pub_test_set": 56.1,
        "reference_paper_claimed_bytetrack_motion_only": 54.8,
        "reference_paper_claimed_bytetrack_msfp": 63.2,
        "note": "mean_official uses the AUTHORS' OWN unmodified BYTETracker class "
                "(github.com/ifzhang/ByteTrack) with their own official default "
                "hyperparameters, not boxmot's reimplementation or defaults.",
    }
    out_path = Path("~/msfp/checkpoints/official_bytetrack_code_verify.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Official ByteTrack code (motion-only): {mean_official:.2f}  "
          f"(paper claims 54.8 val-set; real MOT17-test public leaderboard BYTE_Pub=56.1)")
    print(f"real_tracker.py + MSFP (tau=0.6):       {mean_msfp_06:.2f}  (paper claims 63.2, test-server)")
    print(f"Real Delta (MSFP vs official motion-only): {result['delta_msfp_vs_official_motion_only']:+.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
