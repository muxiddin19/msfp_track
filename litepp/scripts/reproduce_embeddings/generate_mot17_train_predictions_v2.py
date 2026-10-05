#!/usr/bin/env python3
"""
v2 (real GMC + real NSA, via real_tracker_v2.py) companion to
generate_mot17_train_predictions.py: completes the 42-file MOT17 archive's
21 TRAIN-sequence files, writing into the SAME output directory as
generate_mot17_test_submission_v2.py so a single zip covers all 14
sequences x 3 detectors.
"""
import json
import time
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_mot17_test_submission as g
import generate_mot17_train_predictions as gt
from real_tracker_v2 import run_tracker_v2_streamed
import verify_official_trackers as vo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    out_dir = Path("~/msfp/mot17_test_submission_v2").expanduser()  # same dir as test predictions v2
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head + real ATL encoder...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [g.LAYER_CHANNELS_V8M[n] for n in g.LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    from atl_paper_faithful import ATLPaperFaithful
    atl_ckpt = torch.load(Path("~/msfp/checkpoints/atl_paper_faithful.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = g.HookedBackbone(device)

    t0 = time.time()
    seq_info = {}
    for seq_name in gt.TRAIN_SEQUENCES:
        seq_dir = mot_root / seq_name
        base_name = seq_name.replace("-FRCNN", "")
        print(f"[{seq_name}] extracting real public-detection features + real ATL scene features...")
        records, n_frames, scene_feat = g.extract_test_sequence(seq_dir, backbone, device)
        n_dets = sum(len(v) for v in records.values())
        seq_info[base_name] = n_frames

        with torch.no_grad():
            tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())

        fused_by_frame = {f: g.fuse_features(r, fusion, device) for f, r in records.items()}
        lines = run_tracker_v2_streamed(seq_dir / "img1", records, fused_by_frame, tau, n_frames,
                                         frame_start=1, frame_offset=0, tau_l_ratio=0.5,
                                         use_gmc=True, use_nsa=True)
        lines_text = "\n".join(lines)

        for det_suffix in gt.DETECTOR_SUFFIXES:
            out_file = out_dir / f"{base_name}-{det_suffix}.txt"
            out_file.write_text(lines_text, encoding="utf-8")
        print(f"  {n_frames} frames, {n_dets} real detections, ATL tau={tau:.3f}, "
              f"{len(lines)} output lines -> {base_name}-{{DPM,FRCNN,SDP}}.txt")

    elapsed = time.time() - t0
    print(f"\nGenerated predictions for {len(gt.TRAIN_SEQUENCES)} train sequences "
          f"({len(gt.TRAIN_SEQUENCES) * 3} files) in {elapsed:.1f}s")

    print("\nComputing real local HOTA on train sequences (not an official score; GT is public)...")
    workdir = Path("~/msfp/trackeval_train_full_v2").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    splits_zero = {seq: 0 for seq in gt.TRAIN_SEQUENCES}
    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, gt.TRAIN_SEQUENCES, "trainfullv2", splits_zero)
    for seq_name in gt.TRAIN_SEQUENCES:
        base_name = seq_name.replace("-FRCNN", "")
        src = (out_dir / f"{base_name}-FRCNN.txt").read_text(encoding="utf-8")
        out_file = trackers_folder / "trainfullv2" / "data" / f"{seq_name}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(src, encoding="utf-8")
    seq_info_full = {seq: seq_info[seq.replace("-FRCNN", "")] for seq in gt.TRAIN_SEQUENCES}
    hota = vo.evaluate_hota(gt_folder, trackers_folder, gt.TRAIN_SEQUENCES, "trainfullv2", seq_info_full)
    mean_hota = float(np.mean(list(hota.values())))
    print(f"Real local HOTA (full train sequences, v2 GMC+NSA pipeline): mean={mean_hota:.2f}")
    for s, v in hota.items():
        print(f"  {s}: {v:.2f}")

    meta_path = out_dir.parent / "mot17_train_predictions_v2_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"real_local_hota_full_train": hota, "mean": mean_hota,
                   "tracker": "real_tracker_v2 (GMC+NSA)",
                   "previous_v1_mean_for_cross_check": 44.31,
                   "note": "Not an official score -- MOT17-train GT is public and was used to train "
                           "the deployed models; included as an honest extra data point and to "
                           "complete the 42-file v2 submission archive."}, f, indent=2)
    print(f"Saved -> {meta_path}")


if __name__ == "__main__":
    main()
