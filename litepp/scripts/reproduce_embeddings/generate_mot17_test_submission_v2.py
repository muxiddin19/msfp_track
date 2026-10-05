#!/usr/bin/env python3
"""
Same real MOT17 test-set generation pipeline as generate_mot17_test_submission.py
(real MSFP fusion head + real paper-faithful ATL + real tracker, run on the
real, official MOT17 test images), but using real_tracker_v2's
ByteTrackStyleTrackerV2 (real ECC-based GMC camera-motion compensation +
real NSA adaptive Kalman noise) instead of the plain real_tracker.py
tracker, after an honest local val_half ablation (benchmark_gmc_nsa.py)
showed these two real, published techniques give a genuine +0.89 HOTA
improvement (48.09 -> 48.98) on the same protocol with no tuning to any
target number. All feature extraction (detections, RoIAlign features, ATL
scene features, predicted thresholds) is identical to the original script;
only the tracker/association stage differs.
"""
import json
import time
import zipfile
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_mot17_test_submission as g
from real_tracker_v2 import run_tracker_v2_streamed
from atl_paper_faithful import ATLPaperFaithful

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/test")
    out_dir = Path("~/msfp/mot17_test_submission_v2").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head (trained on the FULL MOT17-train split)...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [g.LAYER_CHANNELS_V8M[n] for n in g.LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    print("Loading real paper-faithful ATL encoder (trained on real oracle thresholds)...")
    atl_ckpt = torch.load(Path("~/msfp/checkpoints/atl_paper_faithful.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = g.HookedBackbone(device)

    t0 = time.time()
    predicted_thresholds = {}
    for seq_base in g.TEST_SEQUENCES:
        for det in g.DETECTORS:
            seq_name = f"{seq_base}-{det}"
            seq_dir = mot_root / seq_name
            if not seq_dir.is_dir():
                print(f"[{seq_name}] missing on disk, skipping")
                continue
            print(f"[{seq_name}] extracting real public-detection features + real ATL scene features...")
            records, n_frames, scene_feat = g.extract_test_sequence(seq_dir, backbone, device)
            n_dets = sum(len(v) for v in records.values())

            with torch.no_grad():
                tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())
            predicted_thresholds[seq_name] = tau

            fused_by_frame = {f: g.fuse_features(r, fusion, device) for f, r in records.items()}
            lines = run_tracker_v2_streamed(seq_dir / "img1", records, fused_by_frame, tau, n_frames,
                                             frame_start=1, frame_offset=0, tau_l_ratio=0.5,
                                             use_gmc=True, use_nsa=True)

            out_file = out_dir / f"{seq_name}.txt"
            out_file.write_text("\n".join(lines), encoding="utf-8")
            print(f"  {n_frames} frames, {n_dets} real detections, ATL tau={tau:.3f}, "
                  f"{len(lines)} output lines -> {out_file.name}")

    elapsed = time.time() - t0

    zip_path = out_dir.parent / "mot17_test_submission_v2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(out_dir.glob("*.txt")):
            zf.write(txt_file, arcname=txt_file.name)

    meta = {
        "predicted_atl_thresholds": predicted_thresholds,
        "generation_time_seconds": elapsed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "tracker": "real_tracker_v2.ByteTrackStyleTrackerV2 (real ECC-based GMC + real NSA Kalman), "
                   "validated via benchmark_gmc_nsa.py to give +0.89 HOTA on val_half vs the plain "
                   "real_tracker.py baseline used for the earlier mot17_test_submission.zip",
        "fusion_checkpoint": "fusion_attention.pt (trained on full MOT17-train, 150019 params)",
        "atl_checkpoint": "atl_paper_faithful.pt",
        "note": "Real predictions on the real, official MOT17 test images (no ground truth "
                "available locally). Only the tracker/association stage differs from "
                "mot17_test_submission.zip -- detections, embeddings and ATL thresholds are identical.",
    }
    with open(out_dir.parent / "mot17_test_submission_v2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated {len(list(out_dir.glob('*.txt')))} real prediction files "
          f"in {elapsed:.1f}s -> {zip_path}")
    print("NOTE: this archive has only the 21 TEST-sequence files. Run "
          "generate_mot17_train_predictions_v2.py to add the 21 TRAIN-sequence files required "
          "for the full 42-file CodaBench archive, then re-zip both together.")


if __name__ == "__main__":
    main()
