#!/usr/bin/env python3
"""
v2 (real GMC + real NSA, via real_tracker_v2.py) companion to
generate_mot20_submission.py. Same real feature extraction (reused
MOT17-trained fusion head + MOT20-specific real ATL encoder), only the
tracker/association stage differs.
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
import prepare_mot20_atl as m20
import generate_mot20_submission as g20
from real_tracker_v2 import run_tracker_v2_streamed
from atl_paper_faithful import ATLPaperFaithful
import verify_official_trackers as vo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("~/msfp/mot20_submission_v2").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading REUSED MOT17-trained MSFP fusion head...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [m20.LAYER_CHANNELS_V8M[n] for n in m20.LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    print("Loading MOT20-specific real ATL encoder...")
    atl_ckpt = torch.load(Path("~/msfp/checkpoints/atl_paper_faithful_mot20.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = m20.HookedBackbone(device)

    t0 = time.time()
    predicted_thresholds = {}
    train_seq_results = {}
    for seq_name, seq_path in g20.ALL_SEQUENCES.items():
        seq_dir = Path(seq_path)
        print(f"[{seq_name}] extracting real public-detection features + real ATL scene features...")
        records, n_frames, scene_feat = m20.extract_sequence(seq_dir, backbone, device)
        n_dets = sum(len(v) for v in records.values())

        with torch.no_grad():
            tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())
        predicted_thresholds[seq_name] = tau

        fused_by_frame = {f: m20.fuse_features(r, fusion, device) for f, r in records.items()}
        # tau_l=0.0 (genuinely inclusive of exact conf=0): MOT20's public detections use a
        # strictly binary {0,1} confidence convention (confirmed by direct inspection of every
        # train+test det.txt), so any nonzero floor silently excludes the vast majority of
        # detections on MOT20-03/04/05/06/08 from both association stages, contradicting
        # ByteTrack's own "associate every detection box" design. Verified via
        # mot20_zero_floor_test.py: real +5.79 HOTA / +44% relative gain (13.03 -> 18.82) on
        # the real train_half/val_half protocol from this fix alone.
        lines = run_tracker_v2_streamed(seq_dir / "img1", records, fused_by_frame, tau, n_frames,
                                         frame_start=1, frame_offset=0, tau_l_ratio=0.0, tau_l_floor=0.0,
                                         use_gmc=True, use_nsa=True)

        out_file = out_dir / f"{seq_name}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"  {n_frames} frames, {n_dets} real detections, ATL tau={tau:.3f}, "
              f"{len(lines)} output lines -> {seq_name}.txt")

        if seq_name in g20.TRAIN_SEQS:
            train_seq_results[seq_name] = {"n_frames": n_frames, "n_dets": n_dets, "tau": tau}

    elapsed = time.time() - t0

    print("\nComputing real local HOTA on MOT20-train sequences (not an official score)...")
    mot20_train_root = Path("/nas/Dataset/MOT/MOT20/train")
    workdir = Path("~/msfp/trackeval_mot20_train_full_v2").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    train_list = sorted(g20.TRAIN_SEQS)
    splits_zero = {s: 0 for s in train_list}
    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot20_train_root, train_list,
                                                          "mot20trainfullv2", splits_zero)
    seq_info_full = {s: train_seq_results[s]["n_frames"] for s in train_list}
    for s in train_list:
        src = (out_dir / f"{s}.txt").read_text(encoding="utf-8")
        out_file = trackers_folder / "mot20trainfullv2" / "data" / f"{s}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(src, encoding="utf-8")
    hota = vo.evaluate_hota(gt_folder, trackers_folder, train_list, "mot20trainfullv2", seq_info_full)
    mean_hota = float(np.mean(list(hota.values())))
    print(f"Real local HOTA (MOT20-train, v2 GMC+NSA pipeline): mean={mean_hota:.2f}")
    for s, v in hota.items():
        print(f"  {s}: {v:.2f}")

    zip_path = out_dir.parent / "mot20_submission_v2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(out_dir.glob("*.txt")):
            zf.write(txt_file, arcname=txt_file.name)

    meta = {
        "predicted_atl_thresholds": predicted_thresholds,
        "real_local_hota_train": hota,
        "mean_real_local_hota_train": mean_hota,
        "previous_v1_mean_for_cross_check": 14.21,
        "generation_time_seconds": elapsed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "tracker": "real_tracker_v2.ByteTrackStyleTrackerV2 (real ECC-based GMC + real NSA Kalman)",
        "fusion_checkpoint": "fusion_attention.pt (REUSED from MOT17, not retrained)",
        "atl_checkpoint": "atl_paper_faithful_mot20.pt (retrained specifically on MOT20, per paper protocol)",
        "note": "MOT20-01/02/03/05 are train sequences (GT public, not scored by server, included only "
                "to complete the required 8-file archive). MOT20-04/06/07/08 are the real official test "
                "sequences with no local ground truth -- these are what the server scores.",
    }
    with open(out_dir.parent / "mot20_submission_v2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated {len(list(out_dir.glob('*.txt')))} real prediction files "
          f"in {elapsed:.1f}s -> {zip_path}")
    print("Ready for upload to the MOT20 CodaBench submission page.")


if __name__ == "__main__":
    main()
